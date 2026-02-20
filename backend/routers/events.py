from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlmodel import Session, select

from ..auth import UserContext, require_roles
from ..config import settings
from ..database import get_session
from ..models import Event, Match
from ..schemas import (
    EventClipRead,
    EventClipRequest,
    EventPatch,
    EventRead,
    EventUpsert,
    HighlightExportRead,
    HighlightExportRequest,
)
from ..serializers import event_to_read
from ..services.event_clip_renderer import concat_clips_ffmpeg, render_clip_ffmpeg
from ..services.storage import get_storage_backend
from ..tenant import TenantContext, get_tenant_context
from ..utils import decode_cursor, encode_cursor, ensure_dir, generate_id, utcnow

router = APIRouter(tags=["events"])


def _ensure_match(session: Session, match_id: str, tenant_id: str) -> Match:
    match = session.get(Match, match_id)
    if not match or match.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")
    return match


def _get_event_or_404(session: Session, match_id: str, event_id: str, tenant_id: str) -> Event:
    event = session.exec(
        select(Event)
        .where(Event.match_id == match_id)
        .where(Event.tenant_id == tenant_id)
        .where(Event.id == event_id)
    ).first()
    if not event:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")
    return event


def _clip_window_ms(event: Event, payload: EventClipRequest) -> tuple[int, int]:
    if payload.anchor == "occurred_at":
        anchor_s = float(event.occurred_at_ms) / 1000.0
        start_s = anchor_s - float(payload.pre_seconds)
        end_s = anchor_s + float(payload.post_seconds)
    else:
        start_s = (float(event.start_ms) / 1000.0) - float(payload.pre_seconds)
        end_s = (float(event.end_ms) / 1000.0) + float(payload.post_seconds)

    start_s = max(0.0, start_s)
    if end_s <= start_s:
        end_s = start_s + max(1.0, float(payload.pre_seconds) + float(payload.post_seconds))
    start_ms = max(0, int(round(start_s * 1000.0)))
    end_ms = max(start_ms + 1, int(round(end_s * 1000.0)))
    return start_ms, end_ms


def _find_cached_clip(match: Match, signature: str) -> tuple[Dict[str, object], Dict[str, object]] | None:
    metadata = dict(match.metadata_json or {})
    generated = list(metadata.get("generated_clips", []) or [])
    assets = list(metadata.get("assets", []) or [])

    clip_entry = next((item for item in generated if str(item.get("signature", "")) == signature), None)
    if not clip_entry:
        return None
    asset_id = str(clip_entry.get("asset_id", "")).strip()
    if not asset_id:
        return None
    asset_entry = next((item for item in assets if str(item.get("asset_id", "")) == asset_id), None)
    if not asset_entry:
        return None
    return clip_entry, asset_entry


def _resolve_source_video_path(match: Match) -> str:
    candidates = [str(match.source_video_path or "").strip()]
    metadata = dict(match.metadata_json or {})
    assets = list(metadata.get("assets", []) or [])
    for asset in assets:
        path = str(asset.get("path", "")).strip()
        if path:
            candidates.append(path)
    for path in candidates:
        if path and os.path.exists(path):
            return path
    for path in candidates:
        if path:
            return path
    return ""


@router.get("/matches/{match_id}/events", response_model=Dict[str, object])
def list_events(
    match_id: str,
    event_type: Optional[str] = Query(default=None),
    job_id: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    team_id: Optional[str] = Query(default=None),
    player_id: Optional[str] = Query(default=None),
    period: Optional[str] = Query(default=None),
    min_confidence: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    from_ms: Optional[int] = Query(default=None, ge=0),
    to_ms: Optional[int] = Query(default=None, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    cursor: Optional[str] = Query(default=None),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    _ensure_match(session, match_id, tenant.tenant_id)
    offset = decode_cursor(cursor)

    stmt = select(Event).where(Event.match_id == match_id).where(Event.tenant_id == tenant.tenant_id)
    if event_type:
        stmt = stmt.where(Event.event_type == event_type)
    if job_id:
        stmt = stmt.where(Event.job_id == job_id)
    if status:
        stmt = stmt.where(Event.status == status)
    if team_id:
        stmt = stmt.where(Event.team_id == team_id)
    if player_id:
        stmt = stmt.where(Event.player_id == player_id)
    if period:
        stmt = stmt.where(Event.period == period)
    if min_confidence is not None:
        stmt = stmt.where(Event.confidence >= min_confidence)
    if from_ms is not None:
        stmt = stmt.where(Event.occurred_at_ms >= from_ms)
    if to_ms is not None:
        stmt = stmt.where(Event.occurred_at_ms <= to_ms)

    stmt = stmt.order_by(Event.occurred_at_ms.desc()).offset(offset).limit(limit + 1)
    rows = list(session.exec(stmt))
    has_more = len(rows) > limit
    items = [event_to_read(evt) for evt in rows[:limit]]
    next_cursor = encode_cursor(offset + limit) if has_more else None
    return {"items": [item.model_dump() for item in items], "next_cursor": next_cursor}


@router.get("/matches/{match_id}/events/{event_id}", response_model=EventRead)
def get_event(
    match_id: str,
    event_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> EventRead:
    _ensure_match(session, match_id, tenant.tenant_id)
    event = _get_event_or_404(session, match_id, event_id, tenant.tenant_id)
    return event_to_read(event)


@router.post("/matches/{match_id}/events/{event_id}/clip-on-demand", response_model=EventClipRead)
def render_event_clip_on_demand(
    match_id: str,
    event_id: str,
    payload: EventClipRequest = Body(default_factory=EventClipRequest),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> EventClipRead:
    match = _ensure_match(session, match_id, tenant.tenant_id)
    event = _get_event_or_404(session, match_id, event_id, tenant.tenant_id)
    source_video = _resolve_source_video_path(match)
    if not source_video:
        raise HTTPException(status_code=400, detail="Match has no source_video_path")
    if not os.path.exists(source_video):
        raise HTTPException(status_code=400, detail=f"Source video path not found: {source_video}")

    start_ms, end_ms = _clip_window_ms(event, payload)
    signature = (
        f"{event.id}:{start_ms}:{end_ms}:{int(payload.include_audio)}:"
        f"{int(payload.prefer_gpu)}:{payload.anchor}"
    )

    storage = get_storage_backend()
    cached = _find_cached_clip(match, signature)
    if cached and not payload.force_rebuild:
        clip_meta, asset_meta = cached
        path = str(asset_meta.get("path", ""))
        return EventClipRead(
            clip_id=str(clip_meta.get("clip_id", generate_id("eclip"))),
            match_id=match_id,
            event_id=event_id,
            asset_id=str(asset_meta.get("asset_id", "")),
            path=path,
            download_url=storage.get_download_url(path, expires_seconds=payload.expires_seconds),
            start_ms=int(clip_meta.get("start_ms", start_ms)),
            end_ms=int(clip_meta.get("end_ms", end_ms)),
            duration_ms=max(1, int(clip_meta.get("duration_ms", end_ms - start_ms))),
            include_audio=bool(clip_meta.get("include_audio", payload.include_audio)),
            anchor=str(clip_meta.get("anchor", payload.anchor)),
            reused_existing=True,
        )

    start_s = float(start_ms) / 1000.0
    end_s = float(end_ms) / 1000.0
    temp_dir = ensure_dir(os.path.join(settings.output_root, "event_clip_tmp", match_id))
    temp_filename = f"{event_id}_{start_ms}_{end_ms}_{'a' if payload.include_audio else 'na'}.mp4"
    temp_path = os.path.join(temp_dir, temp_filename)
    try:
        render_clip_ffmpeg(
            video_path=source_video,
            output_path=temp_path,
            start_seconds=start_s,
            end_seconds=end_s,
            include_audio=payload.include_audio,
            prefer_gpu=payload.prefer_gpu,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to render event clip: {exc}") from exc

    with open(temp_path, "rb") as stream:
        stored = storage.save_file(
            stream=stream,
            key_prefix=f"{match_id}/event_clips",
            filename=f"{event_id}_{start_ms}_{end_ms}.mp4",
        )
    try:
        Path(temp_path).unlink(missing_ok=True)
    except Exception:
        pass

    metadata = dict(match.metadata_json or {})
    assets = list(metadata.get("assets", []) or [])
    generated = list(metadata.get("generated_clips", []) or [])
    clip_id = generate_id("eclip")
    created_at = utcnow().isoformat()
    asset_entry = {
        "asset_id": stored.object_id,
        "filename": f"{event_id}_{start_ms}_{end_ms}.mp4",
        "path": stored.path,
        "size_bytes": stored.size_bytes,
        "storage_backend": stored.backend,
        "uploaded_at": created_at,
        "kind": "event_clip",
        "event_id": event_id,
    }
    clip_entry = {
        "clip_id": clip_id,
        "signature": signature,
        "event_id": event_id,
        "asset_id": stored.object_id,
        "path": stored.path,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": max(1, end_ms - start_ms),
        "include_audio": payload.include_audio,
        "anchor": payload.anchor,
        "created_at": created_at,
    }
    assets.append(asset_entry)
    generated.append(clip_entry)
    metadata["assets"] = assets[-5000:]
    metadata["generated_clips"] = generated[-5000:]
    match.metadata_json = metadata
    match.updated_at = utcnow()
    session.add(match)
    session.commit()

    return EventClipRead(
        clip_id=clip_id,
        match_id=match_id,
        event_id=event_id,
        asset_id=stored.object_id,
        path=stored.path,
        download_url=storage.get_download_url(stored.path, expires_seconds=payload.expires_seconds),
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=max(1, end_ms - start_ms),
        include_audio=payload.include_audio,
        anchor=payload.anchor,
        reused_existing=False,
    )


@router.post("/matches/{match_id}/exports/highlights", response_model=HighlightExportRead)
def export_selected_highlights(
    match_id: str,
    payload: HighlightExportRequest,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> HighlightExportRead:
    match = _ensure_match(session, match_id, tenant.tenant_id)
    source_video = _resolve_source_video_path(match)
    if not source_video:
        raise HTTPException(status_code=400, detail="Match has no source video")
    if not os.path.exists(source_video):
        raise HTTPException(status_code=400, detail=f"Source video path not found: {source_video}")

    ordered_event_ids: list[str] = []
    seen: set[str] = set()
    for raw in payload.event_ids:
        event_id = str(raw).strip()
        if event_id and event_id not in seen:
            ordered_event_ids.append(event_id)
            seen.add(event_id)
    if not ordered_event_ids:
        raise HTTPException(status_code=400, detail="At least one event_id is required")

    events = [_get_event_or_404(session, match_id, event_id, tenant.tenant_id) for event_id in ordered_event_ids]
    clip_payload = EventClipRequest(
        pre_seconds=payload.pre_seconds,
        post_seconds=payload.post_seconds,
        anchor=payload.anchor,
        include_audio=payload.include_audio,
        prefer_gpu=payload.prefer_gpu,
        force_rebuild=True,
        expires_seconds=payload.expires_seconds,
    )

    export_id = generate_id("export")
    temp_root = ensure_dir(os.path.join(settings.output_root, "highlight_export_tmp", match_id, export_id))
    clip_paths: list[str] = []
    duration_ms = 0
    try:
        for index, event in enumerate(events, start=1):
            start_ms, end_ms = _clip_window_ms(event, clip_payload)
            clip_path = os.path.join(temp_root, f"part_{index:04d}.mp4")
            render_clip_ffmpeg(
                video_path=source_video,
                output_path=clip_path,
                start_seconds=float(start_ms) / 1000.0,
                end_seconds=float(end_ms) / 1000.0,
                include_audio=payload.include_audio,
                prefer_gpu=payload.prefer_gpu,
            )
            clip_paths.append(clip_path)
            duration_ms += max(1, end_ms - start_ms)

        export_filename = f"{export_id}.mp4"
        export_temp_path = os.path.join(temp_root, export_filename)
        concat_clips_ffmpeg(clip_paths, export_temp_path, include_audio=payload.include_audio)

        storage = get_storage_backend()
        with open(export_temp_path, "rb") as stream:
            stored = storage.save_file(
                stream=stream,
                key_prefix=f"{match_id}/exports",
                filename=export_filename,
            )

        created_at = utcnow().isoformat()
        metadata = dict(match.metadata_json or {})
        assets = list(metadata.get("assets", []) or [])
        exports = list(metadata.get("highlight_exports", []) or [])
        asset_entry = {
            "asset_id": stored.object_id,
            "filename": export_filename,
            "path": stored.path,
            "size_bytes": stored.size_bytes,
            "storage_backend": stored.backend,
            "uploaded_at": created_at,
            "kind": "highlight_export",
            "export_id": export_id,
        }
        export_entry = {
            "export_id": export_id,
            "title": payload.title or "Selected Highlights",
            "event_ids": ordered_event_ids,
            "clip_count": len(ordered_event_ids),
            "duration_ms": duration_ms,
            "asset_id": stored.object_id,
            "path": stored.path,
            "created_at": created_at,
        }
        assets.append(asset_entry)
        exports.append(export_entry)
        metadata["assets"] = assets[-5000:]
        metadata["highlight_exports"] = exports[-5000:]
        match.metadata_json = metadata
        match.updated_at = utcnow()
        session.add(match)
        session.commit()

        return HighlightExportRead(
            export_id=export_id,
            match_id=match_id,
            event_ids=ordered_event_ids,
            clip_count=len(ordered_event_ids),
            asset_id=stored.object_id,
            path=stored.path,
            download_url=storage.get_download_url(stored.path, expires_seconds=payload.expires_seconds),
            duration_ms=duration_ms,
            created_at=created_at,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to export selected highlights: {exc}") from exc
    finally:
        try:
            shutil.rmtree(temp_root, ignore_errors=True)
        except Exception:
            pass


@router.put("/matches/{match_id}/events/{event_id}", response_model=EventRead)
def upsert_event(
    match_id: str,
    event_id: str,
    payload: EventUpsert,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> EventRead:
    match = _ensure_match(session, match_id, tenant.tenant_id)
    existing = session.get(Event, event_id)
    if existing and (existing.match_id != match_id or existing.tenant_id != tenant.tenant_id):
        raise HTTPException(
            status_code=409,
            detail=f"Event ID already exists on a different match: {event_id}",
        )
    event = existing

    if not event:
        event = Event(
            id=event_id,
            tenant_id=match.tenant_id,
            match_id=match_id,
            job_id=payload.job_id,
            event_type=payload.event_type,
            status=payload.status,
            confidence=payload.confidence,
            period=payload.period,
            occurred_at_ms=payload.occurred_at_ms,
            start_ms=payload.start_ms,
            end_ms=payload.end_ms,
            frame_index=payload.frame_index,
            team_id=payload.team_id,
            player_id=payload.player_id,
            jersey_number=payload.jersey_number,
            source_json=payload.source.model_dump(),
            location_json=payload.location.model_dump(),
            participants_json=[participant.model_dump() for participant in payload.participants],
            evidence_json=payload.evidence.model_dump(),
            explanations_json=[exp.model_dump() for exp in payload.explanations],
        )
    else:
        event.job_id = payload.job_id
        event.event_type = payload.event_type
        event.status = payload.status
        event.confidence = payload.confidence
        event.period = payload.period
        event.occurred_at_ms = payload.occurred_at_ms
        event.start_ms = payload.start_ms
        event.end_ms = payload.end_ms
        event.frame_index = payload.frame_index
        event.team_id = payload.team_id
        event.player_id = payload.player_id
        event.jersey_number = payload.jersey_number
        event.source_json = payload.source.model_dump()
        event.location_json = payload.location.model_dump()
        event.participants_json = [participant.model_dump() for participant in payload.participants]
        event.evidence_json = payload.evidence.model_dump()
        event.explanations_json = [exp.model_dump() for exp in payload.explanations]
        event.updated_at = utcnow()

    session.add(event)
    session.commit()
    session.refresh(event)
    return event_to_read(event)


@router.patch("/matches/{match_id}/events/{event_id}", response_model=EventRead)
def patch_event(
    match_id: str,
    event_id: str,
    payload: EventPatch,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> EventRead:
    _ensure_match(session, match_id, tenant.tenant_id)
    event = session.exec(
        select(Event)
        .where(Event.match_id == match_id)
        .where(Event.tenant_id == tenant.tenant_id)
        .where(Event.id == event_id)
    ).first()
    if not event:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")

    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        if key in {"source", "location", "evidence"} and value is not None:
            setattr(event, f"{key}_json", value)
        elif key in {"participants", "explanations"} and value is not None:
            setattr(event, f"{key}_json", value)
        else:
            setattr(event, key, value)

    if event.start_ms > event.occurred_at_ms or event.occurred_at_ms > event.end_ms:
        raise HTTPException(
            status_code=400,
            detail="Must satisfy start_ms <= occurred_at_ms <= end_ms",
        )

    event.updated_at = utcnow()
    session.add(event)
    session.commit()
    session.refresh(event)
    return event_to_read(event)
