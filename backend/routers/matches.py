from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlmodel import Session, select

from ..auth import UserContext, require_roles
from ..database import get_session
from ..models import Match
from ..schemas import MatchCreate, MatchPatch, MatchRead
from ..serializers import match_to_read
from ..services.storage import get_storage_backend
from ..tenant import TenantContext, get_tenant_context
from ..utils import decode_cursor, encode_cursor, utcnow

router = APIRouter(prefix="/matches", tags=["matches"])


@router.post("", response_model=MatchRead, status_code=201)
def create_match(
    payload: MatchCreate,
    session: Session = Depends(get_session),
    user: UserContext = Depends(require_roles("admin", "analyst", "coach")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> MatchRead:
    target_tenant_id = tenant.tenant_id
    if payload.tenant_id:
        if not user.is_global_admin and payload.tenant_id != tenant.tenant_id:
            raise HTTPException(status_code=403, detail="Cannot create a match in another tenant")
        target_tenant_id = payload.tenant_id

    match = Match(
        tenant_id=target_tenant_id,
        name=payload.name,
        home_team_name=payload.home_team_name,
        away_team_name=payload.away_team_name,
        match_date=payload.match_date,
        source_video_path=payload.source_video_path,
        metadata_json=payload.metadata,
    )
    session.add(match)
    session.commit()
    session.refresh(match)
    return match_to_read(match)


@router.get("", response_model=Dict[str, object])
def list_matches(
    limit: int = Query(default=100, ge=1, le=500),
    cursor: str | None = Query(default=None),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    offset = decode_cursor(cursor)
    stmt = (
        select(Match)
        .where(Match.tenant_id == tenant.tenant_id)
        .order_by(Match.created_at.desc())
        .offset(offset)
        .limit(limit + 1)
    )
    rows = list(session.exec(stmt))
    has_more = len(rows) > limit
    items = [match_to_read(match) for match in rows[:limit]]
    next_cursor = encode_cursor(offset + limit) if has_more else None
    return {"items": [item.model_dump() for item in items], "next_cursor": next_cursor}


@router.get("/{match_id}", response_model=MatchRead)
def get_match(
    match_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> MatchRead:
    match = session.get(Match, match_id)
    if not match or match.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")
    return match_to_read(match)


@router.patch("/{match_id}", response_model=MatchRead)
def update_match(
    match_id: str,
    payload: MatchPatch,
    session: Session = Depends(get_session),
    user: UserContext = Depends(require_roles("admin", "analyst", "coach", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> MatchRead:
    match = session.get(Match, match_id)
    if not match or match.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")

    data = payload.model_dump(exclude_unset=True)
    if "tenant_id" in data:
        requested = data["tenant_id"]
        if requested and requested != match.tenant_id:
            if not user.is_global_admin:
                raise HTTPException(status_code=403, detail="Cannot move a match across tenants")
            match.tenant_id = requested
        data.pop("tenant_id", None)
    for key, value in data.items():
        if key == "metadata":
            match.metadata_json = value
        else:
            setattr(match, key, value)
    match.updated_at = utcnow()
    session.add(match)
    session.commit()
    session.refresh(match)
    return match_to_read(match)


@router.post("/{match_id}/assets/upload", response_model=Dict[str, Any], status_code=201)
def upload_match_asset(
    match_id: str,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, Any]:
    match = session.get(Match, match_id)
    if not match or match.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")

    safe_name = file.filename or "upload.bin"
    storage = get_storage_backend()
    stored = storage.save_file(file.file, key_prefix=match_id, filename=safe_name)
    file.file.close()

    assets = list((match.metadata_json or {}).get("assets", []))
    assets.append(
        {
            "asset_id": stored.object_id,
            "filename": safe_name,
            "path": stored.path,
            "size_bytes": stored.size_bytes,
            "storage_backend": stored.backend,
            "uploaded_at": utcnow().isoformat(),
        }
    )
    metadata = dict(match.metadata_json or {})
    metadata["assets"] = assets
    match.metadata_json = metadata

    # Optional convenience: first upload can become source video path.
    if not match.source_video_path:
        match.source_video_path = stored.path

    match.updated_at = utcnow()
    session.add(match)
    session.commit()

    return {
        "asset_id": stored.object_id,
        "match_id": match_id,
        "filename": safe_name,
        "path": stored.path,
        "size_bytes": stored.size_bytes,
        "storage_backend": stored.backend,
    }


@router.get("/{match_id}/assets/{asset_id}/download-url", response_model=Dict[str, Any])
def get_asset_download_url(
    match_id: str,
    asset_id: str,
    expires_seconds: int = Query(default=3600, ge=60, le=86400),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, Any]:
    match = session.get(Match, match_id)
    if not match or match.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")

    assets = list((match.metadata_json or {}).get("assets", []))
    asset = next((item for item in assets if item.get("asset_id") == asset_id), None)
    if not asset:
        raise HTTPException(status_code=404, detail=f"Asset not found: {asset_id}")

    storage = get_storage_backend()
    stored_path = str(asset.get("path", ""))
    url = storage.get_download_url(stored_path=stored_path, expires_seconds=expires_seconds)
    return {
        "match_id": match_id,
        "asset_id": asset_id,
        "storage_backend": asset.get("storage_backend", storage.__class__.__name__),
        "path": stored_path,
        "download_url": url,
        "expires_seconds": expires_seconds,
    }
