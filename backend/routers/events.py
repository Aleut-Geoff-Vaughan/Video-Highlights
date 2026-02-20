from __future__ import annotations

from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from ..auth import UserContext, require_roles
from ..database import get_session
from ..models import Event, Match
from ..schemas import EventPatch, EventRead, EventUpsert
from ..serializers import event_to_read
from ..tenant import TenantContext, get_tenant_context
from ..utils import decode_cursor, encode_cursor, utcnow

router = APIRouter(tags=["events"])


def _ensure_match(session: Session, match_id: str, tenant_id: str) -> Match:
    match = session.get(Match, match_id)
    if not match or match.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")
    return match


@router.get("/matches/{match_id}/events", response_model=Dict[str, object])
def list_events(
    match_id: str,
    event_type: Optional[str] = Query(default=None),
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
    event = session.exec(
        select(Event)
        .where(Event.match_id == match_id)
        .where(Event.tenant_id == tenant.tenant_id)
        .where(Event.id == event_id)
    ).first()
    if not event:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")
    return event_to_read(event)


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
