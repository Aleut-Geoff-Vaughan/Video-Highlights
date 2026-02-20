from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from ..auth import UserContext, require_roles
from ..database import get_session
from ..models import Match
from ..schemas import AgentExplainRequest, AgentQueryRequest, AgentQueryResponse
from ..services.llm_agent import agent_service
from ..tenant import TenantContext, get_tenant_context

router = APIRouter(tags=["agent"])


def _ensure_match(session: Session, match_id: str, tenant_id: str) -> None:
    match = session.get(Match, match_id)
    if not match or match.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")


@router.post("/matches/{match_id}/agent/query", response_model=AgentQueryResponse)
def agent_query(
    match_id: str,
    payload: AgentQueryRequest,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> AgentQueryResponse:
    _ensure_match(session, match_id, tenant.tenant_id)
    result = agent_service.query_match(
        session=session,
        tenant_id=tenant.tenant_id,
        match_id=match_id,
        query=payload.query,
        limit=payload.include_event_limit,
    )
    return AgentQueryResponse(**result)


@router.post("/matches/{match_id}/agent/explain/{event_id}", response_model=AgentQueryResponse)
def agent_explain_event(
    match_id: str,
    event_id: str,
    payload: AgentExplainRequest,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> AgentQueryResponse:
    _ensure_match(session, match_id, tenant.tenant_id)
    result = agent_service.explain_event(
        session=session,
        tenant_id=tenant.tenant_id,
        match_id=match_id,
        event_id=event_id,
        question=payload.question,
    )
    return AgentQueryResponse(**result)
