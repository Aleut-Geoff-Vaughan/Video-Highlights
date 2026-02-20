from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlmodel import Session, select

from ..database import get_session
from ..models import Event, EventFeedback, Match, ProcessingJob, TenantMembership, TrainingRun, UserAccount
from ..schemas import (
    MatchRead,
    TenantAdminSummaryRead,
    TenantAdminUserCreate,
    TenantAdminUserPatch,
    TenantMembershipPatch,
    TenantUserRead,
)
from ..serializers import match_to_read
from ..tenant import TenantContext, require_tenant_admin
from ..utils import utcnow

router = APIRouter(prefix="/admin/tenant", tags=["admin-tenant"])


def _tenant_count(session: Session, table_model, tenant_id: str) -> int:
    value = session.exec(
        select(func.count()).select_from(table_model).where(getattr(table_model, "tenant_id") == tenant_id)
    ).one()
    if isinstance(value, tuple):
        value = value[0]
    return int(value or 0)


def _user_to_tenant_view(membership: TenantMembership, user: UserAccount) -> TenantUserRead:
    return TenantUserRead(
        membership_id=membership.id,
        tenant_id=membership.tenant_id,
        user_id=user.id,
        email=user.email,
        display_name=user.display_name,
        user_status=user.status,
        role=membership.role,
        membership_status=membership.status,
        is_global_admin=user.is_global_admin,
        created_at=membership.created_at,
        updated_at=max(user.updated_at, membership.updated_at),
    )


@router.get("/summary", response_model=TenantAdminSummaryRead)
def get_tenant_summary(
    tenant: TenantContext = Depends(require_tenant_admin),
    session: Session = Depends(get_session),
) -> TenantAdminSummaryRead:
    membership_count = _tenant_count(session, TenantMembership, tenant.tenant_id)
    return TenantAdminSummaryRead(
        tenant_id=tenant.tenant_id,
        tenant_slug=tenant.tenant_slug,
        tenant_name=tenant.tenant_name,
        user_count=membership_count,
        membership_count=membership_count,
        match_count=_tenant_count(session, Match, tenant.tenant_id),
        job_count=_tenant_count(session, ProcessingJob, tenant.tenant_id),
        event_count=_tenant_count(session, Event, tenant.tenant_id),
        feedback_count=_tenant_count(session, EventFeedback, tenant.tenant_id),
        training_run_count=_tenant_count(session, TrainingRun, tenant.tenant_id),
    )


@router.get("/users", response_model=List[TenantUserRead])
def list_tenant_users(
    status: Optional[str] = Query(default=None),
    tenant: TenantContext = Depends(require_tenant_admin),
    session: Session = Depends(get_session),
) -> List[TenantUserRead]:
    stmt = select(TenantMembership).where(TenantMembership.tenant_id == tenant.tenant_id)
    if status:
        stmt = stmt.where(TenantMembership.status == status)
    stmt = stmt.order_by(TenantMembership.created_at.desc())
    memberships = list(session.exec(stmt))
    if not memberships:
        return []
    user_ids = [item.user_id for item in memberships]
    users = {
        user.id: user
        for user in session.exec(select(UserAccount).where(UserAccount.id.in_(user_ids)))
    }
    result: List[TenantUserRead] = []
    for membership in memberships:
        user = users.get(membership.user_id)
        if not user:
            continue
        result.append(_user_to_tenant_view(membership, user))
    return result


@router.post("/users", response_model=TenantUserRead, status_code=201)
def create_tenant_user(
    payload: TenantAdminUserCreate,
    tenant: TenantContext = Depends(require_tenant_admin),
    session: Session = Depends(get_session),
) -> TenantUserRead:
    user = session.get(UserAccount, payload.user_id)
    if not user:
        user = UserAccount(
            id=payload.user_id,
            email=payload.email,
            display_name=payload.display_name or payload.user_id,
            status=payload.user_status,
            is_global_admin=False,
            metadata_json=payload.user_metadata,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
    else:
        if payload.email is not None:
            user.email = payload.email
        if payload.display_name is not None:
            user.display_name = payload.display_name
        user.status = payload.user_status
        if payload.user_metadata:
            user.metadata_json = payload.user_metadata
        user.updated_at = utcnow()
        session.add(user)
        session.commit()
        session.refresh(user)

    membership = session.exec(
        select(TenantMembership)
        .where(TenantMembership.tenant_id == tenant.tenant_id)
        .where(TenantMembership.user_id == user.id)
    ).first()
    if membership:
        membership.role = payload.role
        membership.status = payload.membership_status
        membership.metadata_json = payload.membership_metadata
        membership.updated_at = utcnow()
        session.add(membership)
        session.commit()
        session.refresh(membership)
    else:
        membership = TenantMembership(
            tenant_id=tenant.tenant_id,
            user_id=user.id,
            role=payload.role,
            status=payload.membership_status,
            metadata_json=payload.membership_metadata,
        )
        session.add(membership)
        session.commit()
        session.refresh(membership)

    return _user_to_tenant_view(membership, user)


@router.patch("/users/{user_id}", response_model=TenantUserRead)
def patch_tenant_user(
    user_id: str,
    payload: TenantAdminUserPatch,
    tenant: TenantContext = Depends(require_tenant_admin),
    session: Session = Depends(get_session),
) -> TenantUserRead:
    user = session.get(UserAccount, user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")
    membership = session.exec(
        select(TenantMembership)
        .where(TenantMembership.tenant_id == tenant.tenant_id)
        .where(TenantMembership.user_id == user_id)
    ).first()
    if not membership:
        raise HTTPException(status_code=404, detail=f"Membership not found for tenant/user: {tenant.tenant_id}/{user_id}")

    if payload.email is not None:
        user.email = payload.email
    if payload.display_name is not None:
        user.display_name = payload.display_name
    if payload.user_status is not None:
        user.status = payload.user_status
    if payload.user_metadata is not None:
        user.metadata_json = payload.user_metadata
    user.updated_at = utcnow()
    session.add(user)

    if payload.role is not None:
        membership.role = payload.role
    if payload.membership_status is not None:
        membership.status = payload.membership_status
    if payload.membership_metadata is not None:
        membership.metadata_json = payload.membership_metadata
    membership.updated_at = utcnow()
    session.add(membership)

    session.commit()
    session.refresh(user)
    session.refresh(membership)
    return _user_to_tenant_view(membership, user)


@router.patch("/memberships/{membership_id}", response_model=TenantUserRead)
def patch_tenant_membership(
    membership_id: str,
    payload: TenantMembershipPatch,
    tenant: TenantContext = Depends(require_tenant_admin),
    session: Session = Depends(get_session),
) -> TenantUserRead:
    membership = session.get(TenantMembership, membership_id)
    if not membership or membership.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Membership not found: {membership_id}")
    user = session.get(UserAccount, membership.user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User not found: {membership.user_id}")

    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        if key == "metadata":
            membership.metadata_json = value or {}
        else:
            setattr(membership, key, value)
    membership.updated_at = utcnow()
    session.add(membership)
    session.commit()
    session.refresh(membership)
    return _user_to_tenant_view(membership, user)


@router.get("/matches", response_model=List[MatchRead])
def list_tenant_matches(
    limit: int = Query(default=100, ge=1, le=500),
    tenant: TenantContext = Depends(require_tenant_admin),
    session: Session = Depends(get_session),
) -> List[MatchRead]:
    rows = list(
        session.exec(
            select(Match)
            .where(Match.tenant_id == tenant.tenant_id)
            .order_by(Match.created_at.desc())
            .limit(limit)
        )
    )
    return [match_to_read(item) for item in rows]


@router.get("/inventory", response_model=Dict[str, int])
def get_tenant_inventory(
    tenant: TenantContext = Depends(require_tenant_admin),
    session: Session = Depends(get_session),
) -> Dict[str, int]:
    return {
        "users": _tenant_count(session, TenantMembership, tenant.tenant_id),
        "matches": _tenant_count(session, Match, tenant.tenant_id),
        "jobs": _tenant_count(session, ProcessingJob, tenant.tenant_id),
        "events": _tenant_count(session, Event, tenant.tenant_id),
        "feedback": _tenant_count(session, EventFeedback, tenant.tenant_id),
        "training_runs": _tenant_count(session, TrainingRun, tenant.tenant_id),
    }
