from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlmodel import Session, select

from ..database import get_session
from ..models import (
    Event,
    EventFeedback,
    Match,
    ModelPromotionDecision,
    ModelVersion,
    ProcessingJob,
    Tenant,
    TenantMembership,
    TrainingFeedbackBatch,
    TrainingRun,
    UserAccount,
)
from ..schemas import (
    GlobalAdminSummaryRead,
    TenantCreate,
    TenantMembershipCreate,
    TenantMembershipPatch,
    TenantMembershipRead,
    TenantPatch,
    TenantRead,
    UserAccountCreate,
    UserAccountPatch,
    UserAccountRead,
)
from ..serializers import membership_to_read, tenant_to_read, user_to_read
from ..tenant import ensure_default_tenant, require_global_admin
from ..utils import utcnow

router = APIRouter(prefix="/admin/global", tags=["admin-global"])


def _count_rows(session: Session, table_model) -> int:
    value = session.exec(select(func.count()).select_from(table_model)).one()
    if isinstance(value, tuple):
        value = value[0]
    return int(value or 0)


def _get_tenant_or_404(session: Session, tenant_id: str) -> Tenant:
    tenant = session.get(Tenant, tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail=f"Tenant not found: {tenant_id}")
    return tenant


def _get_user_or_404(session: Session, user_id: str) -> UserAccount:
    user = session.get(UserAccount, user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")
    return user


@router.get("/summary", response_model=GlobalAdminSummaryRead)
def get_global_summary(
    session: Session = Depends(get_session),
    _=Depends(require_global_admin),
) -> GlobalAdminSummaryRead:
    ensure_default_tenant(session)
    return GlobalAdminSummaryRead(
        tenant_count=_count_rows(session, Tenant),
        user_count=_count_rows(session, UserAccount),
        membership_count=_count_rows(session, TenantMembership),
        match_count=_count_rows(session, Match),
        job_count=_count_rows(session, ProcessingJob),
        event_count=_count_rows(session, Event),
        feedback_count=_count_rows(session, EventFeedback),
        training_run_count=_count_rows(session, TrainingRun),
    )


@router.get("/tenants", response_model=List[TenantRead])
def list_tenants(
    status: Optional[str] = Query(default=None),
    session: Session = Depends(get_session),
    _=Depends(require_global_admin),
) -> List[TenantRead]:
    ensure_default_tenant(session)
    stmt = select(Tenant)
    if status:
        stmt = stmt.where(Tenant.status == status)
    stmt = stmt.order_by(Tenant.created_at.desc())
    rows = list(session.exec(stmt))
    return [tenant_to_read(item) for item in rows]


@router.post("/tenants", response_model=TenantRead, status_code=201)
def create_tenant(
    payload: TenantCreate,
    session: Session = Depends(get_session),
    _=Depends(require_global_admin),
) -> TenantRead:
    existing = session.exec(select(Tenant).where(Tenant.slug == payload.slug)).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Tenant slug already exists: {payload.slug}")

    tenant = Tenant(
        slug=payload.slug,
        name=payload.name,
        status=payload.status,
        metadata_json=payload.metadata,
    )
    session.add(tenant)
    session.commit()
    session.refresh(tenant)
    return tenant_to_read(tenant)


@router.patch("/tenants/{tenant_id}", response_model=TenantRead)
def patch_tenant(
    tenant_id: str,
    payload: TenantPatch,
    session: Session = Depends(get_session),
    _=Depends(require_global_admin),
) -> TenantRead:
    tenant = _get_tenant_or_404(session, tenant_id)
    data = payload.model_dump(exclude_unset=True)
    if "slug" in data and data["slug"] and data["slug"] != tenant.slug:
        duplicate = session.exec(select(Tenant).where(Tenant.slug == data["slug"])).first()
        if duplicate:
            raise HTTPException(status_code=409, detail=f"Tenant slug already exists: {data['slug']}")

    for key, value in data.items():
        if key == "metadata":
            tenant.metadata_json = value or {}
        else:
            setattr(tenant, key, value)
    tenant.updated_at = utcnow()
    session.add(tenant)
    session.commit()
    session.refresh(tenant)
    return tenant_to_read(tenant)


@router.get("/users", response_model=List[UserAccountRead])
def list_users(
    status: Optional[str] = Query(default=None),
    session: Session = Depends(get_session),
    _=Depends(require_global_admin),
) -> List[UserAccountRead]:
    stmt = select(UserAccount)
    if status:
        stmt = stmt.where(UserAccount.status == status)
    stmt = stmt.order_by(UserAccount.created_at.desc())
    rows = list(session.exec(stmt))
    return [user_to_read(item) for item in rows]


@router.post("/users", response_model=UserAccountRead, status_code=201)
def create_user(
    payload: UserAccountCreate,
    session: Session = Depends(get_session),
    _=Depends(require_global_admin),
) -> UserAccountRead:
    if session.get(UserAccount, payload.user_id):
        raise HTTPException(status_code=409, detail=f"User already exists: {payload.user_id}")
    user = UserAccount(
        id=payload.user_id,
        email=payload.email,
        display_name=payload.display_name,
        status=payload.status,
        is_global_admin=payload.is_global_admin,
        metadata_json=payload.metadata,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user_to_read(user)


@router.patch("/users/{user_id}", response_model=UserAccountRead)
def patch_user(
    user_id: str,
    payload: UserAccountPatch,
    session: Session = Depends(get_session),
    _=Depends(require_global_admin),
) -> UserAccountRead:
    user = _get_user_or_404(session, user_id)
    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        if key == "metadata":
            user.metadata_json = value or {}
        else:
            setattr(user, key, value)
    user.updated_at = utcnow()
    session.add(user)
    session.commit()
    session.refresh(user)
    return user_to_read(user)


@router.get("/tenants/{tenant_id}/memberships", response_model=List[TenantMembershipRead])
def list_memberships(
    tenant_id: str,
    status: Optional[str] = Query(default=None),
    session: Session = Depends(get_session),
    _=Depends(require_global_admin),
) -> List[TenantMembershipRead]:
    _get_tenant_or_404(session, tenant_id)
    stmt = select(TenantMembership).where(TenantMembership.tenant_id == tenant_id)
    if status:
        stmt = stmt.where(TenantMembership.status == status)
    stmt = stmt.order_by(TenantMembership.created_at.desc())
    rows = list(session.exec(stmt))
    return [membership_to_read(item) for item in rows]


@router.post("/tenants/{tenant_id}/memberships", response_model=TenantMembershipRead, status_code=201)
def create_or_update_membership(
    tenant_id: str,
    payload: TenantMembershipCreate,
    session: Session = Depends(get_session),
    _=Depends(require_global_admin),
) -> TenantMembershipRead:
    _get_tenant_or_404(session, tenant_id)
    _get_user_or_404(session, payload.user_id)
    existing = session.exec(
        select(TenantMembership)
        .where(TenantMembership.tenant_id == tenant_id)
        .where(TenantMembership.user_id == payload.user_id)
    ).first()
    if existing:
        existing.role = payload.role
        existing.status = payload.status
        existing.metadata_json = payload.metadata
        existing.updated_at = utcnow()
        session.add(existing)
        session.commit()
        session.refresh(existing)
        return membership_to_read(existing)

    membership = TenantMembership(
        tenant_id=tenant_id,
        user_id=payload.user_id,
        role=payload.role,
        status=payload.status,
        metadata_json=payload.metadata,
    )
    session.add(membership)
    session.commit()
    session.refresh(membership)
    return membership_to_read(membership)


@router.patch("/memberships/{membership_id}", response_model=TenantMembershipRead)
def patch_membership(
    membership_id: str,
    payload: TenantMembershipPatch,
    session: Session = Depends(get_session),
    _=Depends(require_global_admin),
) -> TenantMembershipRead:
    membership = session.get(TenantMembership, membership_id)
    if not membership:
        raise HTTPException(status_code=404, detail=f"Membership not found: {membership_id}")
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
    return membership_to_read(membership)


@router.get("/inventory", response_model=Dict[str, int])
def get_global_inventory(
    session: Session = Depends(get_session),
    _=Depends(require_global_admin),
) -> Dict[str, int]:
    return {
        "tenants": _count_rows(session, Tenant),
        "users": _count_rows(session, UserAccount),
        "memberships": _count_rows(session, TenantMembership),
        "matches": _count_rows(session, Match),
        "jobs": _count_rows(session, ProcessingJob),
        "events": _count_rows(session, Event),
        "feedback": _count_rows(session, EventFeedback),
        "feedback_batches": _count_rows(session, TrainingFeedbackBatch),
        "training_runs": _count_rows(session, TrainingRun),
        "model_versions": _count_rows(session, ModelVersion),
        "promotion_decisions": _count_rows(session, ModelPromotionDecision),
    }
