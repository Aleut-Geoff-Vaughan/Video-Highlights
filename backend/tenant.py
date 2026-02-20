from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Header, HTTPException
from sqlmodel import Session, select

from .auth import UserContext, get_current_user
from .database import get_session
from .models import Tenant, TenantMembership, UserAccount
from .utils import utcnow

DEFAULT_TENANT_SLUG = "default"
DEFAULT_TENANT_NAME = "Default Tenant"


@dataclass
class TenantContext:
    tenant_id: str
    tenant_slug: str
    tenant_name: str
    user_account_id: str
    membership_id: Optional[str] = None
    membership_role: Optional[str] = None
    is_global_admin: bool = False


def ensure_default_tenant(session: Session) -> Tenant:
    tenant = session.exec(select(Tenant).where(Tenant.slug == DEFAULT_TENANT_SLUG)).first()
    if tenant:
        return tenant

    tenant = Tenant(slug=DEFAULT_TENANT_SLUG, name=DEFAULT_TENANT_NAME, status="active")
    session.add(tenant)
    session.commit()
    session.refresh(tenant)
    return tenant


def _find_tenant(session: Session, tenant_ref: Optional[str]) -> Optional[Tenant]:
    if not tenant_ref:
        return None
    tenant = session.get(Tenant, tenant_ref)
    if tenant:
        return tenant
    return session.exec(select(Tenant).where(Tenant.slug == tenant_ref)).first()


def ensure_user_account(session: Session, user: UserContext) -> UserAccount:
    account = session.get(UserAccount, user.user_id)
    now = utcnow()
    if account:
        changed = False
        if user.is_global_admin and not account.is_global_admin:
            account.is_global_admin = True
            changed = True
        if account.status != "active":
            account.status = "active"
            changed = True
        if changed:
            account.updated_at = now
            session.add(account)
            session.commit()
            session.refresh(account)
        return account

    account = UserAccount(
        id=user.user_id,
        display_name=user.user_id,
        status="active",
        is_global_admin=user.is_global_admin or user.role == "system",
    )
    session.add(account)
    session.commit()
    session.refresh(account)
    return account


def get_tenant_context(
    x_tenant_id: Optional[str] = Header(default=None),
    session: Session = Depends(get_session),
    user: UserContext = Depends(get_current_user),
) -> TenantContext:
    default_tenant = ensure_default_tenant(session)
    user.is_global_admin = user.is_global_admin or user.role == "system"
    account = ensure_user_account(session, user)
    user.is_global_admin = user.is_global_admin or account.is_global_admin

    active_memberships = list(
        session.exec(
            select(TenantMembership)
            .where(TenantMembership.user_id == account.id)
            .where(TenantMembership.status == "active")
        )
    )
    membership_by_tenant = {item.tenant_id: item for item in active_memberships}

    tenant = _find_tenant(session, x_tenant_id or user.tenant_id)
    if not tenant:
        if len(active_memberships) == 1:
            tenant = session.get(Tenant, active_memberships[0].tenant_id)
        elif len(active_memberships) > 1 and not user.is_global_admin:
            raise HTTPException(
                status_code=400,
                detail="Multiple tenant memberships found. Provide X-Tenant-ID header.",
            )
        else:
            tenant = default_tenant

    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    membership = membership_by_tenant.get(tenant.id)
    if not membership and not user.is_global_admin:
        # Default bootstrap path for first-time users in development mode.
        if tenant.id == default_tenant.id:
            membership = TenantMembership(
                tenant_id=tenant.id,
                user_id=account.id,
                role="tenant_admin" if user.role in {"admin", "system"} else user.role,
                status="active",
            )
            session.add(membership)
            session.commit()
            session.refresh(membership)
        else:
            raise HTTPException(
                status_code=403,
                detail=f"Forbidden: user '{account.id}' is not an active member of tenant '{tenant.id}'",
            )

    user.tenant_id = tenant.id
    user.tenant_role = membership.role if membership else None

    return TenantContext(
        tenant_id=tenant.id,
        tenant_slug=tenant.slug,
        tenant_name=tenant.name,
        user_account_id=account.id,
        membership_id=membership.id if membership else None,
        membership_role=membership.role if membership else None,
        is_global_admin=user.is_global_admin,
    )


def require_global_admin(
    session: Session = Depends(get_session),
    user: UserContext = Depends(get_current_user),
) -> UserContext:
    user.is_global_admin = user.is_global_admin or user.role == "system"
    account = ensure_user_account(session, user)
    if not (user.role in {"admin", "system"} or user.is_global_admin or account.is_global_admin):
        raise HTTPException(
            status_code=403,
            detail=f"Forbidden: user '{user.user_id}' is not a global admin",
        )
    user.is_global_admin = True
    return user


def require_tenant_admin(
    tenant: TenantContext = Depends(get_tenant_context),
    user: UserContext = Depends(get_current_user),
) -> TenantContext:
    if user.is_global_admin or user.role == "system":
        return tenant
    if tenant.membership_role not in {"tenant_admin"}:
        raise HTTPException(
            status_code=403,
            detail=f"Forbidden: tenant admin access required (tenant_role={tenant.membership_role})",
        )
    return tenant
