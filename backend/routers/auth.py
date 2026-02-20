from __future__ import annotations

from fastapi import APIRouter, Depends, Header
from sqlmodel import Session

from ..auth import UserContext, get_current_user, issue_jwt_token
from ..config import settings
from ..schemas import AuthMeResponse, AuthTokenIssueRequest, AuthTokenIssueResponse
from ..tenant import TenantContext, get_tenant_context, require_global_admin
from ..database import get_session

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/me", response_model=AuthMeResponse)
def auth_me(
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_tenant_context),
) -> AuthMeResponse:
    return AuthMeResponse(
        user_id=user.user_id,
        role=user.role,
        tenant_id=tenant.tenant_id,
        tenant_role=tenant.membership_role,
        is_global_admin=user.is_global_admin or tenant.is_global_admin,
        auth_source=user.auth_source,
    )


@router.post("/token", response_model=AuthTokenIssueResponse)
def issue_token(
    payload: AuthTokenIssueRequest,
    authorization: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
    x_user_role: str | None = Header(default=None),
    x_tenant_id: str | None = Header(default=None),
    x_bootstrap_key: str | None = Header(default=None),
    session: Session = Depends(get_session),
) -> AuthTokenIssueResponse:
    bootstrap_ok = bool(settings.auth_bootstrap_key and x_bootstrap_key == settings.auth_bootstrap_key)
    if not bootstrap_ok:
        requester = get_current_user(
            authorization=authorization,
            x_user_id=x_user_id,
            x_user_role=x_user_role,
            x_tenant_id=x_tenant_id,
        )
        if requester.role not in {"admin", "tenant_admin", "system"}:
            requester = require_global_admin(session=session, user=requester)

    token_payload = issue_jwt_token(
        user_id=payload.user_id,
        role=payload.role,
        expires_minutes=payload.expires_in_minutes,
        tenant_id=payload.tenant_id,
        is_global_admin=payload.is_global_admin,
    )
    return AuthTokenIssueResponse(
        access_token=token_payload["access_token"],
        token_type=token_payload["token_type"],
        expires_at=token_payload["expires_at"],
        issued_for_user_id=payload.user_id,
        issued_for_role=payload.role,
        issued_for_tenant_id=payload.tenant_id,
        issued_for_is_global_admin=payload.is_global_admin,
    )
