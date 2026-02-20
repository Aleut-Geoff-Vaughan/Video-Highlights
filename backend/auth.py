from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Set

from fastapi import Depends, Header, HTTPException

from .config import settings


@dataclass
class UserContext:
    user_id: str
    role: str
    tenant_id: Optional[str] = None
    tenant_role: Optional[str] = None
    is_global_admin: bool = False
    token: Optional[str] = None
    auth_source: str = "unknown"


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    prefix = "bearer "
    value = authorization.strip()
    if value.lower().startswith(prefix):
        return value[len(prefix) :].strip() or None
    return None


def _decode_jwt_token(token: str) -> Optional[Dict[str, object]]:
    if not settings.jwt_secret:
        return None
    try:
        import jwt
    except Exception:
        return None

    kwargs = {
        "algorithms": [settings.jwt_algorithm],
        "issuer": settings.jwt_issuer,
    }
    if settings.jwt_audience:
        kwargs["audience"] = settings.jwt_audience

    try:
        claims = jwt.decode(token, settings.jwt_secret, **kwargs)
    except Exception:
        return None

    sub = claims.get("sub")
    role = claims.get("role")
    if not sub or not role:
        return None
    tenant_id = claims.get("tenant_id")
    is_global_admin = bool(claims.get("is_global_admin", False))
    return {
        "sub": str(sub),
        "role": str(role).lower(),
        "tenant_id": str(tenant_id) if tenant_id else None,
        "is_global_admin": is_global_admin,
    }


def issue_jwt_token(
    user_id: str,
    role: str,
    expires_minutes: Optional[int] = None,
    tenant_id: Optional[str] = None,
    is_global_admin: bool = False,
) -> Dict[str, str]:
    if not settings.jwt_secret:
        raise HTTPException(status_code=400, detail="JWT is not configured (VH_JWT_SECRET missing)")
    try:
        import jwt
    except Exception as exc:
        raise HTTPException(status_code=500, detail="PyJWT is not installed") from exc

    ttl_minutes = expires_minutes if expires_minutes is not None else settings.jwt_default_exp_minutes
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=max(1, ttl_minutes))
    payload = {
        "sub": user_id,
        "role": role.lower(),
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "iss": settings.jwt_issuer,
    }
    if tenant_id:
        payload["tenant_id"] = tenant_id
    if is_global_admin:
        payload["is_global_admin"] = True
    if settings.jwt_audience:
        payload["aud"] = settings.jwt_audience
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_at": exp.isoformat(),
    }


def get_current_user(
    authorization: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
    x_user_role: Optional[str] = Header(default=None),
    x_tenant_id: Optional[str] = Header(default=None),
) -> UserContext:
    """
    Auth modes:
    1) Env-token mode: VH_API_TOKENS mapping.
    2) JWT mode: VH_JWT_SECRET + bearer JWT.
    3) Dev fallback: allow x-user-role/x-user-id headers or default admin when auth is not required.
    """
    token = _extract_bearer_token(authorization)
    token_map = settings.api_tokens

    if token and token in token_map:
        role = token_map[token]
        return UserContext(
            user_id=x_user_id or f"token_user_{role}",
            role=role,
            tenant_id=x_tenant_id,
            is_global_admin=role == "system",
            token=token,
            auth_source="env_token",
        )

    if token:
        jwt_claims = _decode_jwt_token(token)
        if jwt_claims:
            return UserContext(
                user_id=jwt_claims["sub"],
                role=jwt_claims["role"],
                tenant_id=jwt_claims.get("tenant_id") or x_tenant_id,
                is_global_admin=bool(jwt_claims.get("is_global_admin", False)) or jwt_claims["role"] == "system",
                token=token,
                auth_source="jwt",
            )

    if settings.auth_required:
        raise HTTPException(status_code=401, detail="Unauthorized: valid bearer token required")

    role = (x_user_role or "admin").strip().lower()
    user_id = (x_user_id or "dev_user").strip()
    return UserContext(
        user_id=user_id,
        role=role,
        tenant_id=x_tenant_id,
        is_global_admin=role == "system",
        token=token,
        auth_source="dev_fallback",
    )


def require_roles(*allowed_roles: str):
    allowed: Set[str] = {role.strip().lower() for role in allowed_roles if role.strip()}
    if not allowed:
        allowed = {"admin"}

    def dependency(user: UserContext = Depends(get_current_user)) -> UserContext:
        if user.is_global_admin:
            return user
        if user.role not in allowed:
            raise HTTPException(
                status_code=403,
                detail=f"Forbidden: role '{user.role}' cannot access this resource",
            )
        return user

    return dependency
