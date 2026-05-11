from __future__ import annotations

from fastapi import APIRouter

from ..services.gpu_status import get_gpu_status

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/health/gpu")
def gpu_health() -> dict:
    return get_gpu_status()
