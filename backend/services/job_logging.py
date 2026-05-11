from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from sqlmodel import Session

from ..config import settings
from ..models import JobLogEntry

_DETAIL_ORDER = {"basic": 0, "detailed": 1, "extreme": 2}
_LEVEL_ORDER = {"debug": 10, "info": 20, "warning": 30, "error": 40}

logger = logging.getLogger("video-highlights.job")


def _detail_enabled(entry_detail: str) -> bool:
    configured = (settings.job_log_detail or "basic").strip().lower()
    configured_rank = _DETAIL_ORDER.get(configured, 0)
    entry_rank = _DETAIL_ORDER.get((entry_detail or "basic").strip().lower(), 0)
    return configured_rank >= entry_rank


def _level_enabled(level: str) -> bool:
    configured = (settings.log_level or "INFO").strip().lower()
    configured_rank = _LEVEL_ORDER.get(configured, 20)
    entry_rank = _LEVEL_ORDER.get((level or "info").strip().lower(), 20)
    return entry_rank >= configured_rank


def append_job_log(
    session: Session,
    job_id: str,
    tenant_id: Optional[str],
    level: str,
    stage: Optional[str],
    message: str,
    detail_level: str = "basic",
    data: Optional[Dict[str, Any]] = None,
    force_persist: bool = False,
) -> None:
    level_norm = (level or "info").lower()
    detail_norm = (detail_level or "basic").lower()
    payload = data or {}

    if _level_enabled(level_norm):
        line = f"job={job_id} stage={stage or '-'} detail={detail_norm} msg={message}"
        if payload and _detail_enabled(detail_norm):
            line = f"{line} data={payload}"
        getattr(logger, level_norm if level_norm in {"debug", "info", "warning", "error"} else "info")(line)

    if not settings.persist_job_logs:
        return
    if not force_persist and not _detail_enabled(detail_norm):
        return

    entry = JobLogEntry(
        tenant_id=tenant_id,
        job_id=job_id,
        level=level_norm,
        detail_level=detail_norm,
        stage=stage,
        message=message,
        data_json=payload,
    )
    session.add(entry)
