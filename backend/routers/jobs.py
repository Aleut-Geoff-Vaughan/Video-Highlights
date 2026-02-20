from __future__ import annotations

import json
import os
from typing import Dict

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlmodel import Session, select

from ..auth import UserContext, require_roles
from ..config import settings
from ..database import get_session
from ..models import Event, JobLogEntry, Match, ProcessingJob
from ..schemas import JobCreate, JobRead, JobRerunRequest
from ..serializers import job_log_to_read, job_to_read
from ..services.job_logging import append_job_log
from ..services.job_runner import job_runner
from ..tenant import TenantContext, get_tenant_context
from ..utils import decode_cursor, encode_cursor, utcnow

router = APIRouter(tags=["jobs"])


def _get_tenant_job_or_404(session: Session, tenant_id: str, job_id: str) -> ProcessingJob:
    job = session.get(ProcessingJob, job_id)
    if not job or job.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


def _mark_job_cancel_requested(session: Session, job: ProcessingJob, reason: str) -> ProcessingJob:
    if job.status in {"completed", "failed", "canceled"}:
        append_job_log(
            session=session,
            job_id=job.id,
            tenant_id=job.tenant_id,
            level="info",
            stage=job.stage,
            message="Cancel requested but job already terminal",
            detail_level="detailed",
            data={"status": job.status, "reason": reason},
        )
        return job

    job.cancel_requested = True
    if job.status in {"queued", "claimed"}:
        job.status = "canceled"
        job.stage = "canceled"
        job.progress = 1.0
        job.completed_at = utcnow()
        message = "Job canceled before processing started"
    else:
        job.status = "cancel_requested"
        job.stage = "cancel_requested"
        message = "Cancel requested for running job"
    job.updated_at = utcnow()
    session.add(job)
    append_job_log(
        session=session,
        job_id=job.id,
        tenant_id=job.tenant_id,
        level="warning",
        stage=job.stage,
        message=message,
        detail_level="basic",
        data={"reason": reason},
    )
    return job


def _record_match_processing_mechanics(
    session: Session,
    match: Match,
    job: ProcessingJob,
    source_job_id: str | None = None,
    reason: str | None = None,
) -> None:
    metadata = dict(match.metadata_json or {})
    history = list(metadata.get("processing_history", []))
    config = dict(job.config_json or {})
    history.append(
        {
            "job_id": job.id,
            "source_job_id": source_job_id,
            "status": job.status,
            "created_at": job.created_at.isoformat(),
            "model_version": config.get("model_version"),
            "focus_event_types": config.get("focus_event_types", []),
            "profile_name": config.get("profile_name"),
            "reason": reason,
            "config": config,
        }
    )
    metadata["processing_history"] = history[-200:]
    metadata["latest_job_id"] = job.id
    metadata["latest_model_version"] = config.get("model_version")
    metadata["latest_focus_event_types"] = config.get("focus_event_types", [])
    metadata["last_processing_updated_at"] = utcnow().isoformat()
    match.metadata_json = metadata
    match.updated_at = utcnow()
    session.add(match)


def _refresh_match_latest_processing_metadata(session: Session, match: Match) -> None:
    metadata = dict(match.metadata_json or {})
    jobs = list(
        session.exec(
            select(ProcessingJob)
            .where(ProcessingJob.match_id == match.id)
            .where(ProcessingJob.tenant_id == match.tenant_id)
            .order_by(ProcessingJob.created_at.desc())
        )
    )
    existing_ids = {job.id for job in jobs}
    history = list(metadata.get("processing_history", []))
    metadata["processing_history"] = [entry for entry in history if str(entry.get("job_id")) in existing_ids][-200:]
    if jobs:
        latest = jobs[0]
        config = dict(latest.config_json or {})
        metadata["latest_job_id"] = latest.id
        metadata["latest_model_version"] = config.get("model_version")
        metadata["latest_focus_event_types"] = config.get("focus_event_types", [])
    else:
        metadata.pop("latest_job_id", None)
        metadata.pop("latest_model_version", None)
        metadata.pop("latest_focus_event_types", None)
    metadata["last_processing_updated_at"] = utcnow().isoformat()
    match.metadata_json = metadata
    match.updated_at = utcnow()
    session.add(match)


def _read_live_manifest_bookmarks(job: ProcessingJob) -> list[Dict[str, object]]:
    config = dict(job.config_json or {})
    output_dir = str(config.get("output_dir") or os.path.join(settings.output_root, job.id))
    manifest_path = os.path.join(output_dir, "analysis_bookmarks.json")
    if not os.path.exists(manifest_path):
        return []
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    bookmarks = list(payload.get("bookmarks", []) or [])
    return [entry for entry in bookmarks if isinstance(entry, dict)]


@router.post("/matches/{match_id}/jobs", response_model=JobRead, status_code=201)
def create_job(
    match_id: str,
    payload: JobCreate,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> JobRead:
    match = session.get(Match, match_id)
    if not match or match.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")

    job = ProcessingJob(tenant_id=match.tenant_id, match_id=match_id, config_json=payload.config)
    session.add(job)
    session.commit()
    session.refresh(job)
    append_job_log(
        session=session,
        job_id=job.id,
        tenant_id=job.tenant_id,
        level="info",
        stage="queued",
        message="Processing job created",
        detail_level="basic",
        data={"execution_mode": settings.job_execution_mode},
    )
    append_job_log(
        session=session,
        job_id=job.id,
        tenant_id=job.tenant_id,
        level="debug",
        stage="queued",
        message="Job configuration accepted",
        detail_level="extreme",
        data={"config": payload.config},
    )
    _record_match_processing_mechanics(session=session, match=match, job=job, reason="create_job")
    session.commit()

    if settings.job_execution_mode == "inline":
        job_runner.submit_processing_job(job.id)
    return job_to_read(job)


@router.get("/matches/{match_id}/jobs", response_model=Dict[str, object])
def list_match_jobs(
    match_id: str,
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    cursor: str | None = Query(default=None),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    match = session.get(Match, match_id)
    if not match or match.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")

    offset = decode_cursor(cursor)
    stmt = select(ProcessingJob).where(ProcessingJob.match_id == match_id).where(
        ProcessingJob.tenant_id == tenant.tenant_id
    )
    if status:
        stmt = stmt.where(ProcessingJob.status == status)
    stmt = stmt.order_by(ProcessingJob.created_at.desc()).offset(offset).limit(limit + 1)
    rows = list(session.exec(stmt))
    has_more = len(rows) > limit
    items = [job_to_read(job) for job in rows[:limit]]
    next_cursor = encode_cursor(offset + limit) if has_more else None
    return {"items": [item.model_dump() for item in items], "next_cursor": next_cursor}


@router.get("/jobs/{job_id}", response_model=JobRead)
def get_job(
    job_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> JobRead:
    job = _get_tenant_job_or_404(session, tenant.tenant_id, job_id)
    return job_to_read(job)


@router.get("/jobs/{job_id}/bookmarks", response_model=Dict[str, object])
def get_job_bookmarks(
    job_id: str,
    limit: int = Query(default=2000, ge=1, le=10000),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    job = _get_tenant_job_or_404(session, tenant.tenant_id, job_id)

    rows = list(
        session.exec(
            select(Event)
            .where(Event.job_id == job.id)
            .where(Event.tenant_id == tenant.tenant_id)
            .order_by(Event.occurred_at_ms.asc())
            .limit(limit)
        )
    )
    if rows:
        items = [
            {
                "event_id": item.id,
                "event_type": item.event_type,
                "status": item.status,
                "confidence": item.confidence,
                "occurred_at_ms": item.occurred_at_ms,
                "start_ms": item.start_ms,
                "end_ms": item.end_ms,
                "source": item.source_json or {},
                "explanations": item.explanations_json or [],
            }
            for item in rows
        ]
        return {"source": "events", "status": job.status, "items": items}

    result_bookmarks = list((job.result_json or {}).get("bookmarks", []) or [])
    if result_bookmarks:
        return {"source": "result", "status": job.status, "items": result_bookmarks[:limit]}

    live_bookmarks = _read_live_manifest_bookmarks(job)
    return {"source": "manifest", "status": job.status, "items": live_bookmarks[:limit]}


@router.get("/jobs/{job_id}/logs", response_model=Dict[str, object])
def list_job_logs(
    job_id: str,
    level: str | None = Query(default=None),
    stage: str | None = Query(default=None),
    detail_level: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=5000),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "parent", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    _get_tenant_job_or_404(session, tenant.tenant_id, job_id)

    stmt = select(JobLogEntry).where(JobLogEntry.job_id == job_id).where(JobLogEntry.tenant_id == tenant.tenant_id)
    if level:
        stmt = stmt.where(JobLogEntry.level == level.lower())
    if stage:
        stmt = stmt.where(JobLogEntry.stage == stage)
    if detail_level:
        stmt = stmt.where(JobLogEntry.detail_level == detail_level.lower())
    stmt = stmt.order_by(JobLogEntry.created_at.desc()).limit(limit)
    rows = list(session.exec(stmt))
    items = [job_log_to_read(item).model_dump() for item in rows]
    return {"items": items}


@router.post("/jobs/{job_id}/cancel", response_model=JobRead)
def cancel_job(
    job_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> JobRead:
    job = _get_tenant_job_or_404(session, tenant.tenant_id, job_id)
    _mark_job_cancel_requested(session, job, reason="cancel_endpoint")
    session.commit()
    session.refresh(job)
    return job_to_read(job)


@router.delete("/jobs/{job_id}", response_model=Dict[str, object])
def delete_job(
    job_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    job = _get_tenant_job_or_404(session, tenant.tenant_id, job_id)
    if job.status in {"running", "claimed"}:
        raise HTTPException(status_code=409, detail="Cannot delete active job. Cancel/kill it first.")

    match = session.get(Match, job.match_id)
    logs = list(
        session.exec(
            select(JobLogEntry)
            .where(JobLogEntry.job_id == job.id)
            .where(JobLogEntry.tenant_id == tenant.tenant_id)
        )
    )
    events = list(
        session.exec(
            select(Event)
            .where(Event.job_id == job.id)
            .where(Event.tenant_id == tenant.tenant_id)
        )
    )
    deleted_logs = len(logs)
    deleted_events = len(events)
    for item in logs:
        session.delete(item)
    for item in events:
        session.delete(item)
    session.delete(job)
    if match and match.tenant_id == tenant.tenant_id:
        _refresh_match_latest_processing_metadata(session, match)
    session.commit()
    return {
        "deleted": True,
        "job_id": job_id,
        "deleted_logs": deleted_logs,
        "deleted_events": deleted_events,
    }


@router.post("/jobs/{job_id}/kill-session", response_model=JobRead)
def kill_job_session(
    job_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> JobRead:
    job = _get_tenant_job_or_404(session, tenant.tenant_id, job_id)
    _mark_job_cancel_requested(session, job, reason="kill_session_endpoint")
    session.commit()
    session.refresh(job)
    return job_to_read(job)


@router.post("/jobs/{job_id}/retry", response_model=JobRead, status_code=201)
def retry_job(
    job_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> JobRead:
    job = _get_tenant_job_or_404(session, tenant.tenant_id, job_id)
    match = session.get(Match, job.match_id)
    if not match or match.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found for job: {job_id}")

    retry = ProcessingJob(tenant_id=job.tenant_id, match_id=job.match_id, config_json=job.config_json or {})
    session.add(retry)
    session.commit()
    session.refresh(retry)
    append_job_log(
        session=session,
        job_id=retry.id,
        tenant_id=retry.tenant_id,
        level="info",
        stage="queued",
        message="Retry job created",
        detail_level="basic",
        data={"source_job_id": job.id},
    )
    _record_match_processing_mechanics(
        session=session,
        match=match,
        job=retry,
        source_job_id=job.id,
        reason="retry_job",
    )
    session.commit()

    if settings.job_execution_mode == "inline":
        job_runner.submit_processing_job(retry.id)
    return job_to_read(retry)


@router.post("/jobs/{job_id}/rerun", response_model=JobRead, status_code=201)
def rerun_job(
    job_id: str,
    payload: JobRerunRequest = Body(default_factory=JobRerunRequest),
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> JobRead:
    job = _get_tenant_job_or_404(session, tenant.tenant_id, job_id)
    match = session.get(Match, job.match_id)
    if not match or match.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Match not found for job: {job_id}")

    new_config = dict(job.config_json or {})
    new_config.update(payload.config_overrides or {})
    rerun = ProcessingJob(tenant_id=job.tenant_id, match_id=job.match_id, config_json=new_config)
    session.add(rerun)
    session.commit()
    session.refresh(rerun)

    append_job_log(
        session=session,
        job_id=rerun.id,
        tenant_id=rerun.tenant_id,
        level="info",
        stage="queued",
        message="Rerun job created",
        detail_level="basic",
        data={
            "source_job_id": job.id,
            "reason": payload.reason,
            "config_overrides": payload.config_overrides,
        },
    )
    _record_match_processing_mechanics(
        session=session,
        match=match,
        job=rerun,
        source_job_id=job.id,
        reason=payload.reason or "rerun_job",
    )
    session.commit()

    if settings.job_execution_mode == "inline":
        job_runner.submit_processing_job(rerun.id)
    return job_to_read(rerun)


@router.post("/jobs/worker/run-once", response_model=Dict[str, object])
def worker_run_once(
    _: UserContext = Depends(require_roles("admin", "analyst", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    worked, job_id = job_runner.run_next_queued_job(tenant_id=tenant.tenant_id)
    return {"worked": worked, "job_id": job_id}
