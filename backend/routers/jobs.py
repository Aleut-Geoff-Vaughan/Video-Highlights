from __future__ import annotations

from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from ..auth import UserContext, require_roles
from ..config import settings
from ..database import get_session
from ..models import Match, ProcessingJob
from ..schemas import JobCreate, JobRead
from ..serializers import job_to_read
from ..services.job_runner import job_runner
from ..tenant import TenantContext, get_tenant_context
from ..utils import decode_cursor, encode_cursor, utcnow

router = APIRouter(tags=["jobs"])


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
    job = session.get(ProcessingJob, job_id)
    if not job or job.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job_to_read(job)


@router.post("/jobs/{job_id}/cancel", response_model=JobRead)
def cancel_job(
    job_id: str,
    session: Session = Depends(get_session),
    _: UserContext = Depends(require_roles("admin", "analyst", "coach", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> JobRead:
    job = session.get(ProcessingJob, job_id)
    if not job or job.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.status in {"completed", "failed", "canceled"}:
        return job_to_read(job)

    job.status = "canceled"
    job.stage = "canceled"
    job.progress = 1.0
    job.completed_at = utcnow()
    job.updated_at = utcnow()
    session.add(job)
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
    job = session.get(ProcessingJob, job_id)
    if not job or job.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    retry = ProcessingJob(tenant_id=job.tenant_id, match_id=job.match_id, config_json=job.config_json or {})
    session.add(retry)
    session.commit()
    session.refresh(retry)

    if settings.job_execution_mode == "inline":
        job_runner.submit_processing_job(retry.id)
    return job_to_read(retry)


@router.post("/jobs/worker/run-once", response_model=Dict[str, object])
def worker_run_once(
    _: UserContext = Depends(require_roles("admin", "analyst", "system", "tenant_admin")),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, object]:
    worked, job_id = job_runner.run_next_queued_job(tenant_id=tenant.tenant_id)
    return {"worked": worked, "job_id": job_id}
