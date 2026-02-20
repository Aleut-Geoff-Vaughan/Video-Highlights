from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Column, UniqueConstraint
from sqlmodel import Field, SQLModel

from .utils import generate_id, utcnow


class Tenant(SQLModel, table=True):
    __tablename__ = "tenants"

    id: str = Field(default_factory=lambda: generate_id("tenant"), primary_key=True)
    slug: str = Field(index=True, unique=True)
    name: str = Field(index=True)
    status: str = Field(default="active", index=True)
    metadata_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=utcnow, nullable=False)


class UserAccount(SQLModel, table=True):
    __tablename__ = "users"

    id: str = Field(default_factory=lambda: generate_id("user"), primary_key=True)
    email: Optional[str] = Field(default=None, index=True)
    display_name: Optional[str] = Field(default=None)
    status: str = Field(default="active", index=True)
    is_global_admin: bool = Field(default=False, index=True)
    metadata_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=utcnow, nullable=False)


class TenantMembership(SQLModel, table=True):
    __tablename__ = "tenant_memberships"
    __table_args__ = (UniqueConstraint("tenant_id", "user_id", name="uq_tenant_membership_user"),)

    id: str = Field(default_factory=lambda: generate_id("membership"), primary_key=True)
    tenant_id: str = Field(foreign_key="tenants.id", index=True)
    user_id: str = Field(foreign_key="users.id", index=True)
    role: str = Field(index=True)  # tenant_admin|coach|analyst|parent|player|system
    status: str = Field(default="active", index=True)
    metadata_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=utcnow, nullable=False)


class Match(SQLModel, table=True):
    __tablename__ = "matches"

    id: str = Field(default_factory=lambda: generate_id("match"), primary_key=True)
    tenant_id: Optional[str] = Field(default=None, foreign_key="tenants.id", index=True)
    name: Optional[str] = Field(default=None, index=True)
    home_team_name: Optional[str] = Field(default=None)
    away_team_name: Optional[str] = Field(default=None)
    match_date: Optional[str] = Field(default=None)
    source_video_path: str = Field(index=True)
    metadata_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=utcnow, nullable=False)


class ProcessingJob(SQLModel, table=True):
    __tablename__ = "processing_jobs"

    id: str = Field(default_factory=lambda: generate_id("job"), primary_key=True)
    tenant_id: Optional[str] = Field(default=None, foreign_key="tenants.id", index=True)
    match_id: str = Field(foreign_key="matches.id", index=True)
    status: str = Field(default="queued", index=True)
    stage: Optional[str] = Field(default="queued")
    progress: float = Field(default=0.0)
    config_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    result_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    error_message: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=utcnow, nullable=False)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)


class Event(SQLModel, table=True):
    __tablename__ = "events"

    id: str = Field(default_factory=lambda: generate_id("evt"), primary_key=True)
    tenant_id: Optional[str] = Field(default=None, foreign_key="tenants.id", index=True)
    match_id: str = Field(foreign_key="matches.id", index=True)
    job_id: Optional[str] = Field(default=None, foreign_key="processing_jobs.id", index=True)
    event_type: str = Field(index=True)
    status: str = Field(default="auto_detected", index=True)
    confidence: float = Field(default=0.0)
    period: Optional[str] = Field(default=None, index=True)
    occurred_at_ms: int = Field(default=0, index=True)
    start_ms: int = Field(default=0)
    end_ms: int = Field(default=0)
    frame_index: int = Field(default=0)
    team_id: Optional[str] = Field(default=None, index=True)
    player_id: Optional[str] = Field(default=None, index=True)
    jersey_number: Optional[str] = Field(default=None)
    source_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    location_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    participants_json: List[Dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    evidence_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    explanations_json: List[Dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=utcnow, nullable=False)


class EventFeedback(SQLModel, table=True):
    __tablename__ = "event_feedback"

    id: str = Field(default_factory=lambda: generate_id("fb"), primary_key=True)
    tenant_id: Optional[str] = Field(default=None, foreign_key="tenants.id", index=True)
    match_id: str = Field(foreign_key="matches.id", index=True)
    event_id: Optional[str] = Field(default=None, foreign_key="events.id", index=True)
    feedback_type: str = Field(index=True)
    status: str = Field(default="pending_review", index=True)
    severity: str = Field(default="medium", index=True)
    comment: Optional[str] = Field(default=None)
    submitted_by_user_id: Optional[str] = Field(default=None, index=True)
    submitted_by_role: Optional[str] = Field(default=None)
    correction_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    evidence_json: List[Dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    review_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=utcnow, nullable=False)


class TrainingFeedbackBatch(SQLModel, table=True):
    __tablename__ = "training_feedback_batches"

    id: str = Field(default_factory=lambda: generate_id("fbatch"), primary_key=True)
    tenant_id: Optional[str] = Field(default=None, foreign_key="tenants.id", index=True)
    criteria_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    item_count: int = Field(default=0)
    created_by_user_id: Optional[str] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=utcnow, nullable=False)


class TrainingRun(SQLModel, table=True):
    __tablename__ = "training_runs"

    id: str = Field(default_factory=lambda: generate_id("train"), primary_key=True)
    tenant_id: Optional[str] = Field(default=None, foreign_key="tenants.id", index=True)
    batch_id: Optional[str] = Field(default=None, foreign_key="training_feedback_batches.id", index=True)
    target_model: str = Field(default="event-v0", index=True)
    status: str = Field(default="queued", index=True)
    candidate_model_version: Optional[str] = Field(default=None)
    metrics_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    gates_passed: bool = Field(default=False)
    notes: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=utcnow, nullable=False)


class ModelVersion(SQLModel, table=True):
    __tablename__ = "model_versions"

    id: str = Field(default_factory=lambda: generate_id("model"), primary_key=True)
    tenant_id: Optional[str] = Field(default=None, foreign_key="tenants.id", index=True)
    target_model: str = Field(index=True)
    version: str = Field(index=True)
    run_id: Optional[str] = Field(default=None, foreign_key="training_runs.id", index=True)
    promoted: bool = Field(default=False, index=True)
    promoted_by_user_id: Optional[str] = Field(default=None, index=True)
    promoted_at: Optional[datetime] = Field(default=None)
    metrics_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    notes: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow, nullable=False)


class ModelPromotionDecision(SQLModel, table=True):
    __tablename__ = "model_promotion_decisions"

    id: str = Field(default_factory=lambda: generate_id("promote"), primary_key=True)
    tenant_id: Optional[str] = Field(default=None, foreign_key="tenants.id", index=True)
    run_id: str = Field(foreign_key="training_runs.id", index=True)
    target_model: str = Field(index=True)
    candidate_model_version: str = Field(index=True)
    decision: str = Field(index=True)  # approved | rejected
    decided_by_user_id: str = Field(index=True)
    reason: Optional[str] = Field(default=None)
    force: bool = Field(default=False)
    created_at: datetime = Field(default_factory=utcnow, nullable=False)
