from __future__ import annotations

import os
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Tuple

from sqlmodel import select

from ..config import settings
from ..database import session_scope
from ..models import Match, ProcessingJob, TrainingFeedbackBatch, TrainingRun
from ..utils import ensure_dir


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobRunner:
    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = Lock()
        self._futures: Dict[str, Future] = {}

    def submit_processing_job(self, job_id: str) -> None:
        future = self._executor.submit(self._run_processing_job, job_id)
        with self._lock:
            self._futures[job_id] = future

    def submit_training_run(self, run_id: str) -> None:
        future = self._executor.submit(self._run_training_job, run_id)
        with self._lock:
            self._futures[run_id] = future

    def run_next_queued_job(self, tenant_id: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Claim and process one queued job synchronously in the current thread.
        Useful for dedicated worker processes.
        Returns (worked, job_id).
        """
        job_id: Optional[str] = None
        with session_scope() as session:
            stmt = select(ProcessingJob).where(ProcessingJob.status == "queued")
            if tenant_id:
                stmt = stmt.where(ProcessingJob.tenant_id == tenant_id)
            stmt = stmt.order_by(ProcessingJob.created_at.asc()).limit(1)
            job = session.exec(stmt).first()
            if not job:
                return False, None
            job.status = "claimed"
            job.stage = "claimed"
            job.updated_at = _utcnow()
            session.add(job)
            job_id = job.id

        if not job_id:
            return False, None
        self._run_processing_job(job_id)
        return True, job_id

    def _run_processing_job(self, job_id: str) -> None:
        try:
            with session_scope() as session:
                job = session.get(ProcessingJob, job_id)
                if not job:
                    return
                match = session.get(Match, job.match_id)
                if not match:
                    job.status = "failed"
                    job.error_message = f"Match not found for job {job_id}"
                    job.updated_at = _utcnow()
                    job.completed_at = _utcnow()
                    return

                job.status = "running"
                job.stage = "initializing"
                job.progress = 0.01
                job.started_at = _utcnow()
                job.updated_at = _utcnow()

                config = job.config_json or {}
                video_path = config.get("video_path") or match.source_video_path
                output_dir = config.get("output_dir") or os.path.join(settings.output_root, job.id)
                ensure_dir(output_dir)

                if not os.path.exists(video_path):
                    job.status = "failed"
                    job.error_message = f"Video path not found: {video_path}"
                    job.updated_at = _utcnow()
                    job.completed_at = _utcnow()
                    return

                job.stage = "processing_video"
                job.progress = 0.05
                job.updated_at = _utcnow()

            # Delay import to keep API startup lightweight in environments without CV deps.
            from VideoHighlights import parse_time, process_video_highlights

            def _parse_trim(value: Optional[object]) -> Optional[float]:
                if value is None:
                    return None
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str) and value.strip():
                    return float(parse_time(value))
                return None

            with session_scope() as session:
                job = session.get(ProcessingJob, job_id)
                if not job:
                    return
                config = job.config_json or {}
                match = session.get(Match, job.match_id)
                if not match:
                    job.status = "failed"
                    job.error_message = f"Match not found for job {job_id}"
                    job.updated_at = _utcnow()
                    job.completed_at = _utcnow()
                    return

                video_path = config.get("video_path") or match.source_video_path
                output_dir = config.get("output_dir") or os.path.join(settings.output_root, job.id)

            success = process_video_highlights(
                video_path=video_path,
                output_dir=output_dir,
                select_player=bool(config.get("select_player", False)),
                pre_seconds=float(config.get("pre_seconds", 2.0)),
                post_seconds=float(config.get("post_seconds", 6.0)),
                min_clip_duration=float(config.get("min_clip_duration", config.get("min_clip", 4.0))),
                no_audio=bool(config.get("no_audio", False)),
                overlay=bool(config.get("overlay", False)),
                trim_start=_parse_trim(config.get("trim_start")),
                trim_end=_parse_trim(config.get("trim_end")),
                threads=int(config["threads"]) if config.get("threads") is not None else None,
                require_gpu=bool(config.get("require_gpu", False)),
                speed_sensitivity=float(config.get("speed_sensitivity", 2.0)),
                audio_sensitivity=float(config.get("audio_sensitivity", 2.0)),
            )

            artifacts = sorted(str(path.resolve()) for path in Path(output_dir).glob("*.mp4"))
            result_payload = {
                "output_dir": str(Path(output_dir).resolve()),
                "artifact_count": len(artifacts),
                "artifacts": artifacts,
                "engine": "VideoHighlights.py",
            }

            with session_scope() as session:
                job = session.get(ProcessingJob, job_id)
                if not job:
                    return

                if success:
                    job.status = "completed"
                    job.stage = "completed"
                    job.progress = 1.0
                    job.result_json = result_payload
                    job.error_message = None
                else:
                    job.status = "failed"
                    job.stage = "failed"
                    job.progress = 1.0
                    job.result_json = result_payload
                    job.error_message = "Processing pipeline reported failure"

                job.updated_at = _utcnow()
                job.completed_at = _utcnow()

        except Exception as exc:
            error = f"{exc}\n{traceback.format_exc()}"
            with session_scope() as session:
                job = session.get(ProcessingJob, job_id)
                if not job:
                    return
                job.status = "failed"
                job.stage = "failed"
                job.progress = 1.0
                job.error_message = error
                job.updated_at = _utcnow()
                job.completed_at = _utcnow()

    def _run_training_job(self, run_id: str) -> None:
        try:
            with session_scope() as session:
                run = session.get(TrainingRun, run_id)
                if not run:
                    return
                run.status = "running"
                run.updated_at = _utcnow()

            with session_scope() as session:
                run = session.get(TrainingRun, run_id)
                if not run:
                    return
                run.status = "evaluating"
                run.updated_at = _utcnow()

                batch = session.get(TrainingFeedbackBatch, run.batch_id) if run.batch_id else None
                item_count = int(batch.item_count) if batch else 0
                candidate_version = f"{run.target_model}.{_utcnow().strftime('%Y%m%d%H%M%S')}"

                # Deterministic placeholder metrics for V1 workflow wiring.
                base = min(0.95, 0.60 + (item_count / 1000.0))
                metrics = {
                    "goal_precision": round(base, 3),
                    "goal_recall": round(max(0.4, base - 0.04), 3),
                    "foul_precision": round(max(0.35, base - 0.20), 3),
                    "foul_recall": round(max(0.30, base - 0.25), 3),
                    "feedback_items_used": item_count,
                }

                run.candidate_model_version = candidate_version
                run.metrics_json = metrics
                run.gates_passed = item_count >= 20
                run.status = "completed"
                run.updated_at = _utcnow()

        except Exception as exc:
            with session_scope() as session:
                run = session.get(TrainingRun, run_id)
                if not run:
                    return
                run.status = "failed"
                run.metrics_json = {"error": str(exc)}
                run.updated_at = _utcnow()


job_runner = JobRunner(max_workers=max(1, settings.job_max_workers))
