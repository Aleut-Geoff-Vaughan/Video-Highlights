from __future__ import annotations

import json
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
from ..models import Event, Match, ProcessingJob, TrainingFeedbackBatch, TrainingRun
from .job_logging import append_job_log
from ..utils import ensure_dir


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _read_analysis_manifest(output_dir: str) -> Dict[str, object]:
    manifest_path = Path(output_dir) / "analysis_bookmarks.json"
    if not manifest_path.exists():
        return {}
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            if isinstance(payload, dict):
                return payload
    except Exception:
        return {}
    return {}


def _primary_source_asset_id(match: Match) -> Optional[str]:
    metadata = dict(match.metadata_json or {})
    assets = list(metadata.get("assets", []) or [])
    if not assets:
        return None
    path = (match.source_video_path or "").strip()
    matched = next((asset for asset in assets if str(asset.get("path", "")) == path), None)
    source = matched or assets[0]
    value = source.get("asset_id")
    return str(value) if value else None


def _normalize_event_type(value: object, fallback: str = "shot") -> str:
    allowed = {
        "goal",
        "shot",
        "corner_kick",
        "penalty_kick",
        "free_kick",
        "goal_kick",
        "kickoff",
        "foul",
        "save",
    }
    event_type = str(value or "").strip().lower()
    if event_type in allowed:
        return event_type
    return fallback


def _sync_job_events_from_manifest(
    session,
    job: ProcessingJob,
    match: Match,
    config: Dict[str, object],
    manifest: Dict[str, object],
) -> int:
    bookmarks = list(manifest.get("bookmarks", []) or [])

    # If this job was retried, replace prior rows for deterministic results.
    existing = list(
        session.exec(
            select(Event)
            .where(Event.job_id == job.id)
            .where(Event.match_id == job.match_id)
            .where(Event.tenant_id == job.tenant_id)
        )
    )
    for item in existing:
        session.delete(item)

    detector_version = str(config.get("model_version") or "event-v0")
    source_asset_id = _primary_source_asset_id(match)
    created = 0
    for bookmark in bookmarks:
        if not isinstance(bookmark, dict):
            continue
        start_s = float(bookmark.get("start_s", 0.0) or 0.0)
        end_s = float(bookmark.get("end_s", start_s) or start_s)
        occurred_s = float(bookmark.get("occurred_at_s", (start_s + end_s) / 2.0) or 0.0)

        start_ms = max(0, int(round(start_s * 1000.0)))
        end_ms = max(start_ms, int(round(end_s * 1000.0)))
        occurred_ms = int(round(occurred_s * 1000.0))
        occurred_ms = min(max(start_ms, occurred_ms), end_ms)

        confidence = float(bookmark.get("confidence", 0.0) or 0.0)
        confidence = min(1.0, max(0.0, confidence))
        event_type = _normalize_event_type(bookmark.get("event_type"), fallback="shot")

        signals = bookmark.get("signals", {}) if isinstance(bookmark.get("signals", {}), dict) else {}
        explanations = []
        for key, value in signals.items():
            try:
                explanations.append({"signal": str(key), "value": float(value)})
            except Exception:
                continue

        evidence = {
            "source_asset_id": source_asset_id,
            "bookmark_id": bookmark.get("bookmark_id"),
            "analysis_manifest_path": str(Path(config.get("output_dir") or os.path.join(settings.output_root, job.id)).resolve() / "analysis_bookmarks.json"),
        }

        event = Event(
            tenant_id=job.tenant_id,
            match_id=job.match_id,
            job_id=job.id,
            event_type=event_type,
            status="auto_detected",
            confidence=confidence,
            period=None,
            occurred_at_ms=occurred_ms,
            start_ms=start_ms,
            end_ms=end_ms,
            frame_index=0,
            team_id=None,
            player_id=None,
            jersey_number=None,
            source_json={
                "detector": "videohighlights-multi-factor",
                "detector_version": detector_version,
                "bookmark_label": bookmark.get("label"),
                "sources": bookmark.get("sources", []),
            },
            location_json={},
            participants_json=[],
            evidence_json=evidence,
            explanations_json=explanations,
        )
        session.add(event)
        created += 1
    return created


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
            append_job_log(
                session=session,
                job_id=job.id,
                tenant_id=job.tenant_id,
                level="info",
                stage="claimed",
                message="Worker claimed queued job",
                detail_level="basic",
                data={"tenant_id": tenant_id},
            )
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
                if job.cancel_requested:
                    job.status = "canceled"
                    job.stage = "canceled"
                    job.progress = 1.0
                    job.completed_at = _utcnow()
                    job.updated_at = _utcnow()
                    append_job_log(
                        session=session,
                        job_id=job.id,
                        tenant_id=job.tenant_id,
                        level="warning",
                        stage="canceled",
                        message="Job canceled before initialization",
                        detail_level="basic",
                    )
                    return

                match = session.get(Match, job.match_id)
                if not match:
                    job.status = "failed"
                    job.error_message = f"Match not found for job {job_id}"
                    job.updated_at = _utcnow()
                    job.completed_at = _utcnow()
                    append_job_log(
                        session=session,
                        job_id=job.id,
                        tenant_id=job.tenant_id,
                        level="error",
                        stage="failed",
                        message="Match not found for job",
                        detail_level="basic",
                        data={"match_id": job.match_id},
                    )
                    return

                job.status = "running"
                job.stage = "initializing"
                job.progress = 0.01
                job.started_at = _utcnow()
                job.updated_at = _utcnow()
                append_job_log(
                    session=session,
                    job_id=job.id,
                    tenant_id=job.tenant_id,
                    level="info",
                    stage="initializing",
                    message="Job initialization started",
                    detail_level="basic",
                )

                config = job.config_json or {}
                video_path = config.get("video_path") or match.source_video_path
                output_dir = config.get("output_dir") or os.path.join(settings.output_root, job.id)
                ensure_dir(output_dir)
                append_job_log(
                    session=session,
                    job_id=job.id,
                    tenant_id=job.tenant_id,
                    level="debug",
                    stage="initializing",
                    message="Resolved runtime configuration",
                    detail_level="detailed",
                    data={
                        "video_path": video_path,
                        "output_dir": output_dir,
                        "execution_mode": settings.job_execution_mode,
                    },
                )
                append_job_log(
                    session=session,
                    job_id=job.id,
                    tenant_id=job.tenant_id,
                    level="debug",
                    stage="initializing",
                    message="Raw job configuration",
                    detail_level="extreme",
                    data={"config": config},
                )

                if not os.path.exists(video_path):
                    job.status = "failed"
                    job.error_message = f"Video path not found: {video_path}"
                    job.updated_at = _utcnow()
                    job.completed_at = _utcnow()
                    append_job_log(
                        session=session,
                        job_id=job.id,
                        tenant_id=job.tenant_id,
                        level="error",
                        stage="failed",
                        message="Video path does not exist",
                        detail_level="basic",
                        data={"video_path": video_path},
                    )
                    return

                job.stage = "processing_video"
                job.progress = 0.05
                job.updated_at = _utcnow()
                append_job_log(
                    session=session,
                    job_id=job.id,
                    tenant_id=job.tenant_id,
                    level="info",
                    stage="processing_video",
                    message="Video processing started",
                    detail_level="basic",
                )

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
                if job.cancel_requested:
                    job.status = "canceled"
                    job.stage = "canceled"
                    job.progress = 1.0
                    job.completed_at = _utcnow()
                    job.updated_at = _utcnow()
                    append_job_log(
                        session=session,
                        job_id=job.id,
                        tenant_id=job.tenant_id,
                        level="warning",
                        stage="canceled",
                        message="Job canceled before pipeline invocation",
                        detail_level="basic",
                    )
                    return
                config = job.config_json or {}
                match = session.get(Match, job.match_id)
                if not match:
                    job.status = "failed"
                    job.error_message = f"Match not found for job {job_id}"
                    job.updated_at = _utcnow()
                    job.completed_at = _utcnow()
                    append_job_log(
                        session=session,
                        job_id=job.id,
                        tenant_id=job.tenant_id,
                        level="error",
                        stage="failed",
                        message="Match not found before pipeline run",
                        detail_level="basic",
                        data={"match_id": job.match_id},
                    )
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
                focus_event_types=list(config.get("focus_event_types", []) or []),
                model_version=str(config.get("model_version")) if config.get("model_version") else None,
                analysis_only=bool(config.get("analysis_only", False)),
            )

            artifacts = sorted(str(path.resolve()) for path in Path(output_dir).glob("*.mp4"))
            analysis_manifest = _read_analysis_manifest(output_dir)
            bookmarks = list(analysis_manifest.get("bookmarks", []) or [])
            result_payload = {
                "output_dir": str(Path(output_dir).resolve()),
                "artifact_count": len(artifacts),
                "artifacts": artifacts,
                "engine": "VideoHighlights.py",
                "model_version": config.get("model_version"),
                "focus_event_types": config.get("focus_event_types", []),
                "analysis_only": bool(config.get("analysis_only", False)),
                "bookmarks_count": len(bookmarks),
                "bookmarks": bookmarks,
                "analysis_manifest_path": str((Path(output_dir) / "analysis_bookmarks.json").resolve()),
                "analysis_table_csv_path": str((Path(output_dir) / "analysis_bookmarks.csv").resolve()),
            }

            with session_scope() as session:
                job = session.get(ProcessingJob, job_id)
                if not job:
                    return
                match = session.get(Match, job.match_id)
                if not match:
                    return

                if success:
                    created_events = _sync_job_events_from_manifest(
                        session=session,
                        job=job,
                        match=match,
                        config=config,
                        manifest=analysis_manifest,
                    )
                    if job.cancel_requested:
                        job.status = "canceled"
                        job.stage = "canceled"
                        job.progress = 1.0
                        job.result_json = result_payload
                        job.error_message = "Job canceled after pipeline completion"
                        append_job_log(
                            session=session,
                            job_id=job.id,
                            tenant_id=job.tenant_id,
                            level="warning",
                            stage="canceled",
                            message="Job was cancel_requested; marked canceled after pipeline return",
                            detail_level="detailed",
                        )
                    else:
                        job.status = "completed"
                        job.stage = "completed"
                        job.progress = 1.0
                        job.result_json = result_payload
                        job.error_message = None
                        append_job_log(
                            session=session,
                            job_id=job.id,
                            tenant_id=job.tenant_id,
                            level="info",
                            stage="completed",
                            message="Job completed successfully",
                            detail_level="basic",
                            data={"artifact_count": len(artifacts)},
                        )
                        append_job_log(
                            session=session,
                            job_id=job.id,
                            tenant_id=job.tenant_id,
                            level="info",
                            stage="completed",
                            message="Bookmark analysis persisted",
                            detail_level="detailed",
                            data={"bookmarks_count": len(bookmarks), "events_created": created_events},
                        )
                        append_job_log(
                            session=session,
                            job_id=job.id,
                            tenant_id=job.tenant_id,
                            level="debug",
                            stage="completed",
                            message="Job artifacts",
                            detail_level="extreme",
                            data={"artifacts": artifacts},
                        )
                else:
                    job.status = "failed"
                    job.stage = "failed"
                    job.progress = 1.0
                    job.result_json = result_payload
                    job.error_message = "Processing pipeline reported failure"
                    append_job_log(
                        session=session,
                        job_id=job.id,
                        tenant_id=job.tenant_id,
                        level="error",
                        stage="failed",
                        message="Processing pipeline returned failure",
                        detail_level="basic",
                    )

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
                append_job_log(
                    session=session,
                    job_id=job.id,
                    tenant_id=job.tenant_id,
                    level="error",
                    stage="failed",
                    message="Unhandled exception in processing job",
                    detail_level="basic",
                    data={"error": str(exc)},
                )
                append_job_log(
                    session=session,
                    job_id=job.id,
                    tenant_id=job.tenant_id,
                    level="debug",
                    stage="failed",
                    message="Unhandled exception traceback",
                    detail_level="extreme",
                    data={"traceback": traceback.format_exc()},
                )

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
