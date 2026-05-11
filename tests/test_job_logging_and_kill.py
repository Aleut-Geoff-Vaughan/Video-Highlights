from __future__ import annotations

from fastapi.testclient import TestClient

from backend.config import settings


def _create_match(client: TestClient) -> str:
    response = client.post(
        "/v1/matches",
        json={
            "name": "Logging Match",
            "source_video_path": "C:/tmp/nonexistent_for_logs.mp4",
            "metadata": {},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["match_id"]


def test_job_logs_created_and_queryable(client: TestClient) -> None:
    previous = settings.job_log_detail
    settings.job_log_detail = "extreme"
    try:
        match_id = _create_match(client)

        job = client.post(f"/v1/matches/{match_id}/jobs", json={"config": {"overlay": False}})
        assert job.status_code == 201, job.text
        job_id = job.json()["job_id"]

        logs = client.get(f"/v1/jobs/{job_id}/logs?limit=50")
        assert logs.status_code == 200, logs.text
        items = logs.json()["items"]
        assert len(items) >= 1
        assert any(item["message"] == "Processing job created" for item in items)
    finally:
        settings.job_log_detail = previous


def test_kill_session_marks_cancel_requested(client: TestClient) -> None:
    match_id = _create_match(client)
    job = client.post(f"/v1/matches/{match_id}/jobs", json={"config": {}})
    assert job.status_code == 201
    job_id = job.json()["job_id"]

    kill = client.post(f"/v1/jobs/{job_id}/kill-session")
    assert kill.status_code == 200, kill.text
    payload = kill.json()
    assert payload["cancel_requested"] is True
    assert payload["status"] == "canceled"

    logs = client.get(f"/v1/jobs/{job_id}/logs?limit=20")
    assert logs.status_code == 200
    assert any("canceled" in item["message"].lower() for item in logs.json()["items"])


def test_job_failure_produces_error_log(client: TestClient) -> None:
    previous = settings.job_log_detail
    settings.job_log_detail = "extreme"
    try:
        match_id = _create_match(client)
        job = client.post(f"/v1/matches/{match_id}/jobs", json={"config": {}})
        assert job.status_code == 201
        job_id = job.json()["job_id"]

        run_once = client.post("/v1/jobs/worker/run-once")
        assert run_once.status_code == 200, run_once.text
        assert run_once.json()["job_id"] == job_id

        logs = client.get(f"/v1/jobs/{job_id}/logs?level=error&limit=100")
        assert logs.status_code == 200, logs.text
        items = logs.json()["items"]
        assert any("video path does not exist" in item["message"].lower() for item in items)

        diagnostics = client.get(f"/v1/jobs/{job_id}/diagnostics")
        assert diagnostics.status_code == 200, diagnostics.text
        payload = diagnostics.json()
        assert payload["severity"] == "error"
        assert "video path" in payload["summary"].lower()
        assert "re-register" in payload["next_action"].lower()
        assert payload["error_logs"]
    finally:
        settings.job_log_detail = previous


def test_run_log_profile_persists_process_and_technical_language(client: TestClient) -> None:
    match_id = _create_match(client)
    job = client.post(
        f"/v1/matches/{match_id}/jobs",
        json={"config": {"log_profile": "detailed", "analysis_only": True}},
    )
    assert job.status_code == 201
    job_id = job.json()["job_id"]

    run_once = client.post("/v1/jobs/worker/run-once")
    assert run_once.status_code == 200, run_once.text
    assert run_once.json()["job_id"] == job_id

    logs = client.get(f"/v1/jobs/{job_id}/logs?limit=100")
    assert logs.status_code == 200, logs.text
    items = logs.json()["items"]
    process_logs = [
        item
        for item in items
        if isinstance(item.get("data"), dict)
        and item["data"].get("process_message")
        and item["data"].get("technical_message")
    ]
    assert process_logs
    assert any(item["message"] == "Run plan assembled" for item in process_logs)


def test_rerun_job_with_overrides_persists_processing_history(client: TestClient) -> None:
    match_id = _create_match(client)
    create = client.post(
        f"/v1/matches/{match_id}/jobs",
        json={
            "config": {
                "model_version": "event-v0",
                "focus_event_types": ["goal", "shot"],
                "pre_seconds": 2.0,
                "post_seconds": 6.0,
            }
        },
    )
    assert create.status_code == 201, create.text
    source_job_id = create.json()["job_id"]

    rerun = client.post(
        f"/v1/jobs/{source_job_id}/rerun",
        json={"config_overrides": {"model_version": "event-v1", "focus_event_types": ["corner_kick"]}, "reason": "model-upgrade"},
    )
    assert rerun.status_code == 201, rerun.text
    rerun_payload = rerun.json()
    assert rerun_payload["job_id"] != source_job_id
    assert rerun_payload["config"]["model_version"] == "event-v1"
    assert rerun_payload["config"]["focus_event_types"] == ["corner_kick"]

    match = client.get(f"/v1/matches/{match_id}")
    assert match.status_code == 200, match.text
    history = (match.json().get("metadata") or {}).get("processing_history", [])
    assert len(history) >= 2
    assert history[-1]["source_job_id"] == source_job_id
    assert history[-1]["model_version"] == "event-v1"
