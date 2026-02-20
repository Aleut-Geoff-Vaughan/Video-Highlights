from __future__ import annotations

from fastapi.testclient import TestClient

from backend.config import settings


def test_auth_required_rejects_missing_token(auth_client: TestClient) -> None:
    response = auth_client.post(
        "/v1/matches",
        json={"name": "Auth Match", "source_video_path": "C:/tmp/nonexistent.mp4", "metadata": {}},
    )
    assert response.status_code == 401


def test_auth_accepts_valid_token(auth_client: TestClient) -> None:
    response = auth_client.post(
        "/v1/matches",
        headers={"Authorization": "Bearer admin-token"},
        json={"name": "Auth Match", "source_video_path": "C:/tmp/nonexistent.mp4", "metadata": {}},
    )
    assert response.status_code == 201, response.text


def test_queue_worker_run_once_claims_and_processes(auth_client: TestClient) -> None:
    headers = {"Authorization": "Bearer admin-token"}
    settings.job_execution_mode = "queue"

    create_match = auth_client.post(
        "/v1/matches",
        headers=headers,
        json={"name": "Queue Match", "source_video_path": "C:/tmp/nonexistent_queue.mp4", "metadata": {}},
    )
    assert create_match.status_code == 201, create_match.text
    match_id = create_match.json()["match_id"]

    create_job = auth_client.post(
        f"/v1/matches/{match_id}/jobs",
        headers=headers,
        json={"config": {}},
    )
    assert create_job.status_code == 201, create_job.text
    assert create_job.json()["status"] == "queued"
    job_id = create_job.json()["job_id"]

    run_once = auth_client.post("/v1/jobs/worker/run-once", headers=headers)
    assert run_once.status_code == 200, run_once.text
    assert run_once.json()["worked"] is True
    assert run_once.json()["job_id"] == job_id

    job_state = auth_client.get(f"/v1/jobs/{job_id}", headers=headers)
    assert job_state.status_code == 200
    # Given nonexistent source path, job should fail quickly in worker mode.
    assert job_state.json()["status"] == "failed"

    retry = auth_client.post(f"/v1/jobs/{job_id}/retry", headers=headers)
    assert retry.status_code == 201, retry.text
    retry_job_id = retry.json()["job_id"]
    assert retry_job_id != job_id
    assert retry.json()["status"] == "queued"
