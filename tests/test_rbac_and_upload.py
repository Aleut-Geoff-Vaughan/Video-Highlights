from __future__ import annotations

from fastapi.testclient import TestClient


def test_parent_role_cannot_create_match(auth_client: TestClient) -> None:
    response = auth_client.post(
        "/v1/matches",
        headers={"Authorization": "Bearer parent-token"},
        json={"name": "Parent Attempt", "source_video_path": "C:/tmp/nonexistent.mp4", "metadata": {}},
    )
    assert response.status_code == 403


def test_coach_can_upload_asset(auth_client: TestClient) -> None:
    coach_headers = {"Authorization": "Bearer coach-token"}
    create_match = auth_client.post(
        "/v1/matches",
        headers=coach_headers,
        json={"name": "Upload Match", "source_video_path": "", "metadata": {}},
    )
    assert create_match.status_code == 201, create_match.text
    match_id = create_match.json()["match_id"]

    files = {"file": ("small.mp4", b"fake-binary-content", "video/mp4")}
    upload = auth_client.post(
        f"/v1/matches/{match_id}/assets/upload",
        headers=coach_headers,
        files=files,
    )
    assert upload.status_code == 201, upload.text
    payload = upload.json()
    assert payload["match_id"] == match_id
    assert payload["size_bytes"] > 0
    assert payload["storage_backend"] == "local"


def test_coach_cannot_run_worker_once(auth_client: TestClient) -> None:
    response = auth_client.post(
        "/v1/jobs/worker/run-once",
        headers={"Authorization": "Bearer coach-token"},
    )
    assert response.status_code == 403
