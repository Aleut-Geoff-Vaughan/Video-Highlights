from __future__ import annotations

from pathlib import Path

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


def test_empty_upload_is_rejected(auth_client: TestClient) -> None:
    coach_headers = {"Authorization": "Bearer coach-token"}
    create_match = auth_client.post(
        "/v1/matches",
        headers=coach_headers,
        json={"name": "Empty Upload Match", "source_video_path": "", "metadata": {}},
    )
    assert create_match.status_code == 201, create_match.text
    match_id = create_match.json()["match_id"]

    upload = auth_client.post(
        f"/v1/matches/{match_id}/assets/upload",
        headers=coach_headers,
        files={"file": ("empty.mp4", b"", "video/mp4")},
    )

    assert upload.status_code == 400
    assert "empty" in upload.json()["error"]["message"].lower()


def test_coach_can_register_local_video_path(auth_client: TestClient, tmp_path: Path) -> None:
    coach_headers = {"Authorization": "Bearer coach-token"}
    local_video = tmp_path / "large-match.mp4"
    local_video.write_bytes(b"not-a-real-video-but-nonempty")

    create_match = auth_client.post(
        "/v1/matches",
        headers=coach_headers,
        json={"name": "Local Path Match", "source_video_path": "", "metadata": {}},
    )
    assert create_match.status_code == 201, create_match.text
    match_id = create_match.json()["match_id"]

    register = auth_client.post(
        f"/v1/matches/{match_id}/assets/register-local",
        headers=coach_headers,
        json={"path": str(local_video), "set_as_source": True},
    )
    assert register.status_code == 201, register.text
    payload = register.json()
    assert payload["storage_backend"] == "local_path"
    assert payload["size_bytes"] == local_video.stat().st_size

    match = auth_client.get(f"/v1/matches/{match_id}", headers=coach_headers)
    assert match.status_code == 200, match.text
    assert match.json()["source_video_path"] == str(local_video.resolve())


def test_coach_can_inspect_local_video_path_before_registering(auth_client: TestClient, tmp_path: Path) -> None:
    coach_headers = {"Authorization": "Bearer coach-token"}
    local_video = tmp_path / "inspect-match.mp4"
    local_video.write_bytes(b"not-a-real-video-but-nonempty")

    inspect = auth_client.post(
        "/v1/matches/assets/inspect-local",
        headers=coach_headers,
        json={"path": str(local_video), "set_as_source": False},
    )

    assert inspect.status_code == 200, inspect.text
    payload = inspect.json()
    assert payload["ok"] is True
    assert payload["code"] == "ready"
    assert payload["path"] == str(local_video.resolve())
    assert payload["size_bytes"] == local_video.stat().st_size


def test_inspect_local_video_path_reports_zero_byte_file(auth_client: TestClient, tmp_path: Path) -> None:
    coach_headers = {"Authorization": "Bearer coach-token"}
    local_video = tmp_path / "still-copying.mp4"
    local_video.write_bytes(b"")

    inspect = auth_client.post(
        "/v1/matches/assets/inspect-local",
        headers=coach_headers,
        json={"path": str(local_video), "set_as_source": False},
    )

    assert inspect.status_code == 200, inspect.text
    payload = inspect.json()
    assert payload["ok"] is False
    assert payload["code"] == "zero_bytes"
    assert "0 bytes" in payload["message"]


def test_coach_cannot_run_worker_once(auth_client: TestClient) -> None:
    response = auth_client.post(
        "/v1/jobs/worker/run-once",
        headers={"Authorization": "Bearer coach-token"},
    )
    assert response.status_code == 403
