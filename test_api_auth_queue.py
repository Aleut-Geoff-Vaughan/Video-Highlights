"""
Auth + queue mode smoke test.

Run:
    python test_api_auth_queue.py
"""

import os

os.environ["VH_JOB_EXECUTION_MODE"] = "queue"
os.environ["VH_AUTH_REQUIRED"] = "true"
os.environ["VH_API_TOKENS"] = "admin-token:admin,coach-token:coach"

from fastapi.testclient import TestClient

from backend.database import init_db
from backend.main import app


def main() -> None:
    init_db()
    client = TestClient(app)

    create_no_auth = client.post(
        "/v1/matches",
        json={"name": "auth queue", "source_video_path": "C:/tmp/none.mp4", "metadata": {}},
    )
    assert create_no_auth.status_code == 401, create_no_auth.text

    admin_headers = {"Authorization": "Bearer admin-token"}
    create_match = client.post(
        "/v1/matches",
        headers=admin_headers,
        json={"name": "auth queue", "source_video_path": "C:/tmp/none.mp4", "metadata": {}},
    )
    assert create_match.status_code == 201, create_match.text
    match_id = create_match.json()["match_id"]

    create_job = client.post(
        f"/v1/matches/{match_id}/jobs",
        headers=admin_headers,
        json={"config": {}},
    )
    assert create_job.status_code == 201, create_job.text
    assert create_job.json()["status"] == "queued"
    job_id = create_job.json()["job_id"]

    worker_once = client.post("/v1/jobs/worker/run-once", headers=admin_headers)
    assert worker_once.status_code == 200, worker_once.text
    assert worker_once.json()["worked"] is True
    assert worker_once.json()["job_id"] == job_id

    get_job = client.get(f"/v1/jobs/{job_id}", headers=admin_headers)
    assert get_job.status_code == 200, get_job.text
    assert get_job.json()["status"] in {"failed", "completed"}

    print("Auth + queue smoke test passed.")


if __name__ == "__main__":
    main()
