"""
Basic smoke test for the V1 backend API.

Run:
    python test_api_smoke.py
"""

from fastapi.testclient import TestClient
from uuid import uuid4

from backend.database import init_db
from backend.main import app


def main() -> None:
    init_db()
    client = TestClient(app)

    health = client.get("/v1/health")
    assert health.status_code == 200

    match_resp = client.post(
        "/v1/matches",
        json={
            "name": "Smoke Test Match",
            "home_team_name": "Home",
            "away_team_name": "Away",
            "match_date": "2026-02-20",
            "source_video_path": "C:/tmp/nonexistent.mp4",
            "metadata": {"test": True},
        },
    )
    assert match_resp.status_code == 201, match_resp.text
    match_id = match_resp.json()["match_id"]
    event_id = f"evt_smoke_{uuid4().hex[:12]}"

    event_resp = client.put(
        f"/v1/matches/{match_id}/events/{event_id}",
        json={
            "event_type": "goal",
            "status": "auto_detected",
            "confidence": 0.9,
            "period": "1H",
            "occurred_at_ms": 1000,
            "start_ms": 900,
            "end_ms": 1400,
            "frame_index": 30,
            "team_id": "team_home",
            "player_id": "player_10",
            "jersey_number": "10",
            "source": {"detector": "smoke"},
            "location": {"x_norm": 0.5, "y_norm": 0.5, "zone": "center"},
            "participants": [],
            "evidence": {"source_asset_id": "asset_smoke_1"},
            "explanations": [{"signal": "smoke_signal", "value": 0.7}],
        },
    )
    assert event_resp.status_code == 200, event_resp.text

    feedback_resp = client.post(
        f"/v1/matches/{match_id}/events/{event_id}/feedback",
        json={
            "feedback_type": "wrong_timestamp",
            "comment": "smoke correction",
            "submitted_by": {"user_id": "smoke_user", "role": "analyst"},
            "correction": {"corrected_occurred_at_ms": 980, "corrected_start_ms": 900, "corrected_end_ms": 1200},
            "evidence": [],
        },
    )
    assert feedback_resp.status_code == 201, feedback_resp.text

    feedback_id = feedback_resp.json()["feedback_id"]
    review_resp = client.post(
        f"/v1/matches/{match_id}/feedback/{feedback_id}/review",
        json={"review_decision": "approved", "review_note": "smoke ok"},
    )
    assert review_resp.status_code == 200, review_resp.text

    batch_resp = client.post(
        "/v1/training/feedback-batches",
        json={
            "match_ids": [match_id],
            "feedback_status": "approved",
            "feedback_types": ["wrong_timestamp"],
            "from_date": "2026-01-01",
            "to_date": "2026-12-31",
        },
    )
    assert batch_resp.status_code == 201, batch_resp.text
    batch_id = batch_resp.json()["batch_id"]

    run_resp = client.post(
        "/v1/training/runs",
        json={"batch_id": batch_id, "target_model": "event-v0"},
    )
    assert run_resp.status_code == 202, run_resp.text

    run_id = run_resp.json()["run_id"]
    run_get = client.get(f"/v1/training/runs/{run_id}")
    assert run_get.status_code == 200, run_get.text

    query_resp = client.post(
        f"/v1/matches/{match_id}/agent/query",
        json={"query": "Summarize events"},
    )
    assert query_resp.status_code == 200, query_resp.text

    print("API smoke test passed.")


if __name__ == "__main__":
    main()
