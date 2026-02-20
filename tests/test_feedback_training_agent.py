from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient


def _create_match_and_event(client: TestClient) -> tuple[str, str]:
    match = client.post(
        "/v1/matches",
        json={
            "name": "Feedback Match",
            "source_video_path": "C:/tmp/nonexistent.mp4",
            "metadata": {},
        },
    )
    assert match.status_code == 201, match.text
    match_id = match.json()["match_id"]

    event_id = f"evt_fb_{uuid4().hex[:10]}"
    event = client.put(
        f"/v1/matches/{match_id}/events/{event_id}",
        json={
            "event_type": "goal",
            "status": "auto_detected",
            "confidence": 0.8,
            "period": "2H",
            "occurred_at_ms": 2000,
            "start_ms": 1700,
            "end_ms": 2500,
            "frame_index": 60,
            "team_id": "team_home",
            "player_id": "player_9",
            "jersey_number": "9",
            "source": {"detector": "test"},
            "location": {"x_norm": 0.5, "y_norm": 0.5, "zone": "center"},
            "participants": [],
            "evidence": {"source_asset_id": "asset"},
            "explanations": [{"signal": "s", "value": 0.6}],
        },
    )
    assert event.status_code == 200, event.text
    return match_id, event_id


def test_feedback_review_updates_event(client: TestClient) -> None:
    match_id, event_id = _create_match_and_event(client)

    feedback = client.post(
        f"/v1/matches/{match_id}/events/{event_id}/feedback",
        json={
            "feedback_type": "wrong_timestamp",
            "comment": "late timestamp",
            "submitted_by": {"user_id": "coach_1", "role": "coach"},
            "correction": {
                "expected_event_type": "goal",
                "corrected_occurred_at_ms": 1800,
                "corrected_start_ms": 1600,
                "corrected_end_ms": 2200,
            },
            "evidence": [],
        },
    )
    assert feedback.status_code == 201, feedback.text
    feedback_id = feedback.json()["feedback_id"]

    review = client.post(
        f"/v1/matches/{match_id}/feedback/{feedback_id}/review",
        json={"review_decision": "approved", "review_note": "validated"},
    )
    assert review.status_code == 200, review.text
    assert review.json()["status"] == "approved"

    event = client.get(f"/v1/matches/{match_id}/events/{event_id}")
    assert event.status_code == 200
    payload = event.json()
    assert payload["occurred_at_ms"] == 1800
    assert payload["status"] == "corrected"


def test_missed_event_feedback_creates_event_on_approval(client: TestClient) -> None:
    match = client.post(
        "/v1/matches",
        json={"name": "Missed Event Match", "source_video_path": "C:/tmp/nonexistent.mp4", "metadata": {}},
    )
    assert match.status_code == 201
    match_id = match.json()["match_id"]

    feedback = client.post(
        f"/v1/matches/{match_id}/feedback",
        json={
            "feedback_type": "missed_event",
            "comment": "missed penalty",
            "submitted_by": {"user_id": "analyst_1", "role": "analyst"},
            "correction": {
                "expected_event_type": "penalty_kick",
                "corrected_occurred_at_ms": 5000,
                "corrected_start_ms": 4700,
                "corrected_end_ms": 5400,
                "corrected_team_id": "team_away",
            },
            "evidence": [],
        },
    )
    assert feedback.status_code == 201, feedback.text
    feedback_id = feedback.json()["feedback_id"]

    review = client.post(
        f"/v1/matches/{match_id}/feedback/{feedback_id}/review",
        json={"review_decision": "approved", "review_note": "create event"},
    )
    assert review.status_code == 200

    events = client.get(f"/v1/matches/{match_id}/events?event_type=penalty_kick")
    assert events.status_code == 200
    assert len(events.json()["items"]) >= 1


def test_training_batch_and_run_and_agent_query(client: TestClient) -> None:
    match_id, event_id = _create_match_and_event(client)

    feedback = client.post(
        f"/v1/matches/{match_id}/events/{event_id}/feedback",
        json={
            "feedback_type": "wrong_timestamp",
            "comment": "for training",
            "submitted_by": {"user_id": "analyst_1", "role": "analyst"},
            "correction": {"corrected_occurred_at_ms": 1900, "corrected_start_ms": 1600, "corrected_end_ms": 2200},
            "evidence": [],
        },
    )
    assert feedback.status_code == 201
    feedback_id = feedback.json()["feedback_id"]
    review = client.post(
        f"/v1/matches/{match_id}/feedback/{feedback_id}/review",
        json={"review_decision": "approved", "review_note": "approved for train"},
    )
    assert review.status_code == 200

    batch = client.post(
        "/v1/training/feedback-batches",
        json={
            "match_ids": [match_id],
            "feedback_status": "approved",
            "feedback_types": ["wrong_timestamp"],
            "from_date": "2026-01-01",
            "to_date": "2026-12-31",
        },
    )
    assert batch.status_code == 201, batch.text
    batch_id = batch.json()["batch_id"]
    assert batch.json()["item_count"] >= 1

    run = client.post(
        "/v1/training/runs",
        json={"batch_id": batch_id, "target_model": "event-v0", "notes": "pytest run"},
    )
    assert run.status_code == 202, run.text
    run_id = run.json()["run_id"]

    run_get = client.get(f"/v1/training/runs/{run_id}")
    assert run_get.status_code == 200
    assert run_get.json()["status"] in {"completed", "running", "evaluating", "queued"}

    if run_get.json()["status"] == "completed":
        promote = client.post(
            f"/v1/training/runs/{run_id}/promote",
            json={"decision": "approved", "reason": "pytest promotion", "notes": "promoted by test"},
        )
        if promote.status_code == 409:
            # Candidate can fail gates for small feedback sets; force promotion in test path.
            promote = client.post(
                f"/v1/training/runs/{run_id}/promote",
                json={
                    "decision": "approved",
                    "reason": "pytest force promotion",
                    "notes": "forced promotion by test",
                    "force": True,
                },
            )
        assert promote.status_code == 200, promote.text
        model_id = promote.json()["model_id"]
        assert model_id

        models = client.get("/v1/training/models")
        assert models.status_code == 200, models.text
        assert any(item["model_id"] == model_id for item in models.json())

    agent = client.post(
        f"/v1/matches/{match_id}/agent/query",
        json={"query": "summarize key events", "include_event_limit": 20},
    )
    assert agent.status_code == 200, agent.text
    assert "answer" in agent.json()
