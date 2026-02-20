from __future__ import annotations

from fastapi.testclient import TestClient


def test_invalid_event_type_returns_validation_envelope(client: TestClient) -> None:
    match = client.post(
        "/v1/matches",
        json={"name": "Validation Match", "source_video_path": "C:/tmp/nonexistent.mp4", "metadata": {}},
    )
    assert match.status_code == 201
    match_id = match.json()["match_id"]

    invalid = client.put(
        f"/v1/matches/{match_id}/events/evt_invalid_type",
        json={
            "event_type": "not_real",
            "status": "auto_detected",
            "confidence": 0.7,
            "occurred_at_ms": 1000,
            "start_ms": 900,
            "end_ms": 1200,
            "frame_index": 1,
            "source": {},
            "location": {},
            "participants": [],
            "evidence": {},
            "explanations": [],
        },
    )
    assert invalid.status_code == 400
    payload = invalid.json()
    assert "error" in payload
    assert payload["error"]["code"] == "VALIDATION_ERROR"
    assert isinstance(payload["error"]["details"], list)


def test_not_found_uses_http_error_envelope(client: TestClient) -> None:
    missing = client.get("/v1/matches/match_does_not_exist")
    assert missing.status_code == 404
    payload = missing.json()
    assert payload["error"]["code"] == "HTTP_ERROR"
    assert "Match not found" in payload["error"]["message"]
