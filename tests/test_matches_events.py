from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient


def _create_match(client: TestClient) -> str:
    response = client.post(
        "/v1/matches",
        json={
            "name": "Integration Match",
            "home_team_name": "Home",
            "away_team_name": "Away",
            "match_date": "2026-02-20",
            "source_video_path": "C:/tmp/nonexistent.mp4",
            "metadata": {"competition": "U12"},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["match_id"]


def test_create_and_list_matches(client: TestClient) -> None:
    match_id = _create_match(client)

    single = client.get(f"/v1/matches/{match_id}")
    assert single.status_code == 200
    assert single.json()["match_id"] == match_id

    listing = client.get("/v1/matches")
    assert listing.status_code == 200
    items = listing.json()["items"]
    assert any(item["match_id"] == match_id for item in items)


def test_event_upsert_patch_and_filter(client: TestClient) -> None:
    match_id = _create_match(client)
    event_id = f"evt_test_{uuid4().hex[:10]}"
    job_id = f"job_test_{uuid4().hex[:8]}"

    put_resp = client.put(
        f"/v1/matches/{match_id}/events/{event_id}",
        json={
            "event_type": "goal",
            "status": "auto_detected",
            "confidence": 0.91,
            "period": "1H",
            "occurred_at_ms": 1000,
            "start_ms": 900,
            "end_ms": 1500,
            "frame_index": 30,
            "team_id": "team_home",
            "player_id": "player_10",
            "jersey_number": "10",
            "source": {"detector": "test"},
            "location": {"x_norm": 0.4, "y_norm": 0.2, "zone": "box"},
            "participants": [],
            "evidence": {"source_asset_id": "asset_test"},
            "explanations": [{"signal": "sig", "value": 0.8}],
            "job_id": job_id,
        },
    )
    assert put_resp.status_code == 200, put_resp.text
    assert put_resp.json()["event_id"] == event_id

    patch_resp = client.patch(
        f"/v1/matches/{match_id}/events/{event_id}",
        json={"status": "confirmed", "confidence": 0.99},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["status"] == "confirmed"
    assert abs(patch_resp.json()["confidence"] - 0.99) < 1e-9

    list_resp = client.get(f"/v1/matches/{match_id}/events?event_type=goal&min_confidence=0.95")
    assert list_resp.status_code == 200
    items = list_resp.json()["items"]
    assert any(item["event_id"] == event_id for item in items)

    list_job = client.get(f"/v1/matches/{match_id}/events?job_id={job_id}")
    assert list_job.status_code == 200
    assert any(item["event_id"] == event_id for item in list_job.json()["items"])
