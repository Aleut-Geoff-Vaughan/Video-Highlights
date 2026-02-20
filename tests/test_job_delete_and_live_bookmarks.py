from __future__ import annotations

from uuid import uuid4


def _create_match(client) -> str:
    response = client.post(
        "/v1/matches",
        json={
            "name": "Delete + Bookmarks Match",
            "source_video_path": "C:/tmp/nonexistent_delete_flow.mp4",
            "metadata": {},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["match_id"]


def test_job_bookmarks_endpoint_uses_events_when_present(client) -> None:
    match_id = _create_match(client)
    job = client.post(f"/v1/matches/{match_id}/jobs", json={"config": {}})
    assert job.status_code == 201, job.text
    job_id = job.json()["job_id"]

    event_id = f"evt_live_{uuid4().hex[:10]}"
    put = client.put(
        f"/v1/matches/{match_id}/events/{event_id}",
        json={
            "event_type": "goal",
            "status": "auto_detected",
            "confidence": 0.88,
            "period": "1H",
            "occurred_at_ms": 2000,
            "start_ms": 1500,
            "end_ms": 2500,
            "frame_index": 10,
            "job_id": job_id,
            "source": {"detector": "unit"},
            "location": {},
            "participants": [],
            "evidence": {},
            "explanations": [],
        },
    )
    assert put.status_code == 200, put.text

    bookmarks = client.get(f"/v1/jobs/{job_id}/bookmarks")
    assert bookmarks.status_code == 200, bookmarks.text
    payload = bookmarks.json()
    assert payload["source"] == "events"
    assert any(item["event_id"] == event_id for item in payload["items"])


def test_delete_job_removes_job_logs_and_events(client) -> None:
    match_id = _create_match(client)
    create = client.post(f"/v1/matches/{match_id}/jobs", json={"config": {}})
    assert create.status_code == 201, create.text
    job_id = create.json()["job_id"]

    event_id = f"evt_del_{uuid4().hex[:10]}"
    put = client.put(
        f"/v1/matches/{match_id}/events/{event_id}",
        json={
            "event_type": "shot",
            "status": "auto_detected",
            "confidence": 0.65,
            "period": "1H",
            "occurred_at_ms": 5000,
            "start_ms": 4500,
            "end_ms": 5500,
            "frame_index": 20,
            "job_id": job_id,
            "source": {},
            "location": {},
            "participants": [],
            "evidence": {},
            "explanations": [],
        },
    )
    assert put.status_code == 200, put.text

    deleted = client.delete(f"/v1/jobs/{job_id}")
    assert deleted.status_code == 200, deleted.text
    payload = deleted.json()
    assert payload["deleted"] is True
    assert payload["deleted_logs"] >= 1
    assert payload["deleted_events"] >= 1

    missing = client.get(f"/v1/jobs/{job_id}")
    assert missing.status_code == 404

    events = client.get(f"/v1/matches/{match_id}/events?job_id={job_id}")
    assert events.status_code == 200, events.text
    assert events.json()["items"] == []

