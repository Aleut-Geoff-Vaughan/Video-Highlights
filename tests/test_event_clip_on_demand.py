from __future__ import annotations

from pathlib import Path
from uuid import uuid4


def _create_match(client, source_video_path: str) -> str:
    response = client.post(
        "/v1/matches",
        json={
            "name": "Clip Match",
            "source_video_path": source_video_path,
            "metadata": {},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["match_id"]


def _create_event(client, match_id: str) -> str:
    event_id = f"evt_clip_{uuid4().hex[:10]}"
    response = client.put(
        f"/v1/matches/{match_id}/events/{event_id}",
        json={
            "event_type": "goal",
            "status": "auto_detected",
            "confidence": 0.91,
            "period": "1H",
            "occurred_at_ms": 20000,
            "start_ms": 19000,
            "end_ms": 22500,
            "frame_index": 100,
            "source": {"detector": "test"},
            "location": {},
            "participants": [],
            "evidence": {},
            "explanations": [],
        },
    )
    assert response.status_code == 200, response.text
    return event_id


def test_event_clip_on_demand_create_and_cache(client, tmp_path: Path, monkeypatch) -> None:
    source_video = tmp_path / "source_video.mp4"
    source_video.write_bytes(b"fake-video")
    match_id = _create_match(client, str(source_video))
    event_id = _create_event(client, match_id)

    render_calls = []

    def _fake_render(video_path, output_path, start_seconds, end_seconds, include_audio, prefer_gpu):
        render_calls.append(
            {
                "video_path": video_path,
                "output_path": output_path,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "include_audio": include_audio,
                "prefer_gpu": prefer_gpu,
            }
        )
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake-clip-binary")
        return str(out)

    monkeypatch.setattr("backend.routers.events.render_clip_ffmpeg", _fake_render)

    payload = {
        "pre_seconds": 1.5,
        "post_seconds": 5.0,
        "anchor": "event_window",
        "include_audio": True,
        "prefer_gpu": False,
        "force_rebuild": False,
    }
    first = client.post(f"/v1/matches/{match_id}/events/{event_id}/clip-on-demand", json=payload)
    assert first.status_code == 200, first.text
    first_payload = first.json()
    assert first_payload["reused_existing"] is False
    assert first_payload["event_id"] == event_id
    assert first_payload["asset_id"]
    assert first_payload["path"]
    assert first_payload["download_url"]

    second = client.post(f"/v1/matches/{match_id}/events/{event_id}/clip-on-demand", json=payload)
    assert second.status_code == 200, second.text
    second_payload = second.json()
    assert second_payload["reused_existing"] is True
    assert second_payload["asset_id"] == first_payload["asset_id"]
    assert len(render_calls) == 1

    match = client.get(f"/v1/matches/{match_id}")
    assert match.status_code == 200, match.text
    metadata = match.json().get("metadata") or {}
    generated = list(metadata.get("generated_clips", []))
    assert generated
    assert any(item.get("event_id") == event_id for item in generated)

