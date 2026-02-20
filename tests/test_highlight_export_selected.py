from __future__ import annotations

from pathlib import Path
from uuid import uuid4


def _create_match(client, source_video_path: str) -> str:
    response = client.post(
        "/v1/matches",
        json={
            "name": "Export Match",
            "source_video_path": source_video_path,
            "metadata": {},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["match_id"]


def _create_event(client, match_id: str, event_type: str, occurred_ms: int) -> str:
    event_id = f"evt_export_{uuid4().hex[:10]}"
    response = client.put(
        f"/v1/matches/{match_id}/events/{event_id}",
        json={
            "event_type": event_type,
            "status": "auto_detected",
            "confidence": 0.75,
            "period": "1H",
            "occurred_at_ms": occurred_ms,
            "start_ms": occurred_ms - 500,
            "end_ms": occurred_ms + 500,
            "frame_index": 0,
            "source": {},
            "location": {},
            "participants": [],
            "evidence": {},
            "explanations": [],
        },
    )
    assert response.status_code == 200, response.text
    return event_id


def test_export_selected_highlights(client, tmp_path: Path, monkeypatch) -> None:
    source_video = tmp_path / "source_video.mp4"
    source_video.write_bytes(b"fake-video")
    match_id = _create_match(client, str(source_video))
    event_1 = _create_event(client, match_id, "goal", 10_000)
    event_2 = _create_event(client, match_id, "shot", 15_000)

    def _fake_render(video_path, output_path, start_seconds, end_seconds, include_audio, prefer_gpu):
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"clip")
        return str(out)

    def _fake_concat(clip_paths, output_path, include_audio=True):
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"joined")
        return str(out)

    monkeypatch.setattr("backend.routers.events.render_clip_ffmpeg", _fake_render)
    monkeypatch.setattr("backend.routers.events.concat_clips_ffmpeg", _fake_concat)

    export = client.post(
        f"/v1/matches/{match_id}/exports/highlights",
        json={
            "event_ids": [event_1, event_2],
            "pre_seconds": 1.0,
            "post_seconds": 3.0,
            "anchor": "event_window",
            "include_audio": True,
            "prefer_gpu": False,
            "title": "Coach Cut",
        },
    )
    assert export.status_code == 200, export.text
    payload = export.json()
    assert payload["match_id"] == match_id
    assert payload["clip_count"] == 2
    assert payload["asset_id"]
    assert payload["path"]
    assert payload["download_url"]
    assert payload["event_ids"] == [event_1, event_2]

    match = client.get(f"/v1/matches/{match_id}")
    assert match.status_code == 200, match.text
    metadata = match.json().get("metadata") or {}
    exports = list(metadata.get("highlight_exports", []))
    assert any(item.get("export_id") == payload["export_id"] for item in exports)

