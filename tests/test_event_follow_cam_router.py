from __future__ import annotations

import json
from pathlib import Path

from backend.models import Event
from backend.routers import events as events_router


def _make_event(**source_overrides) -> Event:
    source = {"camera_mode": "follow_action", "zoom_factor": 1.8}
    source.update(source_overrides)
    return Event(
        tenant_id="tenant_test",
        match_id="match_test",
        event_type="goal",
        source_json=source,
        evidence_json={"tracking_manifest_path": "C:/tmp/analysis_tracking.json"},
    )


def test_resolve_follow_cam_profile_reads_tracking_manifest(monkeypatch) -> None:
    event = _make_event()

    monkeypatch.setattr(
        events_router,
        "_resolve_tracking_manifest",
        lambda event: {
            "camera": {"mode": "follow_action", "zoom_factor": 1.8},
            "tracking": {
                "target_track": [{"t": 1.0, "x": 100.0, "y": 50.0}],
                "ball_track": [{"t": 1.0, "x": 120.0, "y": 55.0}],
            },
        },
    )

    mode, zoom, player_track, ball_track = events_router._resolve_follow_cam_profile(event)
    assert mode == "follow_action"
    assert zoom == 1.8
    assert player_track == [(1.0, 100.0, 50.0)]
    assert ball_track == [(1.0, 120.0, 55.0)]


def test_resolve_follow_cam_profile_follows_analysis_manifest_pointer(tmp_path: Path) -> None:
    tracking_path = tmp_path / "analysis_tracking.json"
    tracking_path.write_text(
        json.dumps(
            {
                "camera": {"mode": "follow_player", "zoom_factor": 1.5},
                "tracking": {
                    "target_track": [{"t": 2.0, "x": 90.0, "y": 45.0}],
                    "ball_track": [],
                },
            }
        ),
        encoding="utf-8",
    )
    analysis_path = tmp_path / "analysis_bookmarks.json"
    analysis_path.write_text(json.dumps({"tracking_manifest_path": str(tracking_path)}), encoding="utf-8")

    event = _make_event(camera_mode="follow_player", zoom_factor=1.5)
    event.evidence_json = {"analysis_manifest_path": str(analysis_path)}

    mode, zoom, player_track, ball_track = events_router._resolve_follow_cam_profile(event)

    assert mode == "follow_player"
    assert zoom == 1.5
    assert player_track == [(2.0, 90.0, 45.0)]
    assert ball_track == []


def test_render_window_clip_uses_follow_cam_when_track_exists(monkeypatch) -> None:
    event = _make_event()
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        events_router,
        "_resolve_follow_cam_profile",
        lambda event: ("follow_action", 1.7, [(1.0, 100.0, 50.0)], [(1.0, 120.0, 55.0)]),
    )

    def _fake_follow_cam(**kwargs):  # noqa: ANN003
        calls["follow_cam"] = kwargs

    monkeypatch.setattr(events_router, "render_follow_cam_clip", _fake_follow_cam)
    monkeypatch.setattr(
        events_router,
        "render_clip_ffmpeg",
        lambda **kwargs: calls.setdefault("wide", kwargs),
    )

    mode, zoom = events_router._render_window_clip(
        source_video="C:/tmp/source.mp4",
        event=event,
        output_path="C:/tmp/follow_cam_clip.mp4",
        start_seconds=1.0,
        end_seconds=5.0,
        include_audio=True,
        prefer_gpu=False,
    )

    assert mode == "follow_action"
    assert zoom == 1.7
    assert "follow_cam" in calls
    assert "wide" not in calls


def test_render_window_clip_falls_back_to_wide_without_track(monkeypatch) -> None:
    event = _make_event(camera_mode="wide")
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        events_router,
        "_resolve_follow_cam_profile",
        lambda event: ("wide", 1.0, [], []),
    )
    monkeypatch.setattr(
        events_router,
        "render_follow_cam_clip",
        lambda **kwargs: calls.setdefault("follow_cam", kwargs),
    )

    def _fake_wide(**kwargs):  # noqa: ANN003
        calls["wide"] = kwargs

    monkeypatch.setattr(events_router, "render_clip_ffmpeg", _fake_wide)

    mode, zoom = events_router._render_window_clip(
        source_video="C:/tmp/source.mp4",
        event=event,
        output_path="C:/tmp/wide_clip.mp4",
        start_seconds=1.0,
        end_seconds=5.0,
        include_audio=True,
        prefer_gpu=False,
    )

    assert mode == "wide"
    assert zoom == 1.0
    assert "wide" in calls
    assert "follow_cam" not in calls
