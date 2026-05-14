from __future__ import annotations

from pathlib import Path

from VideoHighlights import TrackPoint, write_full_follow_cam_video


def test_write_full_follow_cam_video_uses_original_window_and_offset(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def _fake_render_follow_cam_clip(**kwargs):  # noqa: ANN003
        calls.update(kwargs)
        output_path = Path(str(kwargs["output_path"]))
        output_path.write_bytes(b"fake-video")
        progress_callback = kwargs.get("progress_callback")
        if callable(progress_callback):
            progress_callback(5, 10)
        return str(output_path.resolve())

    monkeypatch.setattr("VideoHighlights.render_follow_cam_clip", _fake_render_follow_cam_clip)

    player_track = [TrackPoint(t=1.0, xy=(100.0, 50.0))]
    ball_track = [TrackPoint(t=1.5, xy=(130.0, 55.0))]
    progress_events = []

    output = write_full_follow_cam_video(
        video_path="source.mp4",
        interval=(240.0, 360.0),
        out_dir=str(tmp_path),
        player_traj=player_track,
        ball_traj=ball_track,
        camera_mode="follow_action",
        zoom_factor=1.8,
        track_time_offset_seconds=240.0,
        progress_callback=lambda stage, progress, message, data: progress_events.append(
            (stage, progress, message, data)
        ),
    )

    assert output == str((tmp_path / "full_follow_action_zoom.mp4").resolve())
    assert calls["start_seconds"] == 240.0
    assert calls["end_seconds"] == 360.0
    assert calls["zoom_factor"] == 1.8
    assert calls["ball_weight"] == 0.35
    assert calls["player_track"] == [(241.0, 100.0, 50.0)]
    assert calls["ball_track"] == [(241.5, 130.0, 55.0)]
    assert progress_events
    assert progress_events[0][0] == "rendering_full_zoom"
