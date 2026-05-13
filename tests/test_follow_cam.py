from __future__ import annotations

import numpy as np
import pytest

from backend.services.follow_cam import build_follow_cam_centers, crop_frame_to_center, render_follow_cam_clip


def test_build_follow_cam_centers_tracks_player_motion() -> None:
    player_track = [
        (0.0, 120.0, 80.0),
        (1.0, 280.0, 80.0),
    ]

    centers = build_follow_cam_centers(
        player_track=player_track,
        ball_track=None,
        start_seconds=0.0,
        end_seconds=1.0,
        fps=2.0,
        frame_size=(400, 200),
        zoom_factor=2.0,
        smooth_factor=1.0,
    )

    assert len(centers) == 2
    assert centers[0][0] < centers[1][0]
    assert centers[0][1] == centers[1][1] == 80.0


def test_build_follow_cam_centers_follow_action_blends_ball_position() -> None:
    player_track = [
        (0.0, 180.0, 100.0),
        (1.0, 180.0, 100.0),
    ]
    ball_track = [
        (0.0, 300.0, 100.0),
        (1.0, 300.0, 100.0),
    ]

    wide_centers = build_follow_cam_centers(
        player_track=player_track,
        ball_track=ball_track,
        start_seconds=0.0,
        end_seconds=1.0,
        fps=1.0,
        frame_size=(400, 200),
        zoom_factor=1.6,
        ball_weight=0.0,
        smooth_factor=1.0,
    )
    action_centers = build_follow_cam_centers(
        player_track=player_track,
        ball_track=ball_track,
        start_seconds=0.0,
        end_seconds=1.0,
        fps=1.0,
        frame_size=(400, 200),
        zoom_factor=1.6,
        ball_weight=0.5,
        smooth_factor=1.0,
    )

    assert action_centers[0][0] > wide_centers[0][0]


def test_crop_frame_to_center_preserves_output_size() -> None:
    frame = np.arange(80 * 40 * 3, dtype=np.uint8).reshape((40, 80, 3))
    cropped = crop_frame_to_center(frame, center=(60.0, 20.0), zoom_factor=2.0)

    assert cropped.shape == frame.shape


def test_build_follow_cam_centers_recenters_when_player_track_goes_stale() -> None:
    player_track = [
        (0.0, 80.0, 100.0),
    ]

    centers = build_follow_cam_centers(
        player_track=player_track,
        ball_track=None,
        start_seconds=0.0,
        end_seconds=2.1,
        fps=1.0,
        frame_size=(400, 200),
        zoom_factor=2.0,
        smooth_factor=1.0,
        max_player_gap_seconds=0.25,
    )

    assert centers[0] == (100.0, 100.0)
    assert centers[1][0] > centers[0][0]
    assert centers[2][0] > centers[1][0]
    assert centers[2][0] <= 200.0
    assert centers[1][1] == centers[2][1] == 100.0


def test_render_follow_cam_clip_writes_zoomed_video(tmp_path) -> None:
    cv2 = pytest.importorskip("cv2")
    source_path = tmp_path / "source.mp4"
    output_path = tmp_path / "zoomed.mp4"
    fps = 5.0
    width, height = 64, 48
    writer = cv2.VideoWriter(
        str(source_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    assert writer.isOpened()

    player_track = []
    try:
        for frame_index in range(12):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            x = 16 + (frame_index * 3)
            y = 24
            cv2.rectangle(frame, (x - 3, y - 3), (x + 3, y + 3), (0, 255, 0), thickness=-1)
            writer.write(frame)
            player_track.append((frame_index / fps, float(x), float(y)))
    finally:
        writer.release()

    rendered = render_follow_cam_clip(
        video_path=str(source_path),
        output_path=str(output_path),
        start_seconds=0.0,
        end_seconds=2.0,
        player_track=player_track,
        ball_track=None,
        zoom_factor=2.0,
        include_audio=False,
    )

    assert rendered == str(output_path.resolve())
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    cap = cv2.VideoCapture(str(output_path))
    ok, first_frame = cap.read()
    cap.release()

    assert ok
    center_region = first_frame[height // 2 - 4 : height // 2 + 5, width // 2 - 4 : width // 2 + 5]
    assert float(center_region[:, :, 1].mean()) > 60.0
