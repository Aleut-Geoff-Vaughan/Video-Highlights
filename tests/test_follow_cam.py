from __future__ import annotations

import numpy as np

from backend.services.follow_cam import build_follow_cam_centers, crop_frame_to_center


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
