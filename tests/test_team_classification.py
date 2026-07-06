from __future__ import annotations

import numpy as np
import pytest

from backend.services.game_tracking import build_ball_track, estimate_field_geometry
from backend.services.team_classification import (
    TeamConfig,
    classify_player_teams,
    compute_team_stats,
    detect_team_colors,
    hex_to_bgr,
)

cv2 = pytest.importorskip("cv2")

W, H = 640, 360
RED = TeamConfig(name="Lions", color_hex="#d32f2f")
BLUE = TeamConfig(name="Hawks", color_hex="#1976d2")


def test_hex_to_bgr() -> None:
    assert hex_to_bgr("#ff0000") == (0, 0, 255)
    assert hex_to_bgr("00ff00") == (0, 255, 0)


def _write_two_team_video(path, frames=60):
    """Red shirts on the left half, blue shirts on the right half."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (W, H))
    assert writer.isOpened()
    positions = []
    red_spots = [(120, 150), (180, 240), (240, 120)]
    blue_spots = [(420, 160), (480, 250), (540, 130)]
    for i in range(frames):
        t = i / 10.0
        frame = np.full((H, W, 3), (40, 90, 40), dtype=np.uint8)
        for x, y in red_spots:
            cv2.rectangle(frame, (x - 9, y - 12), (x + 9, y + 12), hex_to_bgr(RED.color_hex), -1)
            positions.append((t, float(x), float(y)))
        for x, y in blue_spots:
            cv2.rectangle(frame, (x - 9, y - 12), (x + 9, y + 12), hex_to_bgr(BLUE.color_hex), -1)
            positions.append((t, float(x), float(y)))
        writer.write(frame)
    writer.release()
    return np.asarray(positions, dtype=np.float64)


def test_classify_player_teams_by_jersey_color(tmp_path) -> None:
    video = tmp_path / "teams.mp4"
    positions = _write_two_team_video(video)

    labeled = classify_player_teams(str(video), positions, RED, BLUE)

    assert len(labeled) > 20
    known = labeled[labeled[:, 3] >= 0]
    assert len(known) >= 0.7 * len(labeled), "too many unknowns"
    # Left-half positions must be team 0 (red), right-half team 1 (blue).
    left = known[known[:, 1] < W / 2]
    right = known[known[:, 1] >= W / 2]
    assert len(left) and np.mean(left[:, 3] == 0) > 0.9
    assert len(right) and np.mean(right[:, 3] == 1) > 0.9


def test_compute_team_stats_possession_sides_and_goal_attribution() -> None:
    rng = np.random.default_rng(3)
    n = 3000
    cloud = np.stack(
        [rng.uniform(0, 60, n), rng.uniform(100, 1820, n), rng.uniform(200, 880, n)],
        axis=1,
    )
    geometry = estimate_field_geometry(cloud, (1920, 1080))

    # Labeled positions across 60s: Lions (0) on the left, Hawks (1) right;
    # the ball lives on the LEFT side, next to Lions players -> Lions possession.
    rows = []
    for t in np.arange(0.0, 60.0, 0.5):
        rows.append((t, 500.0, 500.0, 0))
        rows.append((t, 520.0, 560.0, 0))
        rows.append((t, 1400.0, 500.0, 1))
    labeled = np.asarray(rows, dtype=np.float32)
    ball = build_ball_track([(t, 505.0, 510.0) for t in np.arange(0.0, 60.0, 0.5)], (1920, 1080))

    goal_events = [{"t": 30.0, "side": "left", "confidence": 0.9}]
    stats = compute_team_stats(labeled, ball, geometry, goal_events, RED, BLUE, 60.0)

    assert stats["defending_side"]["Lions"] == "left"
    assert stats["defending_side"]["Hawks"] == "right"
    assert stats["possession_pct"]["Lions"] > 80.0
    # Goal INTO the left goal (defended by Lions) is scored BY Hawks.
    assert stats["goals"]["Hawks"] == 1
    assert stats["goals"]["Lions"] == 0
    assert stats["goal_attribution"][0]["team"] == "Hawks"


def test_compute_team_stats_empty_labels_is_graceful() -> None:
    rng = np.random.default_rng(3)
    cloud = np.stack([rng.uniform(0, 60, 200), rng.uniform(100, 1820, 200), rng.uniform(200, 880, 200)], axis=1)
    geometry = estimate_field_geometry(cloud, (1920, 1080))
    stats = compute_team_stats(np.empty((0, 4)), None, geometry, [], RED, BLUE, 60.0)
    assert "note" in stats


def _bgr_dist(hex_a: str, hex_b: str) -> float:
    a, b = np.asarray(hex_to_bgr(hex_a), float), np.asarray(hex_to_bgr(hex_b), float)
    return float(np.linalg.norm(a - b))


def test_detect_team_colors_finds_both_kits(tmp_path) -> None:
    video = tmp_path / "kits.mp4"
    positions = _write_two_team_video(video)

    detected = detect_team_colors(str(video), positions)

    assert detected is not None
    hex_a, hex_b = detected
    # Detected pair must be two genuinely different colors...
    assert _bgr_dist(hex_a, hex_b) > 60.0
    # ...and each configured kit must be close to one of them (order-free).
    for kit in (RED.color_hex, BLUE.color_hex):
        assert min(_bgr_dist(kit, hex_a), _bgr_dist(kit, hex_b)) < 90.0, (
            f"kit {kit} not matched by detected {hex_a}/{hex_b}"
        )


def test_detect_team_colors_needs_enough_signal(tmp_path) -> None:
    video = tmp_path / "kits2.mp4"
    _write_two_team_video(video)
    # Too few tracked positions -> refuse to guess.
    assert detect_team_colors(str(video), None) is None
    few = np.asarray([(0.5, 120.0, 150.0)] * 10, dtype=np.float64)
    assert detect_team_colors(str(video), few) is None
