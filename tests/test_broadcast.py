from __future__ import annotations

import numpy as np
import pytest

from backend.services.broadcast import (
    BroadcastConfig,
    emotion_end,
    refine_intervals,
    story_start,
)
from backend.services.camera_planner import CameraPlannerConfig, plan_camera
from backend.services.game_tracking import (
    GameStateSegment,
    STATE_IN_PLAY,
    build_ball_track,
    estimate_field_geometry,
)

FRAME = (1920, 1080)


def _geometry():
    rng = np.random.default_rng(3)
    n = 3000
    cloud = np.stack(
        [rng.uniform(0, 60, n), rng.uniform(100, 1820, n), rng.uniform(200, 880, n)],
        axis=1,
    )
    return estimate_field_geometry(cloud, FRAME)


def _attack_track(t_turn=20.0, t_goal=28.0):
    """Ball drifts AWAY from the left goal, then turns and attacks it."""
    hz = 10.0
    rows = []
    for i in range(int(t_turn * hz)):
        t = i / hz
        rows.append((t, 600.0 + 20.0 * t, 500.0))  # moving right (away from left goal)
    x0 = 600.0 + 20.0 * t_turn
    for i in range(int((t_goal - t_turn) * hz)):
        t = t_turn + i / hz
        f = (t - t_turn) / (t_goal - t_turn)
        rows.append((t, x0 - (x0 - 160.0) * f, 500.0))  # attacking left
    return build_ball_track(rows, FRAME)


def test_story_start_walks_back_to_attack_origin() -> None:
    track = _attack_track(t_turn=20.0, t_goal=28.0)
    segments = [GameStateSegment(0.0, 30.0, STATE_IN_PLAY, reason="in play")]

    start = story_start(28.0, track, segments, "left")

    # The move began at ~20s (direction turn); with 1.5s preroll the clip
    # starts just before it - NOT at 28-2s and NOT at the 15s lookback cap.
    assert 17.5 <= start <= 21.0, f"story start {start}"


def test_story_start_respects_dead_ball_boundary() -> None:
    track = _attack_track(t_turn=20.0, t_goal=28.0)
    segments = [
        GameStateSegment(0.0, 24.0, "restart_left", side="left", reason="goal kick wait"),
        GameStateSegment(24.0, 30.0, STATE_IN_PLAY, reason="in play"),
    ]
    start = story_start(28.0, track, segments, "left")
    # Cannot start before play resumed at 24s (minus preroll).
    assert start >= 24.0 - BroadcastConfig().preroll_s - 1e-6
    assert start < 26.0


def test_emotion_end_waits_for_crowd_decay() -> None:
    times = np.arange(0.0, 40.0, 0.1)
    rms = np.full_like(times, 0.05)
    # Crowd erupts at t=10 and decays until t=19.
    surge = (times >= 10.0) & (times <= 19.0)
    rms[surge] = 0.5 - 0.05 * (times[surge] - 10.0)

    end = emotion_end(10.0, (times, rms), is_goal=True)

    cfg = BroadcastConfig()
    assert 15.0 <= end <= 10.0 + cfg.max_post_goal_s
    # And well past the minimum post window.
    assert end > 10.0 + cfg.min_post_s


def test_emotion_end_caps_and_handles_missing_audio() -> None:
    cfg = BroadcastConfig()
    assert emotion_end(10.0, None, is_goal=False) == 10.0 + cfg.max_post_s
    # Crowd never decays -> capped.
    times = np.arange(0.0, 60.0, 0.1)
    rms = np.where(times >= 10.0, 0.5, 0.05)
    assert emotion_end(10.0, (times, rms), is_goal=True) == 10.0 + cfg.max_post_goal_s


def test_refine_intervals_extends_goal_clip_and_merges_overlaps() -> None:
    track = _attack_track(t_turn=20.0, t_goal=28.0)
    segments = [GameStateSegment(0.0, 60.0, STATE_IN_PLAY, reason="in play")]
    times = np.arange(0.0, 60.0, 0.1)
    rms = np.full_like(times, 0.05)
    surge = (times >= 28.0) & (times <= 36.0)
    rms[surge] = 0.5 - 0.055 * (times[surge] - 28.0)

    refined = refine_intervals(
        intervals=[(26.0, 32.0), (33.0, 38.0)],
        event_rows=[{"t": 28.0, "event_type": "goal", "side": "left"}],
        ball_track=track,
        segments=segments,
        envelope=(times, rms),
        duration_s=60.0,
    )

    # Goal start walked back toward the attack origin and the two intervals
    # merged after the ending grew.
    assert len(refined) == 1
    assert refined[0][0] <= 21.0
    assert refined[0][1] >= 33.0


def test_camera_deadband_holds_aim_for_jitter() -> None:
    geometry = _geometry()
    hz = 15.0
    rng = np.random.default_rng(5)
    # Ball jitters within a few pixels of a fixed spot (dribbling in place).
    rows = [(i / hz, 800.0 + rng.uniform(-6, 6), 500.0 + rng.uniform(-6, 6))
            for i in range(int(6 * hz))]
    track = build_ball_track(rows, FRAME)
    segments = [GameStateSegment(0.0, 6.0, STATE_IN_PLAY, reason="in play")]

    plan = plan_camera(
        ball_track=track, player_positions=None, geometry=geometry,
        segments=segments, start_seconds=0.0, end_seconds=6.0, fps=30.0,
        frame_size=FRAME, base_zoom=1.8, config=CameraPlannerConfig(),
    )

    late = [d for d in plan.decisions if d.t >= 2.0]
    xs = np.array([d.center_x for d in late])
    ys = np.array([d.center_y for d in late])
    # The camera settles and rests: total wander stays within a few pixels.
    assert xs.max() - xs.min() < 8.0, f"camera hunted: {xs.max() - xs.min():.1f}px"
    assert ys.max() - ys.min() < 8.0
    zooms = np.array([d.zoom for d in late])
    assert zooms.max() - zooms.min() < 0.06


# ---------------------------------------------------------------------------
# Reel building (requires moviepy)
# ---------------------------------------------------------------------------


def test_build_broadcast_reel_with_goal_replay(tmp_path) -> None:
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("moviepy")
    from backend.services.broadcast import build_broadcast_reel

    def _write_clip(path, seconds, color):
        w, h = 128, 96
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
        assert writer.isOpened()
        for i in range(int(seconds * 10)):
            frame = np.full((h, w, 3), color, dtype=np.uint8)
            cv2.circle(frame, (10 + i, h // 2), 4, (0, 255, 255), -1)
            writer.write(frame)
        writer.release()

    clip1 = tmp_path / "highlight_01.mp4"
    clip2 = tmp_path / "highlight_02.mp4"
    _write_clip(clip1, 6.0, (40, 90, 40))
    _write_clip(clip2, 6.0, (90, 40, 40))

    out = tmp_path / "highlights_reel.mp4"
    result = build_broadcast_reel(
        [
            {"path": str(clip1), "start_s": 100.0, "end_s": 106.0,
             "event_type": "goal", "occurred_at_s": 104.0, "confidence": 0.95},
            {"path": str(clip2), "start_s": 200.0, "end_s": 206.0,
             "event_type": None, "occurred_at_s": 203.0, "confidence": 0.6},
        ],
        str(out),
    )

    assert result == str(out)
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1.0, cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    # Cold open (~3s) + clip1 (6s) + slow-mo replay (~5s/0.4 = 12.5s) +
    # clip2 (6s) minus crossfade overlaps: must be well beyond plain concat.
    assert duration > 6.0 + 6.0 + 3.0


def test_build_broadcast_reel_empty_specs_returns_none(tmp_path) -> None:
    pytest.importorskip("moviepy")
    from backend.services.broadcast import build_broadcast_reel

    assert build_broadcast_reel([], str(tmp_path / "reel.mp4")) is None
