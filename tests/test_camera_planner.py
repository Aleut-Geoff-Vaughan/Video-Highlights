from __future__ import annotations

import json
import math

import numpy as np

from backend.services.camera_planner import (
    CameraPlannerConfig,
    plan_camera,
    slice_plan,
)
from backend.services.game_tracking import (
    GameStateSegment,
    STATE_BALL_LOST,
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


def _moving_ball(t0=0.0, t1=6.0, x0=300.0, x1=1500.0, y=540.0, hz=15.0):
    steps = int((t1 - t0) * hz)
    return build_ball_track(
        [(t0 + i / hz, x0 + (x1 - x0) * i / steps, y) for i in range(steps + 1)],
        FRAME,
    )


def test_plan_follows_moving_ball() -> None:
    geometry = _geometry()
    ball = _moving_ball()
    segments = [GameStateSegment(0.0, 6.0, STATE_IN_PLAY, reason="ball visible in field")]

    plan = plan_camera(
        ball_track=ball,
        player_positions=None,
        geometry=geometry,
        segments=segments,
        start_seconds=0.0,
        end_seconds=6.0,
        fps=10.0,
        frame_size=FRAME,
        base_zoom=1.6,
    )

    assert len(plan) == 60
    xs = [d.center_x for d in plan.decisions]
    assert xs[-1] > xs[0] + 400.0  # camera panned with the ball
    assert all(d.reason for d in plan.decisions)
    following = [d for d in plan.decisions if d.focus in {"ball", "ball_lead"}]
    assert len(following) >= 50
    assert plan.summary()["mean_confidence"] > 0.8


def test_plan_holds_at_goal_during_restart_and_does_not_leave() -> None:
    geometry = _geometry()
    # Ball visible only for the first second, then gone for the whole wait.
    ball = _moving_ball(0.0, 1.0, 1500.0, geometry.x_max - 10.0, geometry.right_goal.center[1])
    segments = [
        GameStateSegment(0.0, 1.2, STATE_IN_PLAY, reason="ball visible in field"),
        GameStateSegment(1.2, 15.0, "restart_right", side="right",
                         reason="ball out over the right goal line - waiting for goal kick/corner"),
    ]
    cfg = CameraPlannerConfig()

    plan = plan_camera(
        ball_track=ball,
        player_positions=None,
        geometry=geometry,
        segments=segments,
        start_seconds=0.0,
        end_seconds=15.0,
        fps=10.0,
        frame_size=FRAME,
        base_zoom=1.6,
        config=cfg,
    )

    goal_x, goal_y = geometry.right_goal.center
    aim_x = goal_x - cfg.goal_infield_offset_frac * geometry.width
    # Give the spring 2.5s to settle, then the camera must sit at the goal
    # for the entire remaining wait - it never drifts back to midfield.
    settled = [d for d in plan.decisions if 4.0 <= d.t <= 14.5]
    assert settled
    for decision in settled:
        # Camera center is clamped by the crop, so compare against the
        # clamped aim point rather than the raw goal center.
        crop_half_w = FRAME[0] / decision.zoom / 2.0
        expected_x = min(aim_x, FRAME[0] - crop_half_w)
        assert abs(decision.center_x - expected_x) < 60.0, (
            f"camera left the goal at t={decision.t:.1f}s: {decision.center_x:.0f} vs {expected_x:.0f}"
        )
        assert decision.focus == "goal_right"
        assert "goal" in decision.reason
        assert decision.zoom <= cfg.restart_zoom_cap + 0.05


def test_plan_zooms_out_and_follows_players_when_ball_lost() -> None:
    geometry = _geometry()
    ball = _moving_ball(0.0, 1.0, 900.0, 950.0, 500.0)
    # Player cluster sits far from the last ball spot.
    rng = np.random.default_rng(11)
    n = 800
    players = np.stack(
        [rng.uniform(0, 20, n), rng.normal(1400.0, 40.0, n), rng.normal(700.0, 30.0, n)],
        axis=1,
    )
    segments = [
        GameStateSegment(0.0, 1.2, STATE_IN_PLAY, reason="ball visible in field"),
        GameStateSegment(1.2, 12.0, STATE_BALL_LOST, reason="ball not visible"),
    ]
    base_zoom = 1.6

    plan = plan_camera(
        ball_track=ball,
        player_positions=players,
        geometry=geometry,
        segments=segments,
        start_seconds=0.0,
        end_seconds=12.0,
        fps=10.0,
        frame_size=FRAME,
        base_zoom=base_zoom,
    )

    late = [d for d in plan.decisions if d.t >= 8.0]
    assert late
    for decision in late:
        assert decision.zoom < base_zoom - 0.1  # zoomed out while searching
    assert any(d.focus == "action_centroid" for d in late)
    # Drifted toward the player cluster.
    assert late[-1].center_x > 1150.0


def test_plan_centers_always_within_legal_crop() -> None:
    geometry = _geometry()
    ball = _moving_ball(0.0, 4.0, -200.0, 2200.0, 100.0)  # deliberately out of frame
    segments = [GameStateSegment(0.0, 4.0, STATE_IN_PLAY, reason="test")]

    plan = plan_camera(
        ball_track=ball,
        player_positions=None,
        geometry=geometry,
        segments=segments,
        start_seconds=0.0,
        end_seconds=4.0,
        fps=12.0,
        frame_size=FRAME,
        base_zoom=2.0,
    )

    for decision in plan.decisions:
        half_w = FRAME[0] / decision.zoom / 2.0
        half_h = FRAME[1] / decision.zoom / 2.0
        assert half_w - 0.51 <= decision.center_x <= FRAME[0] - half_w + 0.51
        assert half_h - 0.51 <= decision.center_y <= FRAME[1] - half_h + 0.51
        assert decision.zoom >= 1.0


def test_slice_plan_reindexes_decisions() -> None:
    geometry = _geometry()
    ball = _moving_ball()
    segments = [GameStateSegment(0.0, 6.0, STATE_IN_PLAY, reason="test")]
    plan = plan_camera(
        ball_track=ball, player_positions=None, geometry=geometry, segments=segments,
        start_seconds=0.0, end_seconds=6.0, fps=10.0, frame_size=FRAME, base_zoom=1.6,
    )

    sub = slice_plan(plan, 2.0, 4.0)
    assert len(sub) == 20
    assert sub.decisions[0].index == 0
    assert math.isclose(sub.start_seconds, 2.0, abs_tol=1e-6)
    assert math.isclose(sub.decisions[0].t, 2.0, abs_tol=1e-6)
    # Original untouched.
    assert plan.decisions[20].index == 20


def test_plan_writes_jsonl_decisions(tmp_path) -> None:
    geometry = _geometry()
    ball = _moving_ball(0.0, 1.0)
    segments = [GameStateSegment(0.0, 1.0, STATE_IN_PLAY, reason="test")]
    plan = plan_camera(
        ball_track=ball, player_positions=None, geometry=geometry, segments=segments,
        start_seconds=0.0, end_seconds=1.0, fps=5.0, frame_size=FRAME, base_zoom=1.6,
    )

    path = tmp_path / "decisions.jsonl"
    plan.write_jsonl(str(path))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == len(plan)
    assert {"t", "center_x", "center_y", "zoom", "state", "focus", "reason", "confidence"} <= set(rows[0])
