"""Game-centric virtual camera planning.

Given a cleaned ball track, all player positions, the field geometry, and the
game-state timeline, this module plans one camera decision per output frame:
where the virtual camera should point ("the center of the game"), how far it
should zoom, and - crucially - *why*. Every decision carries a human-readable
reason plus the underlying evidence so the plan can be rendered as a debug
overlay, dumped as training data, and audited frame by frame.

Camera behavior by game state:

* ``in_play``      - follow the ball with a small velocity lead, blended with
                     the centroid of players near the ball so the crop shows
                     the play, not just the ball pixel.
* ``ball_lost``    - hold briefly, then ease toward the player cluster and
                     zoom out until the ball is found again.
* ``restart_*``    - the ball went out for a goal kick / corner / throw-in:
                     lock onto the relevant goal (or the exit point) and DO
                     NOT drift away until the ball is back in play.
* ``goal_*``       - a goal was flagged: hold on that goal.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .follow_cam import _clamp_center
from .game_tracking import (
    BALL_SOURCE_DETECTED,
    GOAL_STATES,
    RESTART_STATES,
    SET_PIECE_STATES,
    STATE_BALL_LOST,
    STATE_CORNER_SETUP,
    STATE_IN_PLAY,
    STATE_RESTART_TOUCHLINE,
    BallTrack,
    FieldGeometry,
    GameStateSegment,
)

LOGGER = logging.getLogger("videohighlights.camera_planner")


@dataclass
class CameraPlannerConfig:
    # How far ahead of the ball (in seconds of ball velocity) to aim.
    lead_time_s: float = 0.35
    # Blend between the ball position and the nearby-player centroid.
    # Kept LOW: the ball outranks the player cluster - it must never be
    # dragged out of frame by a crowd of players.
    action_blend: float = 0.12
    # Players within this fraction of the frame width around the ball count
    # as part of "the action".
    action_radius_frac: float = 0.22
    # After losing the ball, hold the camera this long before drifting.
    hold_last_s: float = 1.2
    # Offline zero-phase smoothing time constants: the plan is computed for
    # the whole video before rendering, so smoothing looks BOTH ways in time.
    # The camera glides and starts moving slightly before the play does -
    # the operator-like anticipation commercial systems are known for.
    smooth_time_constant_s: float = 0.6
    zoom_smooth_time_constant_s: float = 1.2
    # Motion limits for cinematic panning (crop-widths per second / s^2).
    max_pan_speed_crop_frac: float = 1.6
    max_pan_accel_crop_frac: float = 2.2
    # --- Goal-threat framing ---
    # When the ball is within this fraction of the field width from a goal
    # and attacking it, blend the aim toward the goal and pick a zoom that
    # keeps BOTH ball and goal in frame (corners arriving, crosses, shots).
    threat_zoom_dist_frac: float = 0.32
    threat_goal_blend_max: float = 0.35
    # Near-goal attacks may tighten up to this multiple of the base zoom as
    # the ball closes in on the goal.
    threat_tighten_scale: float = 1.3
    # Margin (fraction of frame) kept around the ball/goal pair when zooming
    # to fit both.
    both_in_frame_margin_frac: float = 0.12
    # Operator deadband: within one state, aim changes smaller than this
    # (fraction of frame width / absolute zoom) are ignored - a human
    # operator holds steady instead of chasing millimeters.
    deadband_frac: float = 0.02
    zoom_deadband: float = 0.05
    # Zoom levels relative to the configured base zoom.
    lost_zoom_scale: float = 0.8
    restart_zoom_cap: float = 1.35
    goal_zoom_scale: float = 1.0
    min_zoom: float = 1.05
    # Fast-ball zoom-out: at fast_ball_speed_frame_widths_per_s the zoom
    # widens by fast_ball_zoom_out_frac of the base zoom (linear below that).
    fast_ball_speed_frame_widths_per_s: float = 1.2
    fast_ball_zoom_out_frac: float = 0.25
    # Aim slightly infield from the goal center during restarts so the crop
    # shows both the goal and the approach play.
    goal_infield_offset_frac: float = 0.05
    # Player-position time bin used for centroid lookups.
    player_bin_s: float = 0.5


@dataclass(slots=True)
class CameraDecision:
    """One per output frame: where the camera points and why.

    slots=True matters here: an hour of 60fps video produces ~216k of these,
    and slotted instances roughly halve the per-decision memory.
    """

    index: int
    t: float
    center_x: float
    center_y: float
    zoom: float
    state: str
    focus: str  # ball | ball_lead | action_centroid | goal_left | goal_right | exit_point | hold | frame_center
    reason: str
    confidence: float
    ball_x: Optional[float] = None
    ball_y: Optional[float] = None
    ball_source: Optional[str] = None
    target_x: Optional[float] = None
    target_y: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "t": round(self.t, 3),
            "center_x": round(self.center_x, 2),
            "center_y": round(self.center_y, 2),
            "zoom": round(self.zoom, 3),
            "state": self.state,
            "focus": self.focus,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "ball_x": round(self.ball_x, 2) if self.ball_x is not None else None,
            "ball_y": round(self.ball_y, 2) if self.ball_y is not None else None,
            "ball_source": self.ball_source,
            "target_x": round(self.target_x, 2) if self.target_x is not None else None,
            "target_y": round(self.target_y, 2) if self.target_y is not None else None,
        }


@dataclass
class CameraPlan:
    start_seconds: float
    fps: float
    frame_size: Tuple[int, int]
    base_zoom: float
    decisions: List[CameraDecision] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.decisions)

    def write_jsonl(
        self,
        path: str,
        transform: Optional[Callable[[Dict[str, object]], Dict[str, object]]] = None,
    ) -> str:
        """Write one JSON object per decision; ``transform`` can enrich rows
        (e.g. adding source-video timestamps) so there is exactly one
        serialization of the training-data format."""
        with open(path, "w", encoding="utf-8") as handle:
            for decision in self.decisions:
                row = decision.to_dict()
                if transform is not None:
                    row = transform(row)
                handle.write(json.dumps(row) + "\n")
        return path

    def summary(self) -> Dict[str, object]:
        focus_counts: Dict[str, int] = {}
        for d in self.decisions:
            focus_counts[d.focus] = focus_counts.get(d.focus, 0) + 1
        return {
            "frames": len(self.decisions),
            "fps": round(self.fps, 3),
            "base_zoom": round(self.base_zoom, 3),
            "focus_frame_counts": focus_counts,
            "mean_confidence": round(
                float(np.mean([d.confidence for d in self.decisions])) if self.decisions else 0.0, 3
            ),
        }


def slice_plan(plan: CameraPlan, start_seconds: float, end_seconds: float) -> CameraPlan:
    """Cut a sub-plan covering [start_seconds, end_seconds) from a full plan.

    Decisions sit on the plan's fps grid, so slicing is an index range; the
    returned plan is re-indexed and starts at the sliced start time.
    """
    if end_seconds <= start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds")
    first = max(0, int(round((start_seconds - plan.start_seconds) * plan.fps)))
    last = min(len(plan.decisions), int(round((end_seconds - plan.start_seconds) * plan.fps)))
    sliced = CameraPlan(
        start_seconds=plan.start_seconds + first / plan.fps,
        fps=plan.fps,
        frame_size=plan.frame_size,
        base_zoom=plan.base_zoom,
    )
    for new_index, decision in enumerate(plan.decisions[first:last]):
        sliced.decisions.append(replace(decision, index=new_index))
    return sliced


class _PlayerLookup:
    """Time-binned player position lookup for fast centroid queries."""

    def __init__(self, player_positions: Optional[np.ndarray], bin_s: float) -> None:
        self.bin_s = max(0.1, float(bin_s))
        self.bins: Dict[int, np.ndarray] = {}
        if player_positions is None or len(player_positions) == 0:
            return
        arr = np.asarray(player_positions, dtype=np.float64)
        keys = (arr[:, 0] / self.bin_s).astype(np.int64)
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        arr = arr[order]
        boundaries = np.flatnonzero(np.diff(keys)) + 1
        for chunk_keys, chunk in zip(np.split(keys, boundaries), np.split(arr, boundaries)):
            if len(chunk_keys):
                self.bins[int(chunk_keys[0])] = chunk[:, 1:3]

    def positions_near(self, t: float) -> Optional[np.ndarray]:
        key = int(t / self.bin_s)
        chunks = [self.bins[k] for k in (key - 1, key, key + 1) if k in self.bins]
        if not chunks:
            return None
        return np.concatenate(chunks, axis=0)

    def centroid(self, t: float, around: Optional[Tuple[float, float]] = None,
                 radius: Optional[float] = None) -> Optional[Tuple[float, float]]:
        positions = self.positions_near(t)
        if positions is None or len(positions) == 0:
            return None
        if around is not None and radius is not None:
            deltas = positions - np.asarray(around, dtype=np.float64)
            mask = np.hypot(deltas[:, 0], deltas[:, 1]) <= radius
            if mask.sum() >= 2:
                positions = positions[mask]
        return float(np.mean(positions[:, 0])), float(np.mean(positions[:, 1]))


def _segment_lookup(segments: Sequence[GameStateSegment]):
    """Return a stateful function mapping monotonically increasing t -> segment."""
    ordered = sorted(segments, key=lambda seg: seg.start_s)
    idx = 0

    def lookup(t: float) -> Optional[GameStateSegment]:
        nonlocal idx
        while idx + 1 < len(ordered) and t >= ordered[idx].end_s:
            idx += 1
        if not ordered:
            return None
        return ordered[idx]

    return lookup


def plan_camera(
    *,
    ball_track: BallTrack,
    player_positions: Optional[np.ndarray],
    geometry: FieldGeometry,
    segments: Sequence[GameStateSegment],
    start_seconds: float,
    end_seconds: float,
    fps: float,
    frame_size: Tuple[int, int],
    base_zoom: float = 1.6,
    config: Optional[CameraPlannerConfig] = None,
) -> CameraPlan:
    """Plan one camera decision per frame for [start_seconds, end_seconds)."""
    if end_seconds <= start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds")
    if fps <= 0:
        raise ValueError("fps must be positive")

    cfg = config or CameraPlannerConfig()
    frame_w, frame_h = int(frame_size[0]), int(frame_size[1])
    base_zoom = max(cfg.min_zoom, float(base_zoom))
    frame_count = max(1, int(math.ceil((end_seconds - start_seconds) * fps)))
    dt = 1.0 / fps

    players = _PlayerLookup(player_positions, cfg.player_bin_s)
    seg_at = _segment_lookup(segments)
    action_radius = cfg.action_radius_frac * frame_w

    plan = CameraPlan(
        start_seconds=float(start_seconds),
        fps=float(fps),
        frame_size=(frame_w, frame_h),
        base_zoom=base_zoom,
    )

    # ------------------------------------------------------------------
    # Phase 1: one RAW aim point + zoom per frame, with reasons.
    # ------------------------------------------------------------------
    last_ball_seen_t: Optional[float] = None
    last_ball_xy: Optional[Tuple[float, float]] = None

    lost_zoom = max(cfg.min_zoom, base_zoom * cfg.lost_zoom_scale)
    restart_zoom = max(cfg.min_zoom, min(base_zoom, cfg.restart_zoom_cap))
    goal_zoom = max(cfg.min_zoom, base_zoom * cfg.goal_zoom_scale)
    infield_offset = cfg.goal_infield_offset_frac * geometry.width
    margin = cfg.both_in_frame_margin_frac

    def _zoom_to_frame_both(a: Tuple[float, float], b: Tuple[float, float],
                            max_zoom: float) -> float:
        """Widest-necessary zoom that keeps points a and b in the crop."""
        half_w = abs(a[0] - b[0]) / 2.0 + margin * frame_w
        half_h = abs(a[1] - b[1]) / 2.0 + margin * frame_h
        fit = min(frame_w / (2.0 * half_w), frame_h / (2.0 * half_h))
        return max(cfg.min_zoom, min(max_zoom, fit))

    raw_targets = np.empty((frame_count, 2), dtype=np.float64)
    raw_zooms = np.empty(frame_count, dtype=np.float64)
    metas: List[Tuple[str, str, str, float, Optional[Tuple[float, float, str]]]] = []
    # Hard framing constraints: points that MUST stay inside the crop for
    # this frame (the goal during set pieces/restarts/goal holds, ball+goal
    # during a goal threat). Enforced after smoothing so the offline filter
    # can never ease them out of frame.
    keep_points: List[List[Tuple[float, float]]] = []
    deadband_px = cfg.deadband_frac * frame_w
    prev_key: Optional[Tuple[str, str]] = None
    prev_target: Optional[Tuple[float, float]] = None
    prev_zoom: Optional[float] = None

    for index in range(frame_count):
        t = start_seconds + index * dt
        segment = seg_at(t)
        state = segment.state if segment is not None else STATE_IN_PLAY
        state_reason = segment.reason if segment is not None else ""

        ball = ball_track.position_at(t)
        if ball is not None:
            last_ball_seen_t = t
            last_ball_xy = (ball[0], ball[1])

        target: Tuple[float, float]
        target_zoom = base_zoom
        focus = "frame_center"
        reason = state_reason or "no signal"
        confidence = 0.2
        keeps: List[Tuple[float, float]] = []

        if state in GOAL_STATES and segment is not None:
            goal = geometry.goal_for_side(segment.side or "left")
            gx, gy = goal.center
            gx += infield_offset if goal.side == "left" else -infield_offset
            target = (gx, gy)
            target_zoom = goal_zoom
            focus = f"goal_{goal.side}"
            confidence = 0.9
            reason = state_reason or f"holding on {goal.side} goal after goal"
            keeps = [goal.center]
        elif state in SET_PIECE_STATES and segment is not None:
            goal = geometry.goal_for_side(segment.side or "left")
            anchor = last_ball_xy if last_ball_xy is not None else goal.center
            if ball is not None:
                anchor = (ball[0], ball[1])
            target = (
                anchor[0] * 0.45 + goal.center[0] * 0.55,
                anchor[1] * 0.45 + goal.center[1] * 0.55,
            )
            target_zoom = _zoom_to_frame_both(anchor, goal.center, base_zoom)
            focus = "set_piece"
            confidence = 0.85
            keeps = [goal.center, anchor]
            if state == STATE_CORNER_SETUP:
                reason = state_reason or (
                    f"corner kick setup - wide framing of the corner and the {goal.side} goal"
                )
            else:
                reason = state_reason or (
                    f"free kick setup - keeping the {goal.side} goal in view during the run-up"
                )
        elif state in RESTART_STATES and segment is not None:
            if state == STATE_RESTART_TOUCHLINE:
                anchor = last_ball_xy or ((geometry.x_min + geometry.x_max) / 2.0,
                                          (geometry.y_min + geometry.y_max) / 2.0)
                target = anchor
                focus = "exit_point"
                confidence = 0.6
                reason = state_reason or "ball out over touchline - holding at exit point"
                keeps = [anchor]
            else:
                goal = geometry.goal_for_side(segment.side or "left")
                gx, gy = goal.center
                gx += infield_offset if goal.side == "left" else -infield_offset
                target = (gx, gy)
                focus = f"goal_{goal.side}"
                confidence = 0.75
                reason = state_reason or (
                    f"ball out near {goal.side} goal - staying on the goal until play restarts"
                )
                keeps = [goal.center]
            target_zoom = restart_zoom
        elif ball is not None:
            bx, by, source = ball
            vx, vy = ball_track.velocity_at(t)
            lead_x = bx + vx * cfg.lead_time_s
            lead_y = by + vy * cfg.lead_time_s
            centroid = players.centroid(t, around=(bx, by), radius=action_radius)
            if centroid is not None and cfg.action_blend > 0.0:
                target = (
                    lead_x * (1.0 - cfg.action_blend) + centroid[0] * cfg.action_blend,
                    lead_y * (1.0 - cfg.action_blend) + centroid[1] * cfg.action_blend,
                )
                focus = "ball"
                reason = "following ball (lead + nearby-player blend)"
            else:
                target = (lead_x, lead_y)
                focus = "ball_lead" if (abs(vx) + abs(vy)) > 1.0 else "ball"
                reason = "following ball"
            confidence = 0.92 if source == BALL_SOURCE_DETECTED else 0.6
            if source != BALL_SOURCE_DETECTED:
                reason += " (interpolated across a short detection gap)"
            # TOP PRIORITY: the visible ball never leaves the crop, no
            # matter what the player-cluster blend or smoothing wants.
            keeps = [(bx, by)]
            # Fast ball -> slightly wider shot so the play stays in frame.
            speed = math.hypot(vx, vy)
            speed_frac = min(1.0, speed / (cfg.fast_ball_speed_frame_widths_per_s * frame_w))
            target_zoom = max(
                cfg.min_zoom, base_zoom * (1.0 - cfg.fast_ball_zoom_out_frac * speed_frac)
            )
            # Goal-threat framing: ball attacking a goal -> aim between ball
            # and goal, zoom to keep both in frame; the shot naturally
            # tightens as the ball closes in (corners arriving, crosses,
            # shots on target).
            threat_r = cfg.threat_zoom_dist_frac * geometry.width
            goal = geometry.left_goal if bx - geometry.x_min < geometry.x_max - bx else geometry.right_goal
            gx, gy = goal.center
            dist = math.hypot(bx - gx, by - gy)
            attacking = (vx < -10.0 if goal.side == "left" else vx > 10.0)
            if dist <= threat_r and (attacking or dist <= 0.55 * threat_r):
                closeness = 1.0 - (dist / threat_r)
                w = cfg.threat_goal_blend_max * closeness
                target = (target[0] * (1.0 - w) + gx * w, target[1] * (1.0 - w) + gy * w)
                # Fit ball AND goal; the shot tightens (beyond base zoom, up
                # to threat_tighten_scale) as the ball closes on the goal.
                target_zoom = _zoom_to_frame_both(
                    (bx, by), (gx, gy), base_zoom * cfg.threat_tighten_scale
                )
                focus = "ball_goal_threat"
                reason = f"attacking the {goal.side} goal - framing ball and goal together"
                confidence = max(confidence, 0.9)
                keeps = [(bx, by), (gx, gy)]
        else:
            recently_seen = (
                last_ball_seen_t is not None and (t - last_ball_seen_t) <= cfg.hold_last_s
            )
            if recently_seen and last_ball_xy is not None:
                target = last_ball_xy
                focus = "hold"
                confidence = 0.5
                reason = "ball just went out of sight - holding last known spot"
                target_zoom = base_zoom
            else:
                centroid = players.centroid(t, around=last_ball_xy, radius=action_radius * 2.0)
                if centroid is None:
                    centroid = players.centroid(t)
                if centroid is not None:
                    target = centroid
                    focus = "action_centroid"
                    confidence = 0.35
                    reason = (state_reason or "ball not visible") + " - following player cluster"
                else:
                    target = (frame_w / 2.0, frame_h / 2.0)
                    focus = "frame_center"
                    confidence = 0.1
                    reason = "no ball and no players visible - centering frame"
                target_zoom = lost_zoom

        # Deadband: hold the previous aim for sub-threshold changes within
        # the same state/focus so the camera rests instead of micro-hunting.
        key = (state, focus)
        if prev_key == key and prev_target is not None:
            if math.hypot(target[0] - prev_target[0], target[1] - prev_target[1]) < deadband_px:
                target = prev_target
            if prev_zoom is not None and abs(target_zoom - prev_zoom) < cfg.zoom_deadband:
                target_zoom = prev_zoom
        prev_key = key
        prev_target = target
        prev_zoom = target_zoom

        clamped = _clamp_center(target, (frame_w, frame_h), max(cfg.min_zoom, target_zoom))
        raw_targets[index, 0] = clamped[0]
        raw_targets[index, 1] = clamped[1]
        raw_zooms[index] = max(cfg.min_zoom, target_zoom)
        metas.append((state, focus, reason, confidence, ball))
        keep_points.append(keeps)

    # ------------------------------------------------------------------
    # Phase 2: offline zero-phase smoothing + physical motion limits.
    # ------------------------------------------------------------------
    smoothed_x = _zero_phase_smooth(raw_targets[:, 0], dt, cfg.smooth_time_constant_s)
    smoothed_y = _zero_phase_smooth(raw_targets[:, 1], dt, cfg.smooth_time_constant_s)
    smoothed_zoom = _zero_phase_smooth(raw_zooms, dt, cfg.zoom_smooth_time_constant_s)

    cam_x, cam_y = float(smoothed_x[0]), float(smoothed_y[0])
    vel_x = vel_y = 0.0

    for index in range(frame_count):
        zoom = max(cfg.min_zoom, float(smoothed_zoom[index]))
        crop_w = frame_w / zoom
        max_speed = cfg.max_pan_speed_crop_frac * crop_w
        max_accel = cfg.max_pan_accel_crop_frac * crop_w

        desired_vx = (float(smoothed_x[index]) - cam_x) / dt
        desired_vy = (float(smoothed_y[index]) - cam_y) / dt
        dvx, dvy = desired_vx - vel_x, desired_vy - vel_y
        dv_mag = math.hypot(dvx, dvy)
        max_dv = max_accel * dt
        if dv_mag > max_dv > 0:
            scale = max_dv / dv_mag
            dvx *= scale
            dvy *= scale
        vel_x += dvx
        vel_y += dvy
        v_mag = math.hypot(vel_x, vel_y)
        if v_mag > max_speed > 0:
            scale = max_speed / v_mag
            vel_x *= scale
            vel_y *= scale
        cam_x += vel_x * dt
        cam_y += vel_y * dt
        cam_x, cam_y = _clamp_center((cam_x, cam_y), (frame_w, frame_h), zoom)

        # Hard framing constraints beat smoothing: shift the crop the
        # minimum needed to keep the anchors (goal, restart ball) in frame.
        for keep_x, keep_y in keep_points[index]:
            pad = 6.0
            keep_half_w = frame_w / (2.0 * zoom) - pad
            keep_half_h = frame_h / (2.0 * zoom) - pad
            if keep_half_w > 0:
                cam_x = min(max(cam_x, keep_x - keep_half_w), keep_x + keep_half_w)
            if keep_half_h > 0:
                cam_y = min(max(cam_y, keep_y - keep_half_h), keep_y + keep_half_h)
        if keep_points[index]:
            cam_x, cam_y = _clamp_center((cam_x, cam_y), (frame_w, frame_h), zoom)

        state, focus, reason, confidence, ball = metas[index]
        plan.decisions.append(
            CameraDecision(
                index=index,
                t=float(start_seconds + index * dt),
                center_x=float(cam_x),
                center_y=float(cam_y),
                zoom=float(zoom),
                state=state,
                focus=focus,
                reason=reason,
                confidence=float(confidence),
                ball_x=float(ball[0]) if ball is not None else None,
                ball_y=float(ball[1]) if ball is not None else None,
                ball_source=ball[2] if ball is not None else None,
                target_x=float(raw_targets[index, 0]),
                target_y=float(raw_targets[index, 1]),
            )
        )

    LOGGER.info("camera plan built: %s", plan.summary())
    return plan


def _zero_phase_smooth(values: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """Forward+backward exponential smoothing (zero phase lag).

    Because the whole plan exists before rendering, the backward pass lets
    the camera begin easing toward upcoming action BEFORE it happens -
    smooth, anticipatory motion instead of reactive chasing.
    """
    if tau <= 0 or len(values) < 3:
        return values.astype(np.float64, copy=True)
    alpha = dt / (tau + dt)
    fwd = np.empty(len(values), dtype=np.float64)
    acc = float(values[0])
    for i in range(len(values)):
        acc += alpha * (float(values[i]) - acc)
        fwd[i] = acc
    out = np.empty(len(values), dtype=np.float64)
    acc = fwd[-1]
    for i in range(len(values) - 1, -1, -1):
        acc += alpha * (fwd[i] - acc)
        out[i] = acc
    return out
