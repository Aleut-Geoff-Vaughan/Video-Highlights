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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .follow_cam import _clamp_center
from .game_tracking import (
    BALL_SOURCE_DETECTED,
    GOAL_STATES,
    RESTART_STATES,
    STATE_BALL_LOST,
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
    action_blend: float = 0.25
    # Players within this fraction of the frame width around the ball count
    # as part of "the action".
    action_radius_frac: float = 0.22
    # After losing the ball, hold the camera this long before drifting.
    hold_last_s: float = 1.2
    # Critically damped spring stiffness (rad/s) for camera motion.
    spring_omega: float = 3.0
    # Max pan speed as crop-widths per second.
    max_pan_speed_crop_frac: float = 1.6
    # Zoom easing rate (per second).
    zoom_smooth_rate: float = 1.6
    # Zoom levels relative to the configured base zoom.
    lost_zoom_scale: float = 0.8
    restart_zoom_cap: float = 1.35
    goal_zoom_scale: float = 1.0
    min_zoom: float = 1.05
    # Aim slightly infield from the goal center during restarts so the crop
    # shows both the goal and the approach play.
    goal_infield_offset_frac: float = 0.05
    # Player-position time bin used for centroid lookups.
    player_bin_s: float = 0.5


@dataclass
class CameraDecision:
    """One per output frame: where the camera points and why."""

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

    def centers(self) -> List[Tuple[float, float]]:
        return [(d.center_x, d.center_y) for d in self.decisions]

    def zooms(self) -> List[float]:
        return [d.zoom for d in self.decisions]

    def write_jsonl(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as handle:
            for decision in self.decisions:
                handle.write(json.dumps(decision.to_dict()) + "\n")
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
        copied = CameraDecision(**{**decision.__dict__, "index": new_index})
        sliced.decisions.append(copied)
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

    # Camera physical state (position + velocity for the damped spring).
    cam_pos: Optional[Tuple[float, float]] = None
    cam_vel = (0.0, 0.0)
    zoom = base_zoom
    last_ball_seen_t: Optional[float] = None
    last_ball_xy: Optional[Tuple[float, float]] = None

    lost_zoom = max(cfg.min_zoom, base_zoom * cfg.lost_zoom_scale)
    restart_zoom = max(cfg.min_zoom, min(base_zoom, cfg.restart_zoom_cap))
    goal_zoom = max(cfg.min_zoom, base_zoom * cfg.goal_zoom_scale)
    infield_offset = cfg.goal_infield_offset_frac * geometry.width

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

        if state in GOAL_STATES and segment is not None:
            goal = geometry.goal_for_side(segment.side or "left")
            gx, gy = goal.center
            gx += infield_offset if goal.side == "left" else -infield_offset
            target = (gx, gy)
            target_zoom = goal_zoom
            focus = f"goal_{goal.side}"
            confidence = 0.9
            reason = state_reason or f"holding on {goal.side} goal after goal"
        elif state in RESTART_STATES and segment is not None:
            if state == STATE_RESTART_TOUCHLINE:
                anchor = last_ball_xy or ((geometry.x_min + geometry.x_max) / 2.0,
                                          (geometry.y_min + geometry.y_max) / 2.0)
                target = anchor
                focus = "exit_point"
                confidence = 0.6
                reason = state_reason or "ball out over touchline - holding at exit point"
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
            # Fast ball -> slightly wider shot so the play stays in frame.
            speed = math.hypot(vx, vy)
            speed_frac = min(1.0, speed / (1.2 * frame_w))
            target_zoom = max(cfg.min_zoom, base_zoom * (1.0 - 0.25 * speed_frac))
        else:
            # Ball not visible: hold briefly, then follow the player cluster.
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

        # --- Smooth zoom first, then motion, then clamp to legal crop. ---
        zoom += (target_zoom - zoom) * min(1.0, cfg.zoom_smooth_rate * dt)
        zoom = max(cfg.min_zoom, zoom)

        clamped_target = _clamp_center(target, (frame_w, frame_h), zoom)

        if cam_pos is None:
            cam_pos = clamped_target
            cam_vel = (0.0, 0.0)
        else:
            omega = cfg.spring_omega
            ax = (omega * omega) * (clamped_target[0] - cam_pos[0]) - 2.0 * omega * cam_vel[0]
            ay = (omega * omega) * (clamped_target[1] - cam_pos[1]) - 2.0 * omega * cam_vel[1]
            cam_vel = (cam_vel[0] + ax * dt, cam_vel[1] + ay * dt)
            crop_w = frame_w / zoom
            max_speed = cfg.max_pan_speed_crop_frac * crop_w
            vel_mag = math.hypot(*cam_vel)
            if vel_mag > max_speed > 0:
                scale = max_speed / vel_mag
                cam_vel = (cam_vel[0] * scale, cam_vel[1] * scale)
            cam_pos = (cam_pos[0] + cam_vel[0] * dt, cam_pos[1] + cam_vel[1] * dt)

        cam_pos = _clamp_center(cam_pos, (frame_w, frame_h), zoom)

        plan.decisions.append(
            CameraDecision(
                index=index,
                t=float(t),
                center_x=float(cam_pos[0]),
                center_y=float(cam_pos[1]),
                zoom=float(zoom),
                state=state,
                focus=focus,
                reason=reason,
                confidence=float(confidence),
                ball_x=float(ball[0]) if ball is not None else None,
                ball_y=float(ball[1]) if ball is not None else None,
                ball_source=ball[2] if ball is not None else None,
                target_x=float(clamped_target[0]),
                target_y=float(clamped_target[1]),
            )
        )

    LOGGER.info("camera plan built: %s", plan.summary())
    return plan
