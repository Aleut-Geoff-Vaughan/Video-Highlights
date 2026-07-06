"""Ball tracking, field geometry, and game-state analysis.

This module turns raw per-frame ball detections into a cleaned ball track,
estimates where the field and the two goals are inside the frame, and then
classifies the match timeline into game states:

* ``in_play``            - the ball is visible and inside the field of play.
* ``ball_lost``          - the ball is not visible and we have no strong reason
                           to believe it left the field (occlusion, missed
                           detections).
* ``restart_left/right`` - the ball went out over a goal line (or vanished next
                           to one). The game is waiting for a goal kick or a
                           corner, so the camera must stay at that goal.
* ``restart_touchline``  - the ball went out over a touchline (throw-in wait).
* ``goal_left/right``    - the ball entered the goal. These are also emitted
                           as explicit goal events so they can be flagged as
                           bookmarks.

Everything is heuristic but fully explainable: each segment and each goal
event carries a human-readable ``reason`` plus the raw evidence values, so the
output can be reviewed, debugged, and used as training data.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("videohighlights.game_tracking")

BALL_SOURCE_DETECTED = "detected"
BALL_SOURCE_INTERPOLATED = "interpolated"

STATE_IN_PLAY = "in_play"
STATE_BALL_LOST = "ball_lost"
STATE_RESTART_LEFT = "restart_left"
STATE_RESTART_RIGHT = "restart_right"
STATE_RESTART_TOUCHLINE = "restart_touchline"
STATE_GOAL_LEFT = "goal_left"
STATE_GOAL_RIGHT = "goal_right"
STATE_CORNER_SETUP = "corner_kick_setup"
STATE_FREE_KICK_SETUP = "free_kick_setup"

RESTART_STATES = {STATE_RESTART_LEFT, STATE_RESTART_RIGHT, STATE_RESTART_TOUCHLINE}
GOAL_STATES = {STATE_GOAL_LEFT, STATE_GOAL_RIGHT}
SET_PIECE_STATES = {STATE_CORNER_SETUP, STATE_FREE_KICK_SETUP}


# ---------------------------------------------------------------------------
# Ball track building
# ---------------------------------------------------------------------------


@dataclass
class BallTrackConfig:
    """Tuning knobs for turning raw detections into a clean ball track."""

    # Hard physical gate: reject detections implying speeds above this many
    # frame-widths per second relative to the current filtered state.
    max_speed_frame_widths_per_s: float = 2.2
    # Base association gate in pixels (grows with elapsed time).
    base_gate_px: float = 80.0
    # Cap on the association gate as a fraction of the frame width. Without
    # a cap, the gate outgrows the frame after ~0.4s of detector dropout and
    # a single false positive (scoreboard, spare ball) teleports the track;
    # gaps longer than this are handled by the re-acquisition path instead.
    max_gate_frac: float = 0.35
    # A fresh/re-acquired track is confirmed once this many mutually
    # consistent detections land inside a short window.
    confirm_detections: int = 2
    confirm_window_s: float = 0.7
    # After this long without an accepted detection, a consistent cluster of
    # detections anywhere in the frame restarts the track (ball re-appeared).
    reacquire_after_s: float = 0.9
    # Gaps up to this long are bridged by interpolation; longer gaps mean the
    # ball is genuinely "not visible".
    max_interpolation_gap_s: float = 1.25
    # Alpha-beta filter gains.
    smoothing_alpha: float = 0.5
    smoothing_beta: float = 0.3
    # A query time counts as "detected" (vs interpolated) when a real sample
    # is within this distance in time.
    detected_time_tolerance_s: float = 0.14


@dataclass
class BallTrack:
    """Cleaned ball trajectory with interpolation-aware queries."""

    times: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    frame_size: Tuple[int, int]
    config: BallTrackConfig
    stats: Dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.times.shape[0])

    def position_at(self, t: float) -> Optional[Tuple[float, float, str]]:
        """Return (x, y, source) at time ``t`` or None when not visible.

        ``source`` is ``detected`` when a real detection is nearby in time and
        ``interpolated`` when the value is bridged across a short gap.
        """
        if len(self) == 0:
            return None
        times = self.times
        idx = int(np.searchsorted(times, t))
        tol = self.config.detected_time_tolerance_s
        if idx <= 0:
            gap = float(times[0] - t)
            if gap > tol:
                return None
            return float(self.xs[0]), float(self.ys[0]), BALL_SOURCE_DETECTED
        if idx >= len(times):
            gap = float(t - times[-1])
            if gap > tol:
                return None
            return float(self.xs[-1]), float(self.ys[-1]), BALL_SOURCE_DETECTED

        left_t = float(times[idx - 1])
        right_t = float(times[idx])
        nearest_gap = min(t - left_t, right_t - t)
        span = right_t - left_t
        if span > self.config.max_interpolation_gap_s and nearest_gap > tol:
            return None
        if span <= 1e-9:
            return float(self.xs[idx - 1]), float(self.ys[idx - 1]), BALL_SOURCE_DETECTED
        alpha = (t - left_t) / span
        x = float(self.xs[idx - 1] + (self.xs[idx] - self.xs[idx - 1]) * alpha)
        y = float(self.ys[idx - 1] + (self.ys[idx] - self.ys[idx - 1]) * alpha)
        source = BALL_SOURCE_DETECTED if nearest_gap <= tol else BALL_SOURCE_INTERPOLATED
        return x, y, source

    def velocity_at(self, t: float, window_s: float = 0.4) -> Tuple[float, float]:
        """Finite-difference velocity (px/s) from samples within ``t`` +/- ``window_s``.

        Uses the actual samples inside the window (not interpolated queries),
        so it stays accurate right up to the edge of a visible segment - e.g.
        the instant a shot disappears into the goal.
        """
        if len(self) < 2:
            return 0.0, 0.0
        # searchsorted keeps this O(log n); it is called once per planned
        # frame, so a linear scan here would make the planner O(n^2).
        first = int(np.searchsorted(self.times, t - window_s, side="left"))
        last = int(np.searchsorted(self.times, t + window_s, side="right")) - 1
        if last - first < 1:
            return 0.0, 0.0
        dt = float(self.times[last] - self.times[first])
        if dt <= 1e-6:
            return 0.0, 0.0
        return (
            float(self.xs[last] - self.xs[first]) / dt,
            float(self.ys[last] - self.ys[first]) / dt,
        )

    def visibility_gaps(self, start_s: float, end_s: float) -> List[Tuple[float, float]]:
        """Time ranges within [start_s, end_s] where the ball is not visible."""
        gaps: List[Tuple[float, float]] = []
        max_gap = self.config.max_interpolation_gap_s
        if len(self) == 0:
            return [(start_s, end_s)] if end_s > start_s else []
        if float(self.times[0]) - start_s > max_gap:
            gaps.append((start_s, float(self.times[0])))
        diffs = np.diff(self.times)
        for i in np.where(diffs > max_gap)[0]:
            gap_start = float(self.times[i])
            gap_end = float(self.times[i + 1])
            if gap_end > start_s and gap_start < end_s:
                gaps.append((max(gap_start, start_s), min(gap_end, end_s)))
        if end_s - float(self.times[-1]) > max_gap:
            gaps.append((float(self.times[-1]), end_s))
        return gaps

    def coverage_fraction(self, start_s: float, end_s: float, step_s: float = 0.25) -> float:
        if end_s <= start_s:
            return 0.0
        steps = max(1, int(math.ceil((end_s - start_s) / step_s)))
        visible = 0
        for i in range(steps):
            if self.position_at(start_s + i * step_s) is not None:
                visible += 1
        return visible / steps

    def to_rows(self) -> List[Dict[str, float]]:
        return [
            {"t": round(float(t), 3), "x": round(float(x), 2), "y": round(float(y), 2)}
            for t, x, y in zip(self.times, self.xs, self.ys)
        ]


def _extract_detection(item: object) -> Optional[Tuple[float, float, float]]:
    """Accept TrackPoint-like objects or (t, x, y[, ...]) tuples."""
    t = getattr(item, "t", None)
    xy = getattr(item, "xy", None)
    if t is not None and xy is not None:
        try:
            return float(t), float(xy[0]), float(xy[1])
        except Exception:
            return None
    try:
        seq = tuple(item)  # type: ignore[arg-type]
        return float(seq[0]), float(seq[1]), float(seq[2])
    except Exception:
        return None


def build_ball_track(
    raw_detections: Iterable[object],
    frame_size: Tuple[int, int],
    config: Optional[BallTrackConfig] = None,
) -> BallTrack:
    """Filter raw ball detections into a clean, physically plausible track.

    The filter keeps a constant-velocity state, associates the nearest
    detection per frame inside a speed-based gate, rejects teleporting
    outliers (scoreboards, bald heads, spare balls), and re-acquires the ball
    after long gaps once a consistent cluster of detections appears.
    """
    cfg = config or BallTrackConfig()
    frame_w = max(1, int(frame_size[0]))
    speed_limit = cfg.max_speed_frame_widths_per_s * frame_w

    detections: List[Tuple[float, float, float]] = []
    for item in raw_detections or []:
        parsed = _extract_detection(item)
        if parsed is not None:
            detections.append(parsed)
    detections.sort(key=lambda row: row[0])

    accepted: List[Tuple[float, float, float]] = []
    rejected = 0

    # Filter state.
    pos: Optional[Tuple[float, float]] = None
    vel = (0.0, 0.0)
    last_t: Optional[float] = None
    pending: List[Tuple[float, float, float]] = []

    def _pending_consistent() -> Optional[List[Tuple[float, float, float]]]:
        """Return a confirmable cluster from pending detections, if any."""
        if len(pending) < cfg.confirm_detections:
            return None
        window = [p for p in pending if pending[-1][0] - p[0] <= cfg.confirm_window_s]
        if len(window) < cfg.confirm_detections:
            return None
        for a, b in zip(window[:-1], window[1:]):
            dt = max(1e-3, b[0] - a[0])
            dist = math.hypot(b[1] - a[1], b[2] - a[2])
            if dist / dt > speed_limit:
                return None
        return window

    # Group detections that share (nearly) the same timestamp.
    groups: List[List[Tuple[float, float, float]]] = []
    for det in detections:
        if groups and abs(det[0] - groups[-1][0][0]) <= 1e-4:
            groups[-1].append(det)
        else:
            groups.append([det])

    for group in groups:
        t = group[0][0]
        if pos is None:
            pending.extend(group)
            pending = [p for p in pending if t - p[0] <= max(cfg.confirm_window_s, 1.0)]
            window = _pending_consistent()
            if window is not None:
                for row in window:
                    accepted.append(row)
                tail, prev = window[-1], window[-2] if len(window) >= 2 else window[-1]
                dt = max(1e-3, tail[0] - prev[0])
                pos = (tail[1], tail[2])
                vel = ((tail[1] - prev[1]) / dt, (tail[2] - prev[2]) / dt) if len(window) >= 2 else (0.0, 0.0)
                last_t = tail[0]
                pending = []
            continue

        assert last_t is not None
        dt = max(1e-3, t - last_t)
        # Prediction with capped extrapolation so long gaps don't fling the
        # predicted point off-screen.
        pred_dt = min(dt, 0.5)
        pred = (pos[0] + vel[0] * pred_dt, pos[1] + vel[1] * pred_dt)
        gate = min(cfg.base_gate_px + speed_limit * dt, cfg.max_gate_frac * frame_w)

        best = min(group, key=lambda row: math.hypot(row[1] - pred[0], row[2] - pred[1]))
        dist = math.hypot(best[1] - pred[0], best[2] - pred[1])

        if dist <= gate:
            residual = (best[1] - pred[0], best[2] - pred[1])
            new_x = pred[0] + cfg.smoothing_alpha * residual[0]
            new_y = pred[1] + cfg.smoothing_alpha * residual[1]
            vel = (
                vel[0] + cfg.smoothing_beta * residual[0] / dt,
                vel[1] + cfg.smoothing_beta * residual[1] / dt,
            )
            speed = math.hypot(*vel)
            if speed > speed_limit:
                scale = speed_limit / speed
                vel = (vel[0] * scale, vel[1] * scale)
            pos = (new_x, new_y)
            last_t = t
            accepted.append((t, new_x, new_y))
            pending = []
        else:
            rejected += len(group)
            pending.extend(group)
            pending = [p for p in pending if t - p[0] <= max(cfg.confirm_window_s, 1.0)]
            if t - last_t > cfg.reacquire_after_s:
                window = _pending_consistent()
                if window is not None:
                    LOGGER.debug(
                        "ball re-acquired at t=%.2fs (%.0f, %.0f) after %.2fs silence",
                        window[-1][0], window[-1][1], window[-1][2], t - last_t,
                    )
                    for row in window:
                        accepted.append(row)
                    tail, prev = window[-1], window[-2] if len(window) >= 2 else window[-1]
                    dtw = max(1e-3, tail[0] - prev[0])
                    pos = (tail[1], tail[2])
                    vel = ((tail[1] - prev[1]) / dtw, (tail[2] - prev[2]) / dtw) if len(window) >= 2 else (0.0, 0.0)
                    last_t = tail[0]
                    pending = []

    accepted.sort(key=lambda row: row[0])
    # Deduplicate identical timestamps (keep the filtered value emitted last).
    deduped: List[Tuple[float, float, float]] = []
    for row in accepted:
        if deduped and abs(row[0] - deduped[-1][0]) <= 1e-6:
            deduped[-1] = row
        else:
            deduped.append(row)

    arr = np.asarray(deduped, dtype=np.float64) if deduped else np.empty((0, 3), dtype=np.float64)
    track = BallTrack(
        times=arr[:, 0] if arr.size else np.empty(0),
        xs=arr[:, 1] if arr.size else np.empty(0),
        ys=arr[:, 2] if arr.size else np.empty(0),
        frame_size=(int(frame_size[0]), int(frame_size[1])),
        config=cfg,
        stats={
            "raw_detections": len(detections),
            "accepted": len(deduped),
            "rejected_outliers": rejected,
        },
    )
    LOGGER.info(
        "ball track built: %d raw detections -> %d accepted, %d rejected outliers",
        len(detections), len(deduped), rejected,
    )
    return track


# ---------------------------------------------------------------------------
# Field / goal geometry
# ---------------------------------------------------------------------------


@dataclass
class GoalBox:
    side: str  # "left" | "right"
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (
            self.x1 - margin <= x <= self.x2 + margin
            and self.y1 - margin <= y <= self.y2 + margin
        )

    def to_dict(self) -> Dict[str, float]:
        return {"side": self.side, "x1": round(self.x1, 1), "y1": round(self.y1, 1),
                "x2": round(self.x2, 1), "y2": round(self.y2, 1)}


@dataclass
class FieldGeometry:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    left_goal: GoalBox
    right_goal: GoalBox
    frame_size: Tuple[int, int]
    source: str = "estimated"

    @property
    def width(self) -> float:
        return max(1.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(1.0, self.y_max - self.y_min)

    def goal_for_side(self, side: str) -> GoalBox:
        return self.left_goal if side == "left" else self.right_goal

    def side_if_near_goal(self, x: float, y: float, near_frac: float) -> Optional[str]:
        """Which goal (if any) the point is close to, in field-width units."""
        near_px = near_frac * self.width
        for goal in (self.left_goal, self.right_goal):
            gx, gy = goal.center
            goal_half_h = max(abs(goal.y2 - goal.y1) / 2.0, self.height * 0.25)
            if abs(x - gx) <= near_px and abs(y - gy) <= goal_half_h + near_px:
                return goal.side
        return None

    def to_dict(self) -> Dict[str, object]:
        return {
            "x_min": round(self.x_min, 1),
            "x_max": round(self.x_max, 1),
            "y_min": round(self.y_min, 1),
            "y_max": round(self.y_max, 1),
            "left_goal": self.left_goal.to_dict(),
            "right_goal": self.right_goal.to_dict(),
            "frame_width": int(self.frame_size[0]),
            "frame_height": int(self.frame_size[1]),
            "source": self.source,
        }


def _goal_box_from_override(raw: Optional[Dict[str, object]], side: str,
                            frame_size: Tuple[int, int]) -> Optional[GoalBox]:
    if not isinstance(raw, dict):
        return None
    try:
        x1, y1, x2, y2 = (float(raw["x1"]), float(raw["y1"]), float(raw["x2"]), float(raw["y2"]))
    except Exception:
        return None
    w, h = frame_size
    # Values <= 1.0 are treated as normalized coordinates.
    if max(x1, y1, x2, y2) <= 1.0:
        x1, x2 = x1 * w, x2 * w
        y1, y2 = y1 * h, y2 * h
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return GoalBox(side=side, x1=min(x1, x2), y1=min(y1, y2), x2=max(x1, x2), y2=max(y1, y2))


def estimate_field_geometry(
    player_positions: Optional[np.ndarray],
    frame_size: Tuple[int, int],
    goal_box_left: Optional[Dict[str, object]] = None,
    goal_box_right: Optional[Dict[str, object]] = None,
) -> FieldGeometry:
    """Estimate the playable area and goal mouths from player positions.

    Player positions accumulated over a match trace out the field: robust
    percentiles of x/y give the field bounds, and the vertical position of
    players close to each end line (mostly the goalkeepers) centers the goal
    mouth. Manual goal boxes (pixel or normalized) override the estimate.
    """
    w, h = int(frame_size[0]), int(frame_size[1])
    positions = None
    if player_positions is not None and len(player_positions) >= 50:
        positions = np.asarray(player_positions, dtype=np.float64)

    if positions is not None:
        xs = positions[:, 1]
        ys = positions[:, 2]
        x_min, x_max = float(np.percentile(xs, 0.5)), float(np.percentile(xs, 99.5))
        y_min, y_max = float(np.percentile(ys, 1.5)), float(np.percentile(ys, 98.5))
        source = "estimated"
    else:
        x_min, x_max = w * 0.02, w * 0.98
        y_min, y_max = h * 0.10, h * 0.90
        source = "frame_default"

    field_w = max(1.0, x_max - x_min)
    field_h = max(1.0, y_max - y_min)

    def _goal_y_center(near_x: float) -> float:
        if positions is None:
            return (y_min + y_max) / 2.0
        band = positions[np.abs(positions[:, 1] - near_x) <= field_w * 0.08]
        if len(band) < 20:
            return (y_min + y_max) / 2.0
        return float(np.median(band[:, 2]))

    goal_h = max(40.0, field_h * 0.30)
    goal_depth = max(24.0, field_w * 0.05)

    def _build_goal(side: str) -> GoalBox:
        if side == "left":
            y_c = _goal_y_center(x_min)
            return GoalBox(side="left", x1=max(0.0, x_min - goal_depth), y1=y_c - goal_h / 2.0,
                           x2=x_min, y2=y_c + goal_h / 2.0)
        y_c = _goal_y_center(x_max)
        return GoalBox(side="right", x1=x_max, y1=y_c - goal_h / 2.0,
                       x2=min(float(w), x_max + goal_depth), y2=y_c + goal_h / 2.0)

    left = _goal_box_from_override(goal_box_left, "left", (w, h)) or _build_goal("left")
    right = _goal_box_from_override(goal_box_right, "right", (w, h)) or _build_goal("right")
    if goal_box_left or goal_box_right:
        source = "manual" if (goal_box_left and goal_box_right) else f"{source}+manual"

    geometry = FieldGeometry(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        left_goal=left, right_goal=right, frame_size=(w, h), source=source,
    )
    LOGGER.info(
        "field geometry (%s): x=[%.0f, %.0f] y=[%.0f, %.0f], left goal %s, right goal %s",
        source, x_min, x_max, y_min, y_max, left.to_dict(), right.to_dict(),
    )
    return geometry


# ---------------------------------------------------------------------------
# Goal events
# ---------------------------------------------------------------------------


@dataclass
class GoalEvent:
    t: float
    side: str
    confidence: float
    reason: str
    evidence: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "t": round(self.t, 3),
            "side": self.side,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "evidence": self.evidence,
        }


@dataclass
class GameStateConfig:
    step_s: float = 0.2
    # Distance from a goal line (fraction of field width) that counts as
    # "near the goal" when the ball disappears.
    near_goal_frac: float = 0.14
    # How far beyond the field bounds a visible ball counts as out of play.
    out_margin_frac: float = 0.01
    # The ball must be invisible at least this long before we call it lost.
    lost_grace_s: float = 0.7
    # The ball must be back inside the field this long before a restart hold
    # is released (prevents flicker off single false detections).
    return_confirm_s: float = 0.6
    # Safety valve: never hold a restart longer than this.
    max_restart_hold_s: float = 50.0
    # Goal detection.
    goal_disappear_confirm_s: float = 2.0
    goal_lookback_s: float = 0.7
    goal_hold_s: float = 8.0
    kickoff_center_frac: float = 0.16
    kickoff_search_s: float = 45.0
    min_shot_speed_frame_widths_per_s: float = 0.25
    # Minimum x-velocity (px/s) that counts as the ball entering a goal.
    goal_entry_speed_px_s: float = 20.0
    # Goal candidates on the same side within this window merge into one.
    goal_merge_window_s: float = 6.0
    # How far ahead a vanishing ball's path is extrapolated to the goal line.
    goal_extrapolation_s: float = 0.8
    # Consecutive in-goal sightings further apart than this start a new run.
    goal_run_gap_s: float = 1.5
    # Margin around the estimated goal mouth (posts): max of these two.
    goal_mouth_margin_px: float = 12.0
    goal_mouth_margin_frac: float = 0.15
    # Goal events below this confidence are dropped (e.g. a ball that merely
    # vanishes near a goal as the recording ends).
    min_goal_confidence: float = 0.7
    # --- Set pieces (corners, free kicks, goal kicks, kickoffs) ---
    # The ball must sit still (within the radius) at least this long.
    set_piece_min_stationary_s: float = 1.2
    set_piece_stationary_radius_frac: float = 0.012  # of frame width
    # ...and then accelerate away at least this fast to count as the kick.
    set_piece_kick_speed_frame_widths_per_s: float = 0.15
    # Location classification (fractions of field size).
    corner_radius_frac: float = 0.06
    goal_kick_zone_depth_frac: float = 0.12
    goal_kick_zone_half_height_frac: float = 0.25
    penalty_depth_range_frac: Tuple[float, float] = (0.06, 0.18)
    penalty_half_height_frac: float = 0.12
    # A free kick within this distance of a goal line threatens that goal,
    # so the camera must keep the goal in view during the run-up.
    free_kick_threat_frac: float = 0.38


@dataclass
class GameStateSegment:
    start_s: float
    end_s: float
    state: str
    side: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "state": self.state,
            "side": self.side,
            "reason": self.reason,
        }


def detect_goal_events(
    ball_track: BallTrack,
    geometry: FieldGeometry,
    start_s: float,
    end_s: float,
    config: Optional[GameStateConfig] = None,
) -> List[GoalEvent]:
    """Flag moments where the ball went into a goal.

    Three independent signals produce goal candidates:

    1. The ball is *observed* inside a goal box (behind the goal line between
       the posts) after entering it from the field - a ball that simply
       appears there moving infield (a goal kick being taken) is rejected.
    2. The ball is observed *crossing* the goal line between the posts from
       the field side - this also catches fast shots that skip over the
       shallow goal box between frames.
    3. The ball disappears while moving toward a goal mouth, and its
       extrapolated path crosses the goal line between the posts.

    For every signal, confidence rises if the ball stays gone afterwards and
    if it re-appears near the center circle (the kickoff after a goal).
    """
    cfg = config or GameStateConfig()
    events: List[GoalEvent] = []
    if len(ball_track) == 0:
        return events

    frame_w = ball_track.frame_size[0]
    min_speed = cfg.min_shot_speed_frame_widths_per_s * frame_w
    center_x = (geometry.x_min + geometry.x_max) / 2.0
    center_y = (geometry.y_min + geometry.y_max) / 2.0
    kickoff_radius = cfg.kickoff_center_frac * geometry.width

    def _goal_line_x(goal: GoalBox) -> float:
        return goal.x2 if goal.side == "left" else goal.x1

    def _mouth_y_range(goal: GoalBox) -> Tuple[float, float]:
        # The goal-mouth y estimate can be a bit off; allow a forgiving
        # margin around the posts.
        margin = max(cfg.goal_mouth_margin_px, abs(goal.y2 - goal.y1) * cfg.goal_mouth_margin_frac)
        return goal.y1 - margin, goal.y2 + margin

    # Visibility gaps are shared by signals 2 and 3; compute once.
    all_gaps = ball_track.visibility_gaps(start_s, end_s)
    track_end_t = float(ball_track.times[-1]) if len(ball_track) else start_s

    def _reappears_near_center(after_t: float) -> Optional[float]:
        first = int(np.searchsorted(ball_track.times, after_t, side="right"))
        last = int(np.searchsorted(ball_track.times, after_t + cfg.kickoff_search_s, side="right"))
        if last <= first:
            return None
        dists = np.hypot(
            ball_track.xs[first:last] - center_x, ball_track.ys[first:last] - center_y
        )
        hits = np.flatnonzero(dists <= kickoff_radius)
        if len(hits) == 0:
            return None
        return float(ball_track.times[first + int(hits[0])])

    def _add_event(t: float, side: str, confidence: float, reason: str, evidence: Dict[str, object]) -> None:
        # Merge with an existing event on the same side within a few seconds.
        # Evidence accumulates across signals (booleans OR together), and two
        # independent corroborating signals raise confidence slightly.
        for existing in events:
            if existing.side == side and abs(existing.t - t) < cfg.goal_merge_window_s:
                merged = dict(existing.evidence)
                for key, value in evidence.items():
                    if isinstance(value, bool):
                        merged[key] = bool(merged.get(key)) or value
                    elif merged.get(key) is None:
                        merged[key] = value
                corroborated = bool(merged.get("observed_in_goal_box")) and bool(
                    merged.get("line_crossed_while_visible")
                )
                if confidence > existing.confidence:
                    existing.t = t
                    existing.reason = reason
                existing.confidence = min(
                    0.98, max(existing.confidence, confidence) + (0.05 if corroborated else 0.0)
                )
                existing.evidence = merged
                return
        events.append(GoalEvent(t=t, side=side, confidence=confidence, reason=reason, evidence=evidence))

    def _entered_from_field(goal: GoalBox, first_idx: int) -> bool:
        """True when the ball moved INTO the goal from the field side.

        Rejects goal kicks: there the ball appears inside/near the goal box
        after a long invisible stretch and immediately moves infield.
        """
        t_first = float(ball_track.times[first_idx])
        vx, _vy = ball_track.velocity_at(t_first, window_s=0.5)
        gate = cfg.goal_entry_speed_px_s
        into_goal = vx < -gate if goal.side == "left" else vx > gate
        if into_goal:
            return True
        # Slow/stationary ball in the net: accept if it was visible on the
        # field side of the goal line moments before this sighting.
        if first_idx > 0:
            prev_t = float(ball_track.times[first_idx - 1])
            if t_first - prev_t <= 1.0:
                prev_x = float(ball_track.xs[first_idx - 1])
                line_x = _goal_line_x(goal)
                return prev_x > line_x if goal.side == "left" else prev_x < line_x
        return False

    # Signal 1: ball directly observed inside a goal box.
    for goal in (geometry.left_goal, geometry.right_goal):
        mouth_y1, mouth_y2 = _mouth_y_range(goal)
        mask = (
            (ball_track.xs >= goal.x1) & (ball_track.xs <= goal.x2)
            & (ball_track.ys >= mouth_y1) & (ball_track.ys <= mouth_y2)
            & (ball_track.times >= start_s) & (ball_track.times <= end_s)
        )
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        # Group consecutive index runs into sightings.
        runs: List[Tuple[int, int]] = []
        run_start = idxs[0]
        prev = idxs[0]
        for i in idxs[1:]:
            if float(ball_track.times[i] - ball_track.times[prev]) > cfg.goal_run_gap_s:
                runs.append((run_start, prev))
                run_start = i
            prev = i
        runs.append((run_start, prev))
        for a, b in runs:
            t = float(ball_track.times[a])
            if not _entered_from_field(goal, int(a)):
                LOGGER.debug(
                    "goal-box sighting at t=%.2fs (%s) rejected: ball did not enter from the field "
                    "(likely a goal kick or stray detection)",
                    t, goal.side,
                )
                continue
            confidence = 0.75
            reason = f"ball observed inside {goal.side} goal"
            evidence: Dict[str, object] = {
                "observed_in_goal_box": True,
                "first_seen_in_goal_s": round(t, 3),
                "samples_in_goal": int(b - a + 1),
            }
            kickoff_t = _reappears_near_center(float(ball_track.times[b]))
            if kickoff_t is not None:
                confidence += 0.15
                evidence["kickoff_reappearance_s"] = round(kickoff_t, 3)
                reason += "; kickoff restart observed"
            _add_event(t, goal.side, min(0.98, confidence), reason, evidence)

    # Signal 2: ball observed crossing the goal line between the posts (also
    # catches shots that skip past the shallow goal box between frames).
    if len(ball_track) >= 2:
        xs, ys, times = ball_track.xs, ball_track.ys, ball_track.times
        dts = np.diff(times)
        for goal in (geometry.left_goal, geometry.right_goal):
            line_x = _goal_line_x(goal)
            mouth_y1, mouth_y2 = _mouth_y_range(goal)
            if goal.side == "left":
                crossed = (xs[:-1] >= line_x) & (xs[1:] < line_x)
            else:
                crossed = (xs[:-1] <= line_x) & (xs[1:] > line_x)
            crossed &= dts <= 0.5
            crossed &= (times[1:] >= start_s) & (times[:-1] <= end_s)
            for i in np.flatnonzero(crossed):
                t0, t1 = float(times[i]), float(times[i + 1])
                x0, x1p = float(xs[i]), float(xs[i + 1])
                span = x1p - x0
                alpha = (line_x - x0) / span if abs(span) > 1e-6 else 0.0
                y_cross = float(ys[i]) + (float(ys[i + 1]) - float(ys[i])) * alpha
                if not (mouth_y1 <= y_cross <= mouth_y2):
                    continue
                t_cross = t0 + (t1 - t0) * alpha
                # A crossing alone is weak on real footage (geometry error,
                # ball rolling past the post): it must be corroborated below
                # to clear the confidence floor.
                confidence = 0.55
                evidence = {
                    "observed_in_goal_box": False,
                    "line_crossed_while_visible": True,
                    "crossing_y": round(y_cross, 1),
                    "goal_line_x": round(line_x, 1),
                }
                reason = f"ball observed crossing the {goal.side} goal line between the posts"
                # Corroboration: ball STAYS behind the line for a while
                # (sitting in the net) rather than rolling straight back.
                behind_hi = int(np.searchsorted(times, t_cross + 1.0))
                behind = xs[i + 1 : behind_hi]
                if len(behind) >= 3 and (
                    np.all(behind < line_x) if goal.side == "left" else np.all(behind > line_x)
                ):
                    confidence += 0.15
                    evidence["stayed_behind_line"] = True
                    reason += "; ball stayed behind the line"
                if any(
                    gap_end > t_cross and gap_start < t_cross + 6.0
                    and (gap_end - gap_start) >= cfg.goal_disappear_confirm_s
                    for gap_start, gap_end in all_gaps
                ):
                    confidence += 0.15
                    reason += "; ball out of sight afterwards"
                kickoff_t = _reappears_near_center(t_cross + 0.5)
                if kickoff_t is not None:
                    confidence += 0.15
                    evidence["kickoff_reappearance_s"] = round(kickoff_t, 3)
                    reason += "; kickoff restart observed"
                _add_event(t_cross, goal.side, min(0.98, confidence), reason, evidence)

    # Signal 3: ball vanishes while heading into a goal mouth.
    for gap_start, gap_end in all_gaps:
        if gap_start <= float(ball_track.times[0]):
            continue
        # A gap that runs to the end of the window (recording stopped, keeper
        # held the ball) is weak evidence: no disappearance bonus, and with no
        # kickoff possible the event usually falls below min_goal_confidence.
        is_trailing_gap = gap_start >= track_end_t - 1e-6
        last = ball_track.position_at(gap_start)
        if last is None:
            continue
        vx, vy = ball_track.velocity_at(gap_start - 0.05, window_s=cfg.goal_lookback_s)
        speed = math.hypot(vx, vy)
        if speed < min_speed:
            continue
        for goal in (geometry.left_goal, geometry.right_goal):
            goal_line_x = _goal_line_x(goal)
            heading_out = vx < 0 if goal.side == "left" else vx > 0
            if not heading_out:
                continue
            dx = goal_line_x - last[0]
            time_to_line = dx / vx if abs(vx) > 1e-6 else float("inf")
            if not (0.0 <= time_to_line <= 0.8):
                continue
            y_at_line = last[1] + vy * time_to_line
            mouth_y1, mouth_y2 = _mouth_y_range(goal)
            if not (mouth_y1 <= y_at_line <= mouth_y2):
                continue
            gap_len = gap_end - gap_start
            confidence = 0.55
            evidence = {
                "observed_in_goal_box": False,
                "last_seen_xy": [round(last[0], 1), round(last[1], 1)],
                "velocity_px_s": [round(vx, 1), round(vy, 1)],
                "projected_goal_line_y": round(y_at_line, 1),
                "disappeared_for_s": round(gap_len, 3),
            }
            reason = f"ball vanished heading into {goal.side} goal mouth"
            if gap_len >= cfg.goal_disappear_confirm_s and not is_trailing_gap:
                confidence += 0.15
                reason += f"; stayed out of sight {gap_len:.1f}s"
            kickoff_t = _reappears_near_center(gap_end - 0.1)
            if kickoff_t is not None:
                confidence += 0.15
                evidence["kickoff_reappearance_s"] = round(kickoff_t, 3)
                reason += "; kickoff restart observed"
            _add_event(gap_start, goal.side, min(0.98, confidence), reason, evidence)

    kept: List[GoalEvent] = []
    for event in sorted(events, key=lambda e: e.t):
        if event.confidence < cfg.min_goal_confidence:
            LOGGER.info(
                "goal candidate dropped (confidence %.2f < %.2f): t=%.2fs side=%s (%s)",
                event.confidence, cfg.min_goal_confidence, event.t, event.side, event.reason,
            )
            continue
        LOGGER.info("goal flagged: t=%.2fs side=%s confidence=%.2f (%s)",
                    event.t, event.side, event.confidence, event.reason)
        kept.append(event)
    return kept


# ---------------------------------------------------------------------------
# Game state timeline
# ---------------------------------------------------------------------------


def analyze_game_states(
    ball_track: BallTrack,
    geometry: FieldGeometry,
    start_s: float,
    end_s: float,
    config: Optional[GameStateConfig] = None,
    goal_events: Optional[Sequence[GoalEvent]] = None,
) -> List[GameStateSegment]:
    """Classify the timeline into game states with restart/goal holds.

    The key behavior: when the ball leaves play over a goal line (or vanishes
    right next to a goal), the state pins to ``restart_<side>`` until the ball
    is confirmed back in play - this is what keeps the camera at the goal
    while everyone waits for the goal kick or corner.
    """
    cfg = config or GameStateConfig()
    if end_s <= start_s:
        return []

    goals = sorted(goal_events or [], key=lambda e: e.t)
    out_margin = cfg.out_margin_frac * geometry.width

    steps = max(1, int(math.ceil((end_s - start_s) / cfg.step_s)))
    raw_states: List[Tuple[float, str, Optional[str], str]] = []  # (t, state, side, reason)

    state = STATE_IN_PLAY
    side: Optional[str] = None
    reason = "start of window"
    invisible_since: Optional[float] = None
    in_play_streak_start: Optional[float] = None
    restart_started_at: Optional[float] = None
    last_seen_xy: Optional[Tuple[float, float]] = None
    goal_idx = 0
    goal_hold_until: Optional[float] = None
    goal_side: Optional[str] = None

    for i in range(steps + 1):
        t = min(end_s, start_s + i * cfg.step_s)

        # Goal holds take priority over everything else.
        while goal_idx < len(goals) and goals[goal_idx].t <= t:
            goal_hold_until = goals[goal_idx].t + cfg.goal_hold_s
            goal_side = goals[goal_idx].side
            goal_idx += 1
        if goal_hold_until is not None and t <= goal_hold_until:
            state = STATE_GOAL_LEFT if goal_side == "left" else STATE_GOAL_RIGHT
            side = goal_side
            reason = f"goal scored at {goal_side} goal - holding on goal celebration/kickoff"
            raw_states.append((t, state, side, reason))
            invisible_since = None
            in_play_streak_start = None
            restart_started_at = None
            continue
        if goal_hold_until is not None and t > goal_hold_until:
            goal_hold_until = None
            goal_side = None
            state = STATE_IN_PLAY
            reason = "goal hold released"
            side = None

        pos = ball_track.position_at(t)
        if pos is not None:
            x, y, _source = pos
            last_seen_xy = (x, y)
            invisible_since = None
            beyond_left = x < geometry.x_min - out_margin
            beyond_right = x > geometry.x_max + out_margin
            beyond_touch = y < geometry.y_min - out_margin or y > geometry.y_max + out_margin
            in_field = not (beyond_left or beyond_right or beyond_touch)

            if in_field:
                if in_play_streak_start is None:
                    in_play_streak_start = t
                if state in RESTART_STATES:
                    held_long_enough = (t - in_play_streak_start) >= cfg.return_confirm_s
                    hit_cap = restart_started_at is not None and (t - restart_started_at) >= cfg.max_restart_hold_s
                    if held_long_enough or hit_cap:
                        state = STATE_IN_PLAY
                        side = None
                        reason = "ball back in play"
                        restart_started_at = None
                else:
                    state = STATE_IN_PLAY
                    side = None
                    reason = "ball visible in field"
            else:
                in_play_streak_start = None
                if beyond_left or beyond_right:
                    new_side = "left" if beyond_left else "right"
                    if state not in RESTART_STATES or side != new_side:
                        restart_started_at = t
                    state = STATE_RESTART_LEFT if new_side == "left" else STATE_RESTART_RIGHT
                    side = new_side
                    reason = (
                        f"ball out over the {new_side} goal line - waiting for goal kick/corner, "
                        "holding camera at the goal"
                    )
                else:
                    if state != STATE_RESTART_TOUCHLINE:
                        restart_started_at = t
                    state = STATE_RESTART_TOUCHLINE
                    side = None
                    reason = "ball out over the touchline - waiting for throw-in"
        else:
            in_play_streak_start = None
            if invisible_since is None:
                invisible_since = t
            invisible_for = t - invisible_since
            if state in RESTART_STATES or state in GOAL_STATES:
                # Keep holding; the ball being invisible is expected while
                # someone fetches it.
                if restart_started_at is not None and (t - restart_started_at) >= cfg.max_restart_hold_s:
                    state = STATE_BALL_LOST
                    side = None
                    reason = "restart hold exceeded safety cap - reverting to ball_lost"
                    restart_started_at = None
            elif invisible_for >= cfg.lost_grace_s:
                near_side = None
                if last_seen_xy is not None:
                    near_side = geometry.side_if_near_goal(
                        last_seen_xy[0], last_seen_xy[1], cfg.near_goal_frac
                    )
                if near_side is not None:
                    if state not in RESTART_STATES or side != near_side:
                        restart_started_at = t
                    state = STATE_RESTART_LEFT if near_side == "left" else STATE_RESTART_RIGHT
                    side = near_side
                    reason = (
                        f"ball vanished near the {near_side} goal - assuming goal kick/corner wait, "
                        "holding camera at the goal"
                    )
                else:
                    state = STATE_BALL_LOST
                    side = None
                    reason = f"ball not visible for {invisible_for:.1f}s - following player cluster"

        raw_states.append((t, state, side, reason))

    # Collapse consecutive identical states into segments.
    segments: List[GameStateSegment] = []
    for t, st, sd, rsn in raw_states:
        if segments and segments[-1].state == st and segments[-1].side == sd:
            segments[-1].end_s = t
        else:
            if segments:
                segments[-1].end_s = t
            segments.append(GameStateSegment(start_s=t, end_s=t, state=st, side=sd, reason=rsn))
    if segments:
        segments[-1].end_s = end_s

    # Drop zero-length artifacts.
    segments = [seg for seg in segments if seg.end_s - seg.start_s > 1e-6]
    for seg in segments:
        LOGGER.debug(
            "game state %.1fs-%.1fs: %s%s (%s)",
            seg.start_s, seg.end_s, seg.state, f"[{seg.side}]" if seg.side else "", seg.reason,
        )
    return segments


@dataclass
class SetPieceEvent:
    """A dead-ball restart: the ball sat still, then was kicked."""

    kind: str  # corner_kick | free_kick | penalty_kick | goal_kick | kickoff
    t_start: float  # when the ball became stationary
    t_kick: float  # when it accelerated away
    x: float
    y: float
    side: Optional[str]  # threatened goal ("left"/"right"), if any
    reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "t_start": round(self.t_start, 3),
            "t_kick": round(self.t_kick, 3),
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "side": self.side,
            "reason": self.reason,
        }


def detect_set_pieces(
    ball_track: BallTrack,
    geometry: FieldGeometry,
    start_s: float,
    end_s: float,
    config: Optional[GameStateConfig] = None,
) -> List[SetPieceEvent]:
    """Find dead-ball restarts from the stationary-ball + kick signature.

    A set piece is a window where the visible ball stays inside a small
    radius for a minimum time and then accelerates away. The location of the
    stationary spot classifies it: field corner -> corner kick, in front of a
    goal on the penalty spot -> penalty, inside the goal-kick zone -> goal
    kick, center circle -> kickoff, anywhere else -> free kick (with the
    threatened goal recorded when it is within shooting range).
    """
    cfg = config or GameStateConfig()
    events: List[SetPieceEvent] = []
    n = len(ball_track)
    if n < 3:
        return events

    frame_w = ball_track.frame_size[0]
    radius = cfg.set_piece_stationary_radius_frac * frame_w
    kick_speed = cfg.set_piece_kick_speed_frame_widths_per_s * frame_w
    times, xs, ys = ball_track.times, ball_track.xs, ball_track.ys
    center_x = (geometry.x_min + geometry.x_max) / 2.0
    center_y = (geometry.y_min + geometry.y_max) / 2.0

    def _classify(x: float, y: float) -> Tuple[str, Optional[str]]:
        corner_r = cfg.corner_radius_frac * geometry.width
        for cx in (geometry.x_min, geometry.x_max):
            for cy in (geometry.y_min, geometry.y_max):
                if math.hypot(x - cx, y - cy) <= corner_r:
                    return "corner_kick", "left" if cx == geometry.x_min else "right"
        for goal in (geometry.left_goal, geometry.right_goal):
            line_x = goal.x2 if goal.side == "left" else goal.x1
            depth = abs(x - line_x)
            toward_field = (x > line_x) if goal.side == "left" else (x < line_x)
            if not toward_field:
                continue
            gy = goal.center[1]
            lo, hi = cfg.penalty_depth_range_frac
            if lo * geometry.width <= depth <= hi * geometry.width and abs(y - gy) <= cfg.penalty_half_height_frac * geometry.height:
                return "penalty_kick", goal.side
            if depth <= cfg.goal_kick_zone_depth_frac * geometry.width and abs(y - gy) <= cfg.goal_kick_zone_half_height_frac * geometry.height:
                return "goal_kick", goal.side
        if math.hypot(x - center_x, y - center_y) <= cfg.kickoff_center_frac * geometry.width:
            return "kickoff", None
        threat = cfg.free_kick_threat_frac * geometry.width
        side = None
        if x - geometry.x_min <= threat:
            side = "left"
        elif geometry.x_max - x <= threat:
            side = "right"
        return "free_kick", side

    i = 0
    while i < n - 1:
        if times[i] < start_s:
            i += 1
            continue
        if times[i] > end_s:
            break
        # Grow a stationary window anchored at sample i.
        j = i + 1
        anchor_x, anchor_y = float(xs[i]), float(ys[i])
        while j < n and times[j] <= end_s:
            if (times[j] - times[j - 1]) > ball_track.config.max_interpolation_gap_s:
                break
            if math.hypot(float(xs[j]) - anchor_x, float(ys[j]) - anchor_y) > radius:
                break
            j += 1
        window_len = float(times[j - 1] - times[i])
        if window_len >= cfg.set_piece_min_stationary_s and j < n:
            t_kick = float(times[j - 1])
            # Probe twice after the window ends: a kicked ball is still
            # accelerating, so the later probe catches slower restarts.
            speeds = [
                math.hypot(*ball_track.velocity_at(t_kick + 0.3, window_s=0.5)),
                math.hypot(*ball_track.velocity_at(t_kick + 0.7, window_s=0.5)),
            ]
            if max(speeds) >= kick_speed:
                kind, side = _classify(anchor_x, anchor_y)
                reason = f"{kind.replace('_', ' ')} detected: ball held still {window_len:.1f}s then kicked"
                if side is not None and kind in {"corner_kick", "free_kick", "penalty_kick"}:
                    reason += f"; threatens the {side} goal"
                events.append(
                    SetPieceEvent(
                        kind=kind, t_start=float(times[i]), t_kick=t_kick,
                        x=anchor_x, y=anchor_y, side=side, reason=reason,
                    )
                )
                LOGGER.info("set piece: %s at t=%.1fs-%.1fs (%.0f, %.0f) side=%s",
                            kind, float(times[i]), t_kick, anchor_x, anchor_y, side)
                i = j
                continue
        i += 1
    return events


def overlay_set_piece_states(
    segments: List[GameStateSegment],
    set_pieces: Sequence[SetPieceEvent],
) -> List[GameStateSegment]:
    """Carve corner/free-kick setup states into the base state timeline.

    Goal celebrations keep priority; set-piece setup windows replace whatever
    other state covered [t_start, t_kick] so the camera planner can frame the
    restart (ball AND threatened goal in view).
    """
    overlays: List[GameStateSegment] = []
    for sp in set_pieces:
        if sp.kind == "corner_kick" and sp.side:
            state = STATE_CORNER_SETUP
        elif sp.kind in {"free_kick", "penalty_kick"} and sp.side:
            state = STATE_FREE_KICK_SETUP
        else:
            continue  # goal kicks/kickoffs already behave correctly
        overlays.append(GameStateSegment(
            start_s=sp.t_start, end_s=sp.t_kick, state=state, side=sp.side,
            reason=sp.reason,
        ))

    result = list(segments)
    for overlay in sorted(overlays, key=lambda s: s.start_s):
        updated: List[GameStateSegment] = []
        for seg in result:
            if seg.state in GOAL_STATES or seg.end_s <= overlay.start_s or seg.start_s >= overlay.end_s:
                updated.append(seg)
                continue
            if seg.start_s < overlay.start_s:
                updated.append(GameStateSegment(seg.start_s, overlay.start_s, seg.state, seg.side, seg.reason))
            updated.append(GameStateSegment(
                max(seg.start_s, overlay.start_s), min(seg.end_s, overlay.end_s),
                overlay.state, overlay.side, overlay.reason,
            ))
            if seg.end_s > overlay.end_s:
                updated.append(GameStateSegment(overlay.end_s, seg.end_s, seg.state, seg.side, seg.reason))
        result = updated

    # Merge adjacent identical states created by the splitting.
    merged: List[GameStateSegment] = []
    for seg in sorted(result, key=lambda s: s.start_s):
        if seg.end_s - seg.start_s <= 1e-6:
            continue
        if merged and merged[-1].state == seg.state and merged[-1].side == seg.side \
                and abs(merged[-1].end_s - seg.start_s) < 1e-6:
            merged[-1].end_s = seg.end_s
        else:
            merged.append(seg)
    return merged


def state_at(segments: Sequence[GameStateSegment], t: float) -> Optional[GameStateSegment]:
    """Return the segment covering time ``t`` (or the last one before it)."""
    current: Optional[GameStateSegment] = None
    for seg in segments:
        if seg.start_s <= t:
            current = seg
        else:
            break
        if t < seg.end_s:
            return seg
    return current


def summarize_states(segments: Sequence[GameStateSegment]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for seg in segments:
        totals[seg.state] = totals.get(seg.state, 0.0) + (seg.end_s - seg.start_s)
    return {state: round(duration, 2) for state, duration in sorted(totals.items())}
