"""Team identification by uniform color and team-level match stats.

The user configures two teams by name + jersey color (hex, from a color
picker). Player positions collected during tracking are then labeled by
sampling a small torso patch around each position in ~1Hz sampled frames
and matching its dominant color to the nearer team color in HSV space.

From the labeled positions we derive:

* which side each team defends (median x per period - handles the
  halftime swap),
* possession (nearest labeled player to the ball, sampled over in-play
  time),
* territory (share of team presence in each third),
* goal attribution: a goal INTO a side's goal is scored BY the other team.

Everything lands in ``analysis_team_stats.json`` and the goal/card
bookmarks gain a ``team`` field.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("videohighlights.team_classification")


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for team classification") from exc
    return cv2


def hex_to_bgr(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Invalid hex color: {value!r}")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return b, g, r


@dataclass
class TeamConfig:
    name: str
    color_hex: str  # jersey color, e.g. "#d32f2f"


@dataclass
class TeamClassifierConfig:
    sample_fps: float = 1.0
    patch_half_px: int = 12
    # A patch must be at least this close (HSV distance) to SOME team color
    # to be labeled; otherwise unknown (-1). Hue is circular, weighted high.
    max_color_distance: float = 0.45
    # And clearly closer to one team than the other.
    min_margin: float = 0.08
    # Minimum saturation for a patch to be color-classifiable (grass-green
    # shadows and white kits need the value/hue combination below).
    possession_radius_frac: float = 0.06  # of frame width
    max_samples: int = 4000


def _hsv_of_bgr(bgr: Tuple[int, int, int]) -> Tuple[float, float, float]:
    cv2 = _import_cv2()
    px = np.uint8([[list(bgr)]])
    h, s, v = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0]
    return float(h), float(s), float(v)


def _color_distance(hsv_a: Tuple[float, float, float], hsv_b: Tuple[float, float, float]) -> float:
    """Perceptual-ish HSV distance in [0, ~1.7]: circular hue + sat + value."""
    dh = abs(hsv_a[0] - hsv_b[0])
    dh = min(dh, 180.0 - dh) / 90.0  # 0..1
    ds = abs(hsv_a[1] - hsv_b[1]) / 255.0
    dv = abs(hsv_a[2] - hsv_b[2]) / 255.0
    # Low-saturation colors (white/black kits) carry no hue information.
    sat_weight = min(hsv_a[1], hsv_b[1]) / 255.0
    return dh * (0.6 + 0.8 * sat_weight) + ds * 0.5 + dv * 0.35


def classify_player_teams(
    video_path: str,
    player_positions: Optional[np.ndarray],
    team_a: TeamConfig,
    team_b: TeamConfig,
    config: Optional[TeamClassifierConfig] = None,
) -> np.ndarray:
    """Label player positions by team: returns (t, x, y, team) rows.

    team: 0 = team_a, 1 = team_b, -1 = unknown (referee, keeper in a third
    color, occluded patch, off-color pixels).
    """
    cfg = config or TeamClassifierConfig()
    if player_positions is None or len(player_positions) == 0:
        return np.empty((0, 4), dtype=np.float32)
    cv2 = _import_cv2()

    target_a = _hsv_of_bgr(hex_to_bgr(team_a.color_hex))
    target_b = _hsv_of_bgr(hex_to_bgr(team_b.color_hex))

    positions = np.asarray(player_positions, dtype=np.float64)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        LOGGER.warning("team classification skipped: cannot open %s", video_path)
        return np.empty((0, 4), dtype=np.float32)

    labeled: List[Tuple[float, float, float, int]] = []
    try:
        t_min, t_max = float(positions[:, 0].min()), float(positions[:, 0].max())
        step = 1.0 / max(0.2, cfg.sample_fps)
        t = t_min
        while t <= t_max and len(labeled) < cfg.max_samples:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break
            actual_t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, w = frame.shape[:2]
            mask = np.abs(positions[:, 0] - actual_t) <= 0.15
            for _, px, py in positions[mask][:30]:
                x0 = int(max(0, px - cfg.patch_half_px))
                x1 = int(min(w, px + cfg.patch_half_px))
                y0 = int(max(0, py - cfg.patch_half_px))
                y1 = int(min(h, py + cfg.patch_half_px))
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                patch = hsv[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64)
                # Drop grass pixels (green hue band) before taking the median.
                grass = (patch[:, 0] > 35) & (patch[:, 0] < 85) & (patch[:, 1] > 60)
                kept = patch[~grass]
                if len(kept) < 12:
                    continue
                med = tuple(np.median(kept, axis=0))
                da = _color_distance(med, target_a)
                db = _color_distance(med, target_b)
                if min(da, db) > cfg.max_color_distance or abs(da - db) < cfg.min_margin:
                    team = -1
                else:
                    team = 0 if da < db else 1
                labeled.append((actual_t, float(px), float(py), team))
            t += step
    finally:
        cap.release()

    result = np.asarray(labeled, dtype=np.float32) if labeled else np.empty((0, 4), dtype=np.float32)
    known = int((result[:, 3] >= 0).sum()) if len(result) else 0
    LOGGER.info(
        "team classification: %d samples, %d labeled (%s vs %s)",
        len(result), known, team_a.name, team_b.name,
    )
    return result


def compute_team_stats(
    labeled_positions: np.ndarray,
    ball_track,
    geometry,
    goal_events: Sequence[object],
    team_a: TeamConfig,
    team_b: TeamConfig,
    duration_s: float,
    config: Optional[TeamClassifierConfig] = None,
) -> Dict[str, object]:
    """Team-level stats + goal attribution from labeled positions."""
    cfg = config or TeamClassifierConfig()
    names = {0: team_a.name, 1: team_b.name}
    stats: Dict[str, object] = {
        "teams": [
            {"team": team_a.name, "color": team_a.color_hex},
            {"team": team_b.name, "color": team_b.color_hex},
        ],
        "label_counts": {},
        "defending_side": {},
        "possession_pct": {},
        "territory_pct": {},
        "goals": {team_a.name: 0, team_b.name: 0},
        "goal_attribution": [],
        "periods": [],
    }
    if labeled_positions is None or len(labeled_positions) == 0:
        stats["note"] = "no labeled player positions - check team colors"
        return stats

    rows = labeled_positions
    known = rows[rows[:, 3] >= 0]
    stats["label_counts"] = {
        team_a.name: int((known[:, 3] == 0).sum()),
        team_b.name: int((known[:, 3] == 1).sum()),
        "unknown": int((rows[:, 3] < 0).sum()),
    }

    # Defending side per period (median x per team; handles halftime swap).
    mid_x = (geometry.x_min + geometry.x_max) / 2.0
    period_edges = [0.0, duration_s / 2.0, duration_s]
    periods: List[Dict[str, object]] = []
    for p0, p1 in zip(period_edges[:-1], period_edges[1:]):
        window = known[(known[:, 0] >= p0) & (known[:, 0] < p1)]
        sides: Dict[str, str] = {}
        for team in (0, 1):
            tp = window[window[:, 3] == team]
            if len(tp) >= 20:
                sides[names[team]] = "left" if float(np.median(tp[:, 1])) < mid_x else "right"
        periods.append({"start_s": round(p0, 1), "end_s": round(p1, 1), "defending": sides})
    stats["periods"] = periods
    if periods and periods[0]["defending"]:
        stats["defending_side"] = periods[0]["defending"]

    def _defender_of(side: str, t: float) -> Optional[str]:
        period = periods[0] if t < duration_s / 2.0 else periods[-1]
        for team_name, team_side in (period.get("defending") or {}).items():
            if team_side == side:
                return team_name
        return None

    # Possession: nearest labeled player to the ball at 1s steps.
    radius = cfg.possession_radius_frac * (geometry.frame_size[0] if hasattr(geometry, "frame_size") else 1920)
    counts = {team_a.name: 0, team_b.name: 0}
    thirds = {team_a.name: [0, 0, 0], team_b.name: [0, 0, 0]}
    third_w = geometry.width / 3.0
    for t in np.arange(0.0, duration_s, 1.0):
        window = known[np.abs(known[:, 0] - t) <= 0.6]
        for team in (0, 1):
            tp = window[window[:, 3] == team]
            if len(tp):
                mean_x = float(np.mean(tp[:, 1]))
                idx = int(min(2, max(0, (mean_x - geometry.x_min) // third_w)))
                thirds[names[team]][idx] += 1
        ball = ball_track.position_at(float(t)) if ball_track is not None else None
        if ball is None or len(window) == 0:
            continue
        dists = np.hypot(window[:, 1] - ball[0], window[:, 2] - ball[1])
        nearest = int(np.argmin(dists))
        if dists[nearest] <= radius * 3.0:
            counts[names[int(window[nearest, 3])]] += 1
    total = sum(counts.values())
    if total:
        stats["possession_pct"] = {k: round(100.0 * v / total, 1) for k, v in counts.items()}
    for team_name, buckets in thirds.items():
        s = sum(buckets)
        if s:
            stats["territory_pct"][team_name] = [round(100.0 * b / s, 1) for b in buckets]

    # Goal attribution: goal INTO a side's goal = scored by the OTHER team.
    for goal in goal_events or []:
        side = getattr(goal, "side", None) or (goal.get("side") if isinstance(goal, dict) else None)
        t = float(getattr(goal, "t", None) or (goal.get("t") if isinstance(goal, dict) else 0.0))
        defender = _defender_of(str(side), t)
        scorer = None
        if defender == team_a.name:
            scorer = team_b.name
        elif defender == team_b.name:
            scorer = team_a.name
        if scorer:
            stats["goals"][scorer] = int(stats["goals"].get(scorer, 0)) + 1
        stats["goal_attribution"].append(
            {"t": round(t, 3), "into_goal": side, "team": scorer, "defending_team": defender}
        )
    return stats
