"""Broadcast polish: story-aware clip boundaries and the highlight reel.

Two editor's-craft capabilities:

1. **Boundary refinement** - a highlight starts where the *move* began (the
   dead ball or the change of attacking direction that launched it), and it
   ends when the *emotion* resolves (crowd noise decays back to baseline),
   not at fixed offsets.
2. **Reel building** - a broadcast-style montage: a cold-open teaser of the
   best moment, chronological clips joined with crossfades and per-clip
   audio normalization, and slow-motion replays spliced in after goals.

moviepy 1.x and 2.x are both supported via the small compat helpers.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("videohighlights.broadcast")


@dataclass
class BroadcastConfig:
    # Story-aware starts.
    max_lookback_s: float = 15.0
    preroll_s: float = 1.5
    direction_scan_step_s: float = 0.5
    # Emotion-aware endings.
    min_post_s: float = 4.0
    max_post_s: float = 8.0
    max_post_goal_s: float = 12.0
    decay_fraction: float = 0.25  # end when RMS falls below baseline + frac*(peak-baseline)
    decay_sustain_s: float = 1.0
    # Reel construction.
    cold_open_s: float = 3.0
    crossfade_s: float = 0.5
    replay_speed: float = 0.4
    replay_pre_s: float = 4.0
    replay_post_s: float = 1.0
    fade_out_s: float = 1.0


# ---------------------------------------------------------------------------
# Boundary refinement
# ---------------------------------------------------------------------------


def compute_audio_envelope(video_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(times, rms) crowd-noise envelope, or None when audio is unavailable."""
    try:
        import librosa

        y, sr = librosa.load(video_path, sr=None, mono=True)
        hop = int(0.1 * sr)
        win = int(0.2 * sr)
        rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=True).flatten()
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop, n_fft=win)
        return times.astype(np.float64), rms.astype(np.float64)
    except Exception as exc:
        LOGGER.info("audio envelope unavailable (%s); using fixed clip endings", exc)
        return None


def story_start(
    event_t: float,
    ball_track,
    segments: Sequence[object],
    goal_side: Optional[str],
    config: Optional[BroadcastConfig] = None,
) -> float:
    """Walk back from an event to where the move began.

    The start is the latest of: the end of the previous dead-ball period
    (restart/set piece/goal hold), the moment the attack toward the scored
    goal began (ball x-velocity turned toward it), and a hard lookback cap.
    """
    cfg = config or BroadcastConfig()
    floor_t = max(0.0, event_t - cfg.max_lookback_s)

    dead_ball_end = floor_t
    for seg in segments:
        state = getattr(seg, "state", None) or (seg.get("state") if isinstance(seg, dict) else "")
        end_s = float(getattr(seg, "end_s", None) or (seg.get("end_s") if isinstance(seg, dict) else 0.0))
        if end_s <= event_t and state != "in_play" and end_s > dead_ball_end:
            dead_ball_end = end_s

    direction_origin = floor_t
    if goal_side in {"left", "right"} and ball_track is not None and len(ball_track) >= 2:
        toward = -1.0 if goal_side == "left" else 1.0
        t = event_t
        while t > floor_t:
            vx, _vy = ball_track.velocity_at(t, window_s=0.6)
            if vx * toward <= 0.0 and abs(vx) > 15.0:
                # Ball was clearly moving the other way here: the attack
                # started after this point.
                direction_origin = t
                break
            t -= cfg.direction_scan_step_s

    start = max(floor_t, dead_ball_end, direction_origin) - cfg.preroll_s
    return max(0.0, min(start, event_t - 2.0))


def emotion_end(
    event_t: float,
    envelope: Optional[Tuple[np.ndarray, np.ndarray]],
    is_goal: bool,
    config: Optional[BroadcastConfig] = None,
) -> float:
    """End the clip when crowd noise has decayed back toward baseline."""
    cfg = config or BroadcastConfig()
    max_post = cfg.max_post_goal_s if is_goal else cfg.max_post_s
    if envelope is None:
        return event_t + max_post
    times, rms = envelope
    if len(times) < 8:
        return event_t + max_post

    baseline = float(np.median(rms))
    peak_lo = int(np.searchsorted(times, event_t))
    peak_hi = int(np.searchsorted(times, event_t + 3.0))
    peak = float(rms[peak_lo:peak_hi].max()) if peak_hi > peak_lo else baseline
    if peak <= baseline:
        return event_t + max_post
    threshold = baseline + cfg.decay_fraction * (peak - baseline)

    lo = int(np.searchsorted(times, event_t + cfg.min_post_s))
    hi = int(np.searchsorted(times, event_t + max_post))
    below_since: Optional[float] = None
    for i in range(lo, min(hi, len(times))):
        if rms[i] < threshold:
            if below_since is None:
                below_since = float(times[i])
            elif float(times[i]) - below_since >= cfg.decay_sustain_s:
                return below_since
        else:
            below_since = None
    return event_t + max_post


def refine_intervals(
    intervals: Sequence[Tuple[float, float]],
    event_rows: Sequence[Dict[str, object]],
    ball_track,
    segments: Sequence[object],
    envelope: Optional[Tuple[np.ndarray, np.ndarray]],
    duration_s: float,
    config: Optional[BroadcastConfig] = None,
) -> List[Tuple[float, float]]:
    """Refine clip boundaries around known events.

    ``event_rows``: dicts with ``t``, ``event_type`` and optional ``side``
    (goals/cards/set-piece kicks) in the same timebase as ``intervals``.
    Intervals containing an event get a story-aware start (goals) and an
    emotion-aware end; other intervals keep their start and get the audio
    ending when available.
    """
    cfg = config or BroadcastConfig()
    refined: List[Tuple[float, float]] = []
    for start_s, end_s in intervals:
        row = next(
            (r for r in event_rows if start_s <= float(r.get("t", -1.0)) <= end_s), None
        )
        new_start, new_end = start_s, end_s
        if row is not None:
            event_t = float(row["t"])
            is_goal = str(row.get("event_type")) == "goal"
            if is_goal:
                new_start = min(
                    start_s,
                    story_start(event_t, ball_track, segments, row.get("side"), cfg),
                )
            new_end = max(end_s, emotion_end(event_t, envelope, is_goal, cfg))
        elif envelope is not None:
            mid = (start_s + end_s) / 2.0
            new_end = max(end_s, emotion_end(mid, envelope, False, cfg))
        refined.append((max(0.0, new_start), min(duration_s, new_end)))
    # Boundary growth can create overlaps; merge them.
    refined.sort()
    merged: List[Tuple[float, float]] = []
    for s, e in refined:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


# ---------------------------------------------------------------------------
# Reel building (moviepy 1.x / 2.x compatible)
# ---------------------------------------------------------------------------


def _subclip(clip, start, end):
    try:
        return clip.subclipped(start, end)  # moviepy 2.x
    except AttributeError:
        return clip.subclip(start, end)  # moviepy 1.x


def _speed(clip, factor: float):
    try:
        from moviepy import vfx  # 2.x

        return clip.with_effects([vfx.MultiplySpeed(factor)])
    except Exception:
        from moviepy.video.fx.all import speedx  # 1.x

        return clip.fx(speedx, factor)


def _crossfadein(clip, duration: float):
    try:
        from moviepy import vfx  # 2.x

        return clip.with_effects([vfx.CrossFadeIn(duration)])
    except Exception:
        try:
            return clip.crossfadein(duration)  # 1.x
        except Exception:
            return clip


def _fadeout(clip, duration: float):
    try:
        from moviepy import vfx  # 2.x

        return clip.with_effects([vfx.FadeOut(duration)])
    except Exception:
        try:
            from moviepy.video.fx.all import fadeout  # 1.x

            return clip.fx(fadeout, duration)
        except Exception:
            return clip


def _normalize_audio(clip):
    if getattr(clip, "audio", None) is None:
        return clip
    try:
        from moviepy import afx  # 2.x

        return clip.with_effects([afx.AudioNormalize()])
    except Exception:
        try:
            from moviepy.audio.fx.all import audio_normalize  # 1.x

            return clip.fx(audio_normalize)
        except Exception:
            return clip


def _replay_banner(width: int, height: int):
    """Semi-transparent REPLAY banner as an ImageClip (no ImageMagick needed)."""
    import cv2

    banner_h = max(28, height // 12)
    banner_w = max(140, width // 5)
    img = np.zeros((banner_h, banner_w, 3), dtype=np.uint8)
    img[:] = (16, 16, 16)
    scale = banner_h / 46.0
    cv2.putText(img, "REPLAY", (int(banner_h * 0.4), int(banner_h * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), max(1, int(scale * 2)))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        from moviepy import ImageClip  # 2.x
    except Exception:
        from moviepy.editor import ImageClip  # 1.x
    return ImageClip(rgb), banner_w, banner_h


def build_broadcast_reel(
    clip_specs: Sequence[Dict[str, object]],
    output_path: str,
    config: Optional[BroadcastConfig] = None,
) -> Optional[str]:
    """Assemble the broadcast-style reel.

    ``clip_specs``: chronological dicts with ``path`` (rendered clip file),
    ``start_s``/``end_s`` (source timebase), and optional ``event_type``,
    ``occurred_at_s``, ``confidence``. Goals get a slow-motion replay after
    the live take; the best moment opens the reel as a teaser.
    """
    cfg = config or BroadcastConfig()
    try:
        try:
            from moviepy import CompositeVideoClip, VideoFileClip, concatenate_videoclips  # 2.x
        except Exception:
            from moviepy.editor import (  # 1.x
                CompositeVideoClip,
                VideoFileClip,
                concatenate_videoclips,
            )
    except Exception as exc:
        LOGGER.warning("broadcast reel skipped: moviepy unavailable (%s)", exc)
        return None

    specs = [s for s in clip_specs if s.get("path")]
    if not specs:
        return None

    opened: List[object] = []

    def _open(path: str):
        clip = VideoFileClip(str(path))
        opened.append(clip)
        return clip

    try:
        segments: List[object] = []

        # Cold open: 3s teaser of the best moment (goals outrank everything).
        def _score(s: Dict[str, object]) -> float:
            bonus = 2.0 if s.get("event_type") == "goal" else 0.0
            return bonus + float(s.get("confidence") or 0.0)

        best = max(specs, key=_score)
        best_clip = _open(str(best["path"]))
        if best.get("occurred_at_s") is not None:
            local_t = float(best["occurred_at_s"]) - float(best["start_s"])
        else:
            local_t = best_clip.duration / 2.0
        half = cfg.cold_open_s / 2.0
        t0 = max(0.0, min(local_t - half, best_clip.duration - cfg.cold_open_s))
        t1 = min(best_clip.duration, t0 + cfg.cold_open_s)
        if t1 - t0 >= 1.0:
            segments.append(_fadeout(_normalize_audio(_subclip(best_clip, t0, t1)), 0.4))

        # Chronological body with goal replays.
        for spec in specs:
            clip = _open(str(spec["path"]))
            body = _normalize_audio(clip)
            segments.append(body if not segments else _crossfadein(body, cfg.crossfade_s))

            if spec.get("event_type") == "goal" and spec.get("occurred_at_s") is not None:
                local_t = float(spec["occurred_at_s"]) - float(spec["start_s"])
                r0 = max(0.0, local_t - cfg.replay_pre_s)
                r1 = min(clip.duration, local_t + cfg.replay_post_s)
                if r1 - r0 >= 1.5:
                    replay = _speed(_subclip(clip, r0, r1), cfg.replay_speed)
                    try:
                        banner, bw, bh = _replay_banner(int(clip.w), int(clip.h))
                        banner = banner.with_duration(replay.duration) if hasattr(banner, "with_duration") else banner.set_duration(replay.duration)
                        banner = banner.with_position((int(clip.w) - bw - 12, 12)) if hasattr(banner, "with_position") else banner.set_position((int(clip.w) - bw - 12, 12))
                        banner = banner.with_opacity(0.85) if hasattr(banner, "with_opacity") else banner.set_opacity(0.85)
                        replay = CompositeVideoClip([replay, banner])
                    except Exception as exc:
                        LOGGER.debug("replay banner skipped: %s", exc)
                    if replay.audio is not None:
                        replay = replay.without_audio() if hasattr(replay, "without_audio") else replay.set_audio(None)
                    segments.append(_crossfadein(replay, cfg.crossfade_s))

        if not segments:
            return None
        segments[-1] = _fadeout(segments[-1], cfg.fade_out_s)
        reel = concatenate_videoclips(segments, method="compose", padding=-cfg.crossfade_s)
        reel.write_videofile(str(output_path), codec="libx264", audio_codec="aac", logger=None)
        reel.close()
        LOGGER.info("broadcast reel written: %s (%d segments)", output_path, len(segments))
        return str(output_path)
    finally:
        for clip in opened:
            try:
                clip.close()
            except Exception:
                pass
