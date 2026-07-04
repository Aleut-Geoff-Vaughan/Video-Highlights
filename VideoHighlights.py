"""
Soccer Highlight Agent
----------------------

A no-subscription, local Python pipeline that:
  • Tracks all players with YOLO + ByteTrack (GPU-accelerated with CUDA)
  • Lets you lock onto your child once (interactive box selection)
  • Detects highlight moments from speed/acceleration spikes and audio peaks
  • Exports clean subclips and an optional overlay version with a spotlight circle
  • Supports trimming long videos to focus on specific time ranges
  • Multithreaded clip generation for faster processing

Dependencies (install):
    pip install -r requirements.txt
    # Or manually: pip install ultralytics==8.* opencv-python numpy tqdm moviepy librosa soundfile torch

Performance:
  • Automatically uses CUDA GPU if available (2-3x faster inference)
  • Parallel clip writing with configurable thread count (--threads)
  • Half-precision (FP16) inference on GPU for maximum speed

Usage examples:
    # Basic usage
    python VideoHighlights.py --video /path/to/match.mp4 --out ./highlights_out

    # With player selection and overlay
    python VideoHighlights.py --video match.mp4 --select --overlay

    # Trim long video (2nd half only - 45 min to 90 min)
    python VideoHighlights.py --video match.mp4 --trim-start 45:00 --trim-end 1:30:00

    # Faster processing with custom thread count
    python VideoHighlights.py --video match.mp4 --threads 8

    # Interactive mode (prompts for all options)
    python VideoHighlights.py

Notes:
  • --select opens a window on the FIRST frame so you can drag a box over your child. Press ENTER/SPACE to confirm.
  • If you skip --select, the script picks the longest-lived person track (works surprisingly well when your child plays full-time).
  • --trim-start and --trim-end accept formats: seconds (e.g., 120), MM:SS (e.g., 2:00), or HH:MM:SS (e.g., 1:30:00)
  • --threads controls parallel clip writing (default: auto, max 4). Higher values = faster but more memory.
  • Trimming creates a temporary video for processing, but final clips come from the original video
  • First run will auto-download YOLO weights (~6MB).
  • Works best with 1080p/60 or 4K/60 videos recorded from a stable, elevated sideline or halfway-line vantage.
  • GPU acceleration requires PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
"""

import os
import sys
import math
import argparse
import csv
import json
import logging
import subprocess
import threading
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import numpy as np
import cv2
from tqdm import tqdm

from backend.services.follow_cam import ball_weight_for_mode, render_follow_cam_clip
from backend.services.player_focus import (
    choose_target_track_id,
    resolve_player_roi_box,
    stitch_target_track,
)
from backend.services.game_tracking import (
    FieldGeometry,
    analyze_game_states,
    build_ball_track,
    detect_goal_events,
    detect_set_pieces,
    estimate_field_geometry,
    overlay_set_piece_states,
    summarize_states,
)
from backend.services.card_detection import detect_card_events, stopped_play_windows
from backend.services.camera_planner import CameraPlan, plan_camera, slice_plan
from backend.services.camera_render import render_camera_plan_video

LOGGER = logging.getLogger("videohighlights")

_LOGGING_LOCK = threading.Lock()


class _CurrentStdout:
    """File-like proxy that always writes to the *current* sys.stdout.

    The GUI redirects sys.stdout to a fresh StringIO per run; binding the
    console handler to this proxy (instead of a snapshot of sys.stdout)
    keeps log output flowing to whatever stdout is active.
    """

    def write(self, text: str) -> int:
        return sys.stdout.write(text)

    def flush(self) -> None:
        try:
            sys.stdout.flush()
        except Exception:
            pass


def setup_logging(debug: bool = False, log_file: Optional[str] = None) -> Optional[logging.Handler]:
    """Configure pipeline logging for one run.

    ``debug=True`` prints every debug-level diagnostic to the console.
    ``log_file`` additionally captures the full DEBUG stream (with
    timestamps) regardless of the console level - useful for reviewing a run
    and for building training datasets.

    The console handler is created once and only its level is adjusted, and
    the per-run file handler is RETURNED so the caller can detach it when the
    run finishes (see :func:`teardown_run_logging`). Never removes handlers
    it did not create - concurrent jobs in one process (API worker threads)
    must not strip each other's log files mid-run.
    """
    with _LOGGING_LOCK:
        root = logging.getLogger("videohighlights")
        root.setLevel(logging.DEBUG)
        root.propagate = False

        console = next(
            (h for h in root.handlers if getattr(h, "_vh_console", False)), None
        )
        if console is None:
            console = logging.StreamHandler(_CurrentStdout())
            console._vh_console = True  # type: ignore[attr-defined]
            console.setFormatter(logging.Formatter("[%(levelname).1s] %(message)s"))
            root.addHandler(console)
        console.setLevel(logging.DEBUG if debug else logging.INFO)

        file_handler: Optional[logging.Handler] = None
        if log_file:
            log_dir = os.path.dirname(os.path.abspath(log_file))
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
            )
            root.addHandler(file_handler)
        return file_handler


def teardown_run_logging(file_handler: Optional[logging.Handler]) -> None:
    """Detach and close a run's file handler (prevents fd leaks and, on
    Windows, lingering locks on the output directory)."""
    if file_handler is None:
        return
    with _LOGGING_LOCK:
        root = logging.getLogger("videohighlights")
        root.removeHandler(file_handler)
        try:
            file_handler.close()
        except Exception:
            pass


# --- Optional heavy dependencies (loaded lazily so this module can be
# imported by the API/tests without ultralytics/librosa/moviepy installed) ---


def _import_yolo():
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for tracking. Install with: pip install ultralytics"
        ) from exc
    return YOLO


def _import_moviepy():
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips  # type: ignore
    except ImportError:
        try:
            # moviepy 2.x has a different import structure
            from moviepy.video.io.VideoFileClip import VideoFileClip  # type: ignore
            from moviepy.video.compositing.CompositeVideoClip import (  # type: ignore
                concatenate_videoclips,
            )
        except ImportError as exc:
            raise RuntimeError(
                "moviepy is required for clip export. Install with: pip install moviepy"
            ) from exc
    return VideoFileClip, concatenate_videoclips


@dataclass
class TrackPoint:
    t: float  # seconds
    xy: Tuple[float, float]  # center x,y in pixels
    bbox: Optional[Tuple[float, float, float, float]] = None


FOLLOW_CAM_MODES = {"wide", "follow_player", "follow_action", "follow_ball"}
ProgressCallback = Callable[[str, float, str, Optional[Dict[str, object]]], None]


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def emit_progress(
    progress_callback: Optional[ProgressCallback],
    stage: str,
    progress: float,
    message: str,
    data: Optional[Dict[str, object]] = None,
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(stage, max(0.0, min(1.0, float(progress))), message, data or {})
    except Exception as exc:
        print(f"[warn] Progress callback failed: {exc}")


def parse_time(time_str: str) -> float:
    """Parse time string to seconds. Supports formats: seconds (123), MM:SS (12:30), HH:MM:SS (1:23:45)"""
    if not time_str:
        return 0.0

    time_str = time_str.strip()

    # Try parsing as plain seconds first
    try:
        return float(time_str)
    except ValueError:
        pass

    # Parse as time format (MM:SS or HH:MM:SS)
    parts = time_str.split(':')
    if len(parts) == 2:  # MM:SS
        try:
            minutes, seconds = map(float, parts)
            return minutes * 60 + seconds
        except ValueError:
            raise ValueError(f"Invalid time format: {time_str}. Use MM:SS, HH:MM:SS, or seconds")
    elif len(parts) == 3:  # HH:MM:SS
        try:
            hours, minutes, seconds = map(float, parts)
            return hours * 3600 + minutes * 60 + seconds
        except ValueError:
            raise ValueError(f"Invalid time format: {time_str}. Use MM:SS, HH:MM:SS, or seconds")
    else:
        raise ValueError(f"Invalid time format: {time_str}. Use MM:SS, HH:MM:SS, or seconds")


def format_time(seconds: float) -> str:
    """Format seconds to HH:MM:SS string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def create_trimmed_video(video_path: str, out_dir: str, start_time: Optional[float] = None, end_time: Optional[float] = None) -> Tuple[str, float]:
    """
    Create a trimmed version of the video for processing.
    Returns: (trimmed_video_path, trim_offset_seconds)
    If no trimming needed, returns original path with 0 offset.
    """
    if start_time is None and end_time is None:
        return video_path, 0.0

    # Get video duration
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps if total_frames and fps else 0.0
    cap.release()

    start_time = start_time or 0.0
    end_time = end_time or duration

    # Validate times
    if start_time < 0:
        start_time = 0.0
    if end_time > duration:
        end_time = duration
    if start_time >= end_time:
        raise ValueError(f"Invalid trim times: start ({format_time(start_time)}) must be before end ({format_time(end_time)})")

    print(f"\n[trim] Creating trimmed video from {format_time(start_time)} to {format_time(end_time)} (duration: {format_time(end_time - start_time)})")

    # Create trimmed video
    ensure_dir(out_dir)
    trimmed_path = os.path.join(out_dir, "trimmed_working_video.mp4")

    # Fast path: ffmpeg re-encode (input seeking + veryfast preset) is
    # dramatically faster than the moviepy fallback on long recordings while
    # staying frame-accurate.
    try:
        from backend.services.ffmpeg_tools import ffmpeg_available, ffmpeg_exe

        if ffmpeg_available():
            cmd = [
                ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start_time:.3f}", "-to", f"{end_time:.3f}",
                "-i", video_path,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", "-movflags", "+faststart",
                trimmed_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0 and os.path.exists(trimmed_path) and os.path.getsize(trimmed_path) > 0:
                print(f"[trim] Trimmed video saved to: {trimmed_path} (ffmpeg)")
                return trimmed_path, start_time
            LOGGER.warning("ffmpeg trim failed (%s); falling back to moviepy", result.stderr.strip()[:300])
    except Exception as exc:
        LOGGER.warning("ffmpeg trim unavailable (%s); falling back to moviepy", exc)

    try:
        VideoFileClip, _ = _import_moviepy()
        with VideoFileClip(video_path) as clip:
            # Try both subclip and subclipped (different moviepy versions)
            try:
                trimmed_clip = clip.subclip(start_time, end_time)
            except AttributeError:
                trimmed_clip = clip.subclipped(start_time, end_time)

            trimmed_clip.write_videofile(trimmed_path, codec="libx264", audio_codec="aac")
            trimmed_clip.close()
        print(f"[trim] Trimmed video saved to: {trimmed_path}")
        return trimmed_path, start_time
    except Exception as e:
        raise RuntimeError(f"Failed to create trimmed video: {e}")


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two [x1,y1,x2,y2]."""
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    areaB = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = areaA + areaB - inter
    return float(inter / union) if union > 0 else 0.0


def robust_threshold(series: np.ndarray, k: float = 3.0) -> float:
    """Median + k * MAD as a robust outlier/highlight threshold."""
    if len(series) == 0:
        return float('inf')
    med = np.median(series)
    mad = np.median(np.abs(series - med)) + 1e-9
    return med + k * mad


def merge_intervals(intervals: List[Tuple[float, float]], min_gap: float = 0.75) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s - last_e <= min_gap:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def _interval_overlap_seconds(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _pick_event_type(
    has_speed_signal: bool,
    has_audio_signal: bool,
    requested_targets: List[str],
) -> str:
    allowed_types = {
        "goal",
        "shot",
        "corner_kick",
        "penalty_kick",
        "free_kick",
        "goal_kick",
        "kickoff",
        "foul",
        "save",
    }
    targets = [item for item in requested_targets if item in allowed_types]
    if len(set(targets)) == 1:
        return targets[0]

    if has_speed_signal and has_audio_signal:
        for candidate in ("goal", "shot", "penalty_kick", "save"):
            if candidate in targets:
                return candidate
        return "goal"

    if has_speed_signal:
        for candidate in ("shot", "goal", "penalty_kick", "corner_kick", "free_kick", "save"):
            if candidate in targets:
                return candidate
        return "shot"

    if has_audio_signal:
        for candidate in ("foul", "kickoff", "goal", "corner_kick"):
            if candidate in targets:
                return candidate
        return "foul"

    if targets:
        return targets[0]
    return "shot"


def build_analysis_bookmarks(
    original_intervals: List[Tuple[float, float]],
    speed_intervals: List[Tuple[float, float]],
    audio_intervals: List[Tuple[float, float]],
    requested_targets: List[str],
    live_manifest_path: Optional[str] = None,
    live_manifest_context: Optional[Dict[str, object]] = None,
    goal_events: Optional[List[Dict[str, object]]] = None,
    game_states: Optional[List[Dict[str, object]]] = None,
    card_events: Optional[List[Dict[str, object]]] = None,
    set_piece_events: Optional[List[Dict[str, object]]] = None,
) -> List[Dict[str, object]]:
    """Build the bookmark table for a run.

    ``goal_events`` (dicts with ``t``/``side``/``confidence``/``reason`` in the
    same timebase as ``original_intervals``) upgrade any overlapping interval
    to a confirmed ``goal`` bookmark. ``game_states`` (dicts with
    ``start_s``/``end_s``/``state``) tag each bookmark with the game state at
    its center so tags explain what the game was doing.
    """
    bookmarks: List[Dict[str, object]] = []
    context = dict(live_manifest_context or {})
    goal_rows = list(goal_events or [])
    state_rows = list(game_states or [])
    card_rows = list(card_events or [])
    set_piece_rows = list(set_piece_events or [])

    def _row_within(rows: List[Dict[str, object]], key: str, start_s: float, end_s: float) -> Optional[Dict[str, object]]:
        for row in rows:
            if start_s <= float(row.get(key, -1.0)) <= end_s:
                return row
        return None

    def _state_label_at(t: float) -> Optional[str]:
        for row in state_rows:
            if float(row.get("start_s", 0.0)) <= t < float(row.get("end_s", 0.0)):
                return str(row.get("state"))
        return None

    def _goal_within(start_s: float, end_s: float) -> Optional[Dict[str, object]]:
        for row in goal_rows:
            if start_s <= float(row.get("t", -1.0)) <= end_s:
                return row
        return None

    def _write_live_manifest() -> None:
        if not live_manifest_path:
            return
        payload = dict(context)
        payload["bookmarks"] = list(bookmarks)
        payload["stats"] = dict(payload.get("stats", {}))
        payload["stats"]["bookmark_count"] = len(bookmarks)
        try:
            with open(live_manifest_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception:
            pass

    _write_live_manifest()
    for idx, (start_s, end_s) in enumerate(original_intervals, start=1):
        speed_overlap = sum(_interval_overlap_seconds((start_s, end_s), interval) for interval in speed_intervals)
        audio_overlap = sum(_interval_overlap_seconds((start_s, end_s), interval) for interval in audio_intervals)
        has_speed_signal = speed_overlap > 0.0
        has_audio_signal = audio_overlap > 0.0

        event_type = _pick_event_type(has_speed_signal, has_audio_signal, requested_targets)
        confidence = 0.45
        if has_speed_signal:
            confidence += 0.22
        if has_audio_signal:
            confidence += 0.18
        if requested_targets and event_type in requested_targets:
            confidence += 0.08
        confidence = float(min(0.99, round(confidence, 3)))

        sources: List[str] = []
        if has_speed_signal:
            sources.append("motion")
        if has_audio_signal:
            sources.append("audio")
        if not sources:
            sources.append("motion")

        occurred_at_s = (start_s + end_s) / 2.0
        duration_s = max(0.0, end_s - start_s)

        label = f"{event_type}_candidate"
        signals: Dict[str, object] = {
            "speed_overlap_s": round(speed_overlap, 3),
            "audio_overlap_s": round(audio_overlap, 3),
        }

        # Ball-tracking goal detection overrides the heuristic event type:
        # a flagged goal inside this interval makes it a goal bookmark. A
        # goal-only interval (no motion/audio overlap) must not carry the
        # default "motion" source label - its evidence is the ball track.
        goal_row = _goal_within(start_s, end_s)
        if goal_row is not None and not has_speed_signal and not has_audio_signal:
            sources = []
        if goal_row is not None:
            event_type = "goal"
            label = "goal_detected"
            occurred_at_s = float(goal_row.get("t", occurred_at_s))
            confidence = float(min(0.99, max(confidence, float(goal_row.get("confidence", 0.0)))))
            if "ball_tracking" not in sources:
                sources.append("ball_tracking")
            signals["goal_side"] = goal_row.get("side")
            signals["goal_reason"] = goal_row.get("reason")
            if has_audio_signal:
                # Crowd noise corroborating a detected goal.
                confidence = float(min(0.99, confidence + 0.05))

        # Referee cards outrank set pieces; both defer to detected goals.
        if goal_row is None:
            card_row = _row_within(card_rows, "t", start_s, end_s)
            if card_row is not None:
                event_type = str(card_row.get("kind") or "yellow_card")
                label = f"{event_type}_detected"
                occurred_at_s = float(card_row.get("t", occurred_at_s))
                confidence = float(min(0.99, max(confidence, float(card_row.get("confidence", 0.0)))))
                if not has_speed_signal and not has_audio_signal:
                    sources = []
                if "vision" not in sources:
                    sources.append("vision")
                signals["card_reason"] = card_row.get("reason")
                if card_row.get("crop_path"):
                    signals["card_crop_path"] = card_row.get("crop_path")
            else:
                sp_row = _row_within(set_piece_rows, "t_kick", start_s, end_s)
                if sp_row is not None and sp_row.get("kind") in {
                    "corner_kick", "free_kick", "penalty_kick", "goal_kick", "kickoff",
                }:
                    event_type = str(sp_row["kind"])
                    label = f"{event_type}_detected"
                    occurred_at_s = float(sp_row.get("t_kick", occurred_at_s))
                    confidence = float(min(0.99, max(confidence, 0.8)))
                    if not has_speed_signal and not has_audio_signal:
                        sources = []
                    if "ball_tracking" not in sources:
                        sources.append("ball_tracking")
                    signals["set_piece_side"] = sp_row.get("side")
                    signals["set_piece_reason"] = sp_row.get("reason")

        if not sources:
            sources.append("motion")

        game_state = _state_label_at(occurred_at_s)

        bookmarks.append(
            {
                "bookmark_id": f"bm_{idx:04d}",
                "index": idx,
                "event_type": event_type,
                "label": label,
                "confidence": confidence,
                "start_s": round(start_s, 3),
                "occurred_at_s": round(occurred_at_s, 3),
                "end_s": round(end_s, 3),
                "duration_s": round(duration_s, 3),
                "sources": sources,
                "game_state": game_state,
                "signals": signals,
            }
        )
        _write_live_manifest()
    return bookmarks


def write_analysis_bookmark_files(
    output_dir: str,
    manifest: Dict[str, object],
) -> Tuple[str, str]:
    json_path = os.path.join(output_dir, "analysis_bookmarks.json")
    csv_path = os.path.join(output_dir, "analysis_bookmarks.csv")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    bookmarks = list(manifest.get("bookmarks", []) or [])
    field_names = [
        "bookmark_id",
        "index",
        "event_type",
        "label",
        "confidence",
        "start_s",
        "occurred_at_s",
        "end_s",
        "duration_s",
        "sources",
        "game_state",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for item in bookmarks:
            row = {name: item.get(name) for name in field_names}
            sources = row.get("sources", [])
            if isinstance(sources, list):
                row["sources"] = ",".join(str(source) for source in sources)
            writer.writerow(row)

    return json_path, csv_path


def write_tracking_manifest(
    output_dir: str,
    payload: Dict[str, object],
) -> str:
    tracking_path = os.path.join(output_dir, "analysis_tracking.json")
    with open(tracking_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return tracking_path


def track_video(
    video_path: str,
    fps_hint: Optional[float] = None,
    select_roi: bool = False,
    player_roi: Optional[Dict[str, float]] = None,
    yolo_model: str = "yolo26s.pt",
    tracker_config: str = "botsort.yaml",
    inference_imgsz: int = 960,
    detection_conf: float = 0.18,
    vid_stride: int = 1,
    progress_callback: Optional[ProgressCallback] = None,
    progress_start: float = 0.20,
    progress_end: float = 0.65,
) -> Tuple[Dict[int, List[TrackPoint]], List[TrackPoint], float, Tuple[int, int], Dict[str, object]]:
    """Run YOLO + ByteTrack, return per-ID trajectory, ball trajectory, FPS, and frame size.
    Returns:
        (
            tracks,
            ball_trajectory,
            fps,
            (W,H),
            selection_metadata,
        )
        where tracks[id] = [TrackPoint, ...], ball_trajectory = [TrackPoint, ...]
    """
    # Prepare first frame (for selection)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = fps_hint or cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, first = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read first frame.")
    H, W = first.shape[:2]

    user_box = None  # [x1,y1,x2,y2]
    if player_roi:
        resolved_box = resolve_player_roi_box(player_roi, W, H)
        if resolved_box is not None:
            user_box = np.array(resolved_box, dtype=np.float32)
    elif select_roi:
        # OpenCV ROI returns (x,y,w,h)
        roi = cv2.selectROI("Select your player", first, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select your player")
        x, y, w, h = roi
        if w > 0 and h > 0:
            user_box = np.array([x, y, x + w, y + h], dtype=np.float32)

    # YOLO tracking (persons + sports ball for potential future use)
    model_name = str(yolo_model or "yolo26s.pt").strip()
    tracker_name = str(tracker_config or "botsort.yaml").strip()
    imgsz = max(320, min(1920, int(inference_imgsz or 960)))
    conf = min(0.9, max(0.01, float(detection_conf or 0.18)))
    stride = max(1, int(vid_stride or 1))
    YOLO = _import_yolo()
    model = YOLO(model_name)

    # Enable GPU if available and use half precision for faster inference
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[performance] Using device: {device}")
    if device == 'cuda':
        try:
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            model.to(device)
        except Exception:
            pass
        print(f"[performance] GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"[performance] YOLO runtime: model={model_name}, tracker={tracker_name}, "
        f"imgsz={imgsz}, conf={conf:.2f}, vid_stride={stride}, half={device == 'cuda'}"
    )
    total_tracking_frames = int(math.ceil(total_frames / stride)) if total_frames > 0 else 0
    emit_progress(
        progress_callback,
        "tracking",
        progress_start,
        "YOLO tracking started",
        {
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "yolo_model": model_name,
            "tracker_config": tracker_name,
            "inference_imgsz": imgsz,
            "detection_conf": conf,
            "vid_stride": stride,
            "total_frames": total_frames,
            "estimated_tracking_frames": total_tracking_frames,
        },
    )

    # The stream=True iterator yields per-frame results with .boxes and .boxes.id
    tracks: Dict[int, List[TrackPoint]] = {}
    ball_trajectory: List[TrackPoint] = []
    # All-player positions feed field-geometry percentiles and 0.5s-binned
    # action centroids; both are insensitive to subsampling, so cap the
    # collection rate at ~10Hz (an hour of 60fps x 20 players would otherwise
    # hold millions of tuples in memory for the whole run).
    all_player_positions: List[Tuple[float, float, float]] = []
    player_pos_keep_every = max(1, int(round((fps / max(1, stride)) / 10.0)))
    last_progress_emit = 0.0

    for frame_idx, result in enumerate(
        model.track(
            source=video_path,
            stream=True,
            tracker=tracker_name,
            classes=[0, 32],
            device=device,
            half=True if device == 'cuda' else False,
            imgsz=imgsz,
            conf=conf,
            vid_stride=stride,
            persist=True,
            verbose=False,
        )
    ):
        source_frame_idx = frame_idx * stride
        if progress_callback is not None and total_tracking_frames > 0:
            now = time.monotonic()
            is_first = frame_idx == 0
            is_last = frame_idx + 1 >= total_tracking_frames
            if is_first or is_last or (now - last_progress_emit) >= 2.0:
                fraction = min(1.0, max(0.0, float(frame_idx + 1) / float(total_tracking_frames)))
                emit_progress(
                    progress_callback,
                    "tracking",
                    progress_start + (progress_end - progress_start) * fraction,
                    "YOLO tracking frames",
                    {
                        "processed_tracking_frames": frame_idx + 1,
                        "estimated_tracking_frames": total_tracking_frames,
                        "source_frame_index": source_frame_idx,
                        "total_frames": total_frames,
                        "device": device,
                    },
                )
                last_progress_emit = now
        # We care about persons (class 0) and sports ball (class 32). result.boxes.cls, .id, .xyxy
        if result.boxes is None or result.boxes.id is None:
            continue
        ids = result.boxes.id.cpu().numpy().astype(int)
        cls = result.boxes.cls.cpu().numpy().astype(int)
        xyxy = result.boxes.xyxy.cpu().numpy()

        t = source_frame_idx / fps
        for idx, (track_id, c) in enumerate(zip(ids, cls)):
            x1, y1, x2, y2 = xyxy[idx]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            if c == 0:  # person
                if frame_idx % player_pos_keep_every == 0:
                    all_player_positions.append((t, float(cx), float(cy)))
                tracks.setdefault(track_id, []).append(
                    TrackPoint(
                        t=t,
                        xy=(float(cx), float(cy)),
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                    )
                )
            elif c == 32:  # sports ball
                ball_trajectory.append(
                    TrackPoint(
                        t=t,
                        xy=(float(cx), float(cy)),
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                    )
                )

    # Choose target ID
    emit_progress(
        progress_callback,
        "tracking",
        progress_end,
        "YOLO tracking complete",
        {
            "person_track_count": len(tracks),
            "ball_detection_count": len(ball_trajectory),
            "device": device,
        },
    )
    target_id = None

    if not tracks:
        raise RuntimeError("No player tracks detected in video. Ensure the video contains visible people.")

    if user_box is not None:
        # Pick ID whose early detections overlap the selected ROI best within the opening seconds.
        target_id = choose_target_track_id(tracks, tuple(float(v) for v in user_box), window_t=3.0)

        if target_id is None:
            # Fallback if no tracks match the user selection
            print("[warn] No tracks matched your selection. Using longest-lived track instead.")
            target_id = max(tracks.keys(), key=lambda k: (tracks[k][-1].t - tracks[k][0].t))
    else:
        # default: longest-lived track
        target_id = max(tracks.keys(), key=lambda k: (tracks[k][-1].t - tracks[k][0].t))

    if target_id is None:
        raise RuntimeError("No player track found. Try using --select on the first frame.")

    stitched_track_ids, stitched_traj = stitch_target_track(tracks, int(target_id))
    if not stitched_traj:
        stitched_track_ids = [int(target_id)]
        stitched_traj = list(tracks[int(target_id)])
    if len(stitched_track_ids) > 1:
        print(
            "[info] Re-identified selected player across tracker IDs: "
            + " -> ".join(str(track_id) for track_id in stitched_track_ids)
        )

    print(f"[info] Tracked {len(ball_trajectory)} ball detections across video")
    return (
        {int(target_id): stitched_traj},
        ball_trajectory,
        fps,
        (W, H),
        {
            "target_track_id": int(target_id),
            "stitched_track_ids": stitched_track_ids,
            "stitched_track_count": len(stitched_track_ids),
            "player_positions": (
                np.asarray(all_player_positions, dtype=np.float32)
                if all_player_positions
                else np.empty((0, 3), dtype=np.float32)
            ),
        },
    )


def compute_speed_series(traj: List[TrackPoint], fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return times and speed (pixels/sec) for the trajectory."""
    if len(traj) < 2:
        return np.array([]), np.array([])
    times = np.array([p.t for p in traj])
    centers = np.array([p.xy for p in traj])
    dt = np.diff(times)
    dist = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    # Guard against zeros
    dt = np.where(dt <= 1e-6, 1e-6, dt)
    speed = dist / dt  # pixels/sec because t is in seconds
    # Align speeds to the right time index (skip first timestamp)
    return times[1:], speed


def compute_direction_changes(traj: List[TrackPoint], fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute direction change magnitude for trajectory.
    Detects sudden changes in movement direction (cuts, turns, stops).

    Returns:
        times: Array of timestamps
        direction_changes: Array of direction change magnitudes (radians)
    """
    if len(traj) < 3:
        return np.array([]), np.array([])

    times = np.array([p.t for p in traj])
    centers = np.array([p.xy for p in traj])

    # Compute velocity vectors
    velocity = np.diff(centers, axis=0)  # [dx, dy] between consecutive points

    if len(velocity) < 2:
        return np.array([]), np.array([])

    # Compute angle between consecutive velocity vectors
    direction_changes = []
    for i in range(len(velocity) - 1):
        v1 = velocity[i]
        v2 = velocity[i + 1]

        # Handle zero velocity (stopped)
        v1_mag = np.linalg.norm(v1)
        v2_mag = np.linalg.norm(v2)

        if v1_mag < 1e-6 or v2_mag < 1e-6:
            direction_changes.append(0.0)
            continue

        # Compute angle between vectors using dot product
        cos_angle = np.dot(v1, v2) / (v1_mag * v2_mag)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical stability
        angle_rad = np.arccos(cos_angle)

        direction_changes.append(angle_rad)

    direction_changes = np.array(direction_changes)

    # Align with times (skip first two timestamps due to diff operations)
    aligned_times = times[2:]

    LOGGER.debug(f"Direction changes: {len(direction_changes)} values, max={np.rad2deg(direction_changes.max()):.1f}°, mean={np.rad2deg(direction_changes.mean()):.1f}°")

    return aligned_times, direction_changes


def compute_ball_proximity_score(player_traj: List[TrackPoint], ball_traj: List[TrackPoint], proximity_threshold: float = 200.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ball proximity score for each player trajectory point.
    Returns (times, proximity_scores) where score is 1/distance (higher = closer to ball)

    Args:
        player_traj: Player trajectory points
        ball_traj: Ball trajectory points
        proximity_threshold: Distance in pixels considered "close" (default: 200px for kids soccer from elevation)

    Returns:
        times: Array of timestamps
        proximity_scores: Array of proximity scores (higher = player closer to ball)
    """
    if len(player_traj) == 0 or len(ball_traj) == 0:
        LOGGER.debug("No ball or player trajectory for proximity scoring")
        return np.array([]), np.array([])

    player_times = np.array([p.t for p in player_traj])
    player_centers = np.array([p.xy for p in player_traj])

    ball_times = np.array([b.t for b in ball_traj])
    ball_centers = np.array([b.xy for b in ball_traj])

    # For each player time point, find nearest ball detection in time
    proximity_scores = []
    for i, pt in enumerate(player_times):
        # Find ball detections within +/- 0.5 seconds
        time_window = 0.5
        nearby_ball_indices = np.where(np.abs(ball_times - pt) <= time_window)[0]

        if len(nearby_ball_indices) == 0:
            proximity_scores.append(0.0)  # No ball detected nearby in time
            continue

        # Calculate distance to nearest ball
        player_pos = player_centers[i]
        min_distance = float('inf')
        for ball_idx in nearby_ball_indices:
            ball_pos = ball_centers[ball_idx]
            distance = np.linalg.norm(player_pos - ball_pos)
            min_distance = min(min_distance, distance)

        # Convert distance to proximity score (inverse relationship)
        # Use sigmoid-like function: score is high when distance < threshold
        if min_distance < proximity_threshold:
            score = (proximity_threshold - min_distance) / proximity_threshold
        else:
            score = 0.0

        proximity_scores.append(score)

    proximity_scores = np.array(proximity_scores)
    LOGGER.debug(f"Ball proximity: {len(proximity_scores)} scores, max={proximity_scores.max():.3f}, mean={proximity_scores.mean():.3f}")

    return player_times, proximity_scores


def detect_highlights_from_speed(times: np.ndarray, speed: np.ndarray, pre: float, post: float, k: float = 2.0) -> List[Tuple[float, float]]:
    if len(speed) == 0:
        return []
    thr = robust_threshold(speed, k=k)
    candidates = np.where(speed >= thr)[0]
    LOGGER.debug(f"Speed threshold: {thr:.2f} (k={k}), found {len(candidates)} speed peaks")
    if len(candidates) > 0:
        LOGGER.debug(f"Speed range: min={speed.min():.2f}, max={speed.max():.2f}, median={np.median(speed):.2f}")
    intervals = []
    for idx in candidates:
        t = float(times[idx])
        intervals.append((max(0.0, t - pre), t + post))
    merged = merge_intervals(intervals)
    LOGGER.debug(f"Speed intervals before merge: {len(intervals)}, after merge: {len(merged)}")
    return merged


def detect_audio_peaks(video_path: str, pre: float, post: float, k: float = 2.0) -> List[Tuple[float, float]]:
    try:
        try:
            import librosa  # Lazy: audio analysis is optional at runtime
        except ImportError:
            print(
                "[warn] librosa is not installed - audio peak detection disabled. "
                "Install with: pip install librosa soundfile"
            )
            return []

        # Load audio at native sampling rate
        y, sr = librosa.load(video_path, sr=None, mono=True)
        # Frame over ~100 ms windows
        hop = int(0.05 * sr)
        win = int(0.10 * sr)
        rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=True).flatten()
        # Map frames to times
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop, n_fft=win)
        thr = robust_threshold(rms, k=k)
        peaks = np.where(rms >= thr)[0]
        LOGGER.debug(f"Audio threshold: {thr:.4f} (k={k}), found {len(peaks)} audio peaks")
        if len(peaks) > 0:
            LOGGER.debug(f"Audio RMS range: min={rms.min():.4f}, max={rms.max():.4f}, median={np.median(rms):.4f}")
        intervals = [(max(0.0, float(times[i]) - pre), float(times[i]) + post) for i in peaks]
        merged = merge_intervals(intervals)
        LOGGER.debug(f"Audio intervals before merge: {len(intervals)}, after merge: {len(merged)}")
        return merged
    except Exception as e:
        print(f"[warn] audio peak detection failed: {e}")
        return []


def detect_highlights_multi_factor(
    times: np.ndarray,
    speed: np.ndarray,
    ball_proximity: np.ndarray,
    direction_changes: np.ndarray,
    pre: float,
    post: float,
    speed_weight: float = 0.5,
    proximity_weight: float = 0.3,
    direction_weight: float = 0.2,
    k: float = 2.0
) -> List[Tuple[float, float]]:
    """Detect highlights using multi-factor scoring: speed + ball proximity + direction changes.

    Args:
        times: Timestamp array
        speed: Speed values (pixels/sec)
        ball_proximity: Ball proximity scores (0-1)
        direction_changes: Direction change magnitudes (radians)
        pre: Seconds before event
        post: Seconds after event
        speed_weight: Weight for speed factor (default: 0.5)
        proximity_weight: Weight for proximity factor (default: 0.3)
        direction_weight: Weight for direction change factor (default: 0.2)
        k: Threshold multiplier for MAD (default: 2.0)

    Returns:
        List of (start, end) intervals
    """
    if len(speed) == 0:
        LOGGER.debug("No speed data for multi-factor detection")
        return []

    # Normalize speed to 0-1 range
    speed_norm = (speed - speed.min()) / (speed.max() - speed.min() + 1e-9)

    # Ensure ball_proximity aligns with speed times
    if len(ball_proximity) == 0:
        LOGGER.debug("No ball proximity data, using zero weight")
        proximity_norm = np.zeros_like(speed_norm)
    else:
        # ball_proximity should already be 0-1 range
        proximity_norm = ball_proximity
        # If lengths don't match, interpolate or pad
        if len(proximity_norm) != len(speed_norm):
            LOGGER.debug(f"Length mismatch: speed={len(speed_norm)}, proximity={len(proximity_norm)}")
            # Truncate or pad to match
            if len(proximity_norm) > len(speed_norm):
                proximity_norm = proximity_norm[:len(speed_norm)]
            else:
                proximity_norm = np.pad(proximity_norm, (0, len(speed_norm) - len(proximity_norm)), 'constant')

    # Ensure direction_changes aligns with speed times
    if len(direction_changes) == 0:
        LOGGER.debug("No direction change data, using zero weight")
        direction_norm = np.zeros_like(speed_norm)
    else:
        # Normalize direction changes (0 = no change, pi = complete reversal)
        direction_norm = direction_changes / (np.pi + 1e-9)  # Normalize to 0-1
        # If lengths don't match, pad or truncate
        if len(direction_norm) != len(speed_norm):
            LOGGER.debug(f"Length mismatch: speed={len(speed_norm)}, direction={len(direction_norm)}")
            if len(direction_norm) > len(speed_norm):
                direction_norm = direction_norm[:len(speed_norm)]
            else:
                direction_norm = np.pad(direction_norm, (0, len(speed_norm) - len(direction_norm)), 'constant')

    # Compute combined score
    combined_score = (speed_weight * speed_norm) + (proximity_weight * proximity_norm) + (direction_weight * direction_norm)

    LOGGER.debug(f"Multi-factor score: min={combined_score.min():.3f}, max={combined_score.max():.3f}, mean={combined_score.mean():.3f}")
    LOGGER.debug(f"Weights: speed={speed_weight}, proximity={proximity_weight}, direction={direction_weight}")

    # Apply robust threshold to combined score
    thr = robust_threshold(combined_score, k=k)
    candidates = np.where(combined_score >= thr)[0]

    LOGGER.debug(f"Multi-factor threshold: {thr:.3f} (k={k}), found {len(candidates)} highlights")

    intervals = []
    for idx in candidates:
        t = float(times[idx])
        intervals.append((max(0.0, t - pre), t + post))

    merged = merge_intervals(intervals)
    LOGGER.debug(f"Multi-factor intervals before merge: {len(intervals)}, after merge: {len(merged)}")

    return merged


def detect_review_candidate_intervals(
    times: np.ndarray,
    speed: np.ndarray,
    direction_changes: np.ndarray,
    pre: float,
    post: float,
    max_candidates: int = 3,
) -> List[Tuple[float, float]]:
    if len(times) == 0 or len(speed) == 0:
        return []

    speed_norm = (speed - speed.min()) / (speed.max() - speed.min() + 1e-9)
    if len(direction_changes) == 0:
        direction_norm = np.zeros_like(speed_norm)
    else:
        direction_norm = direction_changes / (np.pi + 1e-9)
        if len(direction_norm) != len(speed_norm):
            if len(direction_norm) > len(speed_norm):
                direction_norm = direction_norm[:len(speed_norm)]
            else:
                direction_norm = np.pad(direction_norm, (0, len(speed_norm) - len(direction_norm)), "constant")

    score = (0.75 * speed_norm) + (0.25 * direction_norm)
    if not np.any(score > 0.0):
        return []

    order = np.argsort(score)[::-1]
    selected_times: List[float] = []
    min_separation = max(8.0, float(pre + post))
    for idx in order:
        t = float(times[int(idx)])
        if all(abs(t - existing) >= min_separation for existing in selected_times):
            selected_times.append(t)
        if len(selected_times) >= max(1, int(max_candidates)):
            break

    intervals = [(max(0.0, t - pre), t + post) for t in sorted(selected_times)]
    merged = merge_intervals(intervals, min_gap=1.0)
    LOGGER.debug(f"Review candidate fallback intervals: {len(merged)}")
    return merged


def check_nvenc_available() -> bool:
    """Check if NVENC GPU encoding is available"""
    try:
        import subprocess
        ffmpeg_binary = 'ffmpeg'
        try:
            from backend.services.ffmpeg_tools import ensure_ffmpeg_on_path, ffmpeg_exe

            ensure_ffmpeg_on_path()
            ffmpeg_binary = ffmpeg_exe()
        except Exception:
            pass
        result = subprocess.run([ffmpeg_binary, '-hide_banner', '-encoders'],
                              capture_output=True, text=True, timeout=5)
        output = f"{result.stdout}\n{result.stderr}".lower()
        # Require the encoder token in the ffmpeg encoder list.
        return " h264_nvenc" in output or "h264_nvenc " in output
    except Exception:
        return False

# Global flag for NVENC availability (checked once)
_NVENC_AVAILABLE = None

def write_single_subclip(video_path: str, interval: Tuple[float, float], clip_num: int, out_dir: str, use_gpu_encoding: bool = True) -> Optional[str]:
    """Write a single subclip (used for parallel processing)"""
    global _NVENC_AVAILABLE

    # Check NVENC availability once
    if _NVENC_AVAILABLE is None:
        _NVENC_AVAILABLE = check_nvenc_available() if use_gpu_encoding else False

    s, e = interval
    clip = None
    sub = None
    try:
        VideoFileClip, _ = _import_moviepy()
        clip = VideoFileClip(video_path)
        s = max(0.0, s)
        e = min(clip.duration, e)
        if e - s <= 0.25:
            return None

        # Try both subclip and subclipped (different moviepy versions)
        try:
            sub = clip.subclip(s, e)
        except AttributeError:
            sub = clip.subclipped(s, e)

        out_path = os.path.join(out_dir, f"highlight_{clip_num:02d}.mp4")

        # Build ordered fallback list. If NVENC fails at runtime, auto-fallback to libx264.
        encoding_attempts = []
        if _NVENC_AVAILABLE:
            encoding_attempts.append(("h264_nvenc", ['-preset', 'fast', '-b:v', '5M']))
        encoding_attempts.append(("libx264", ['-preset', 'faster', '-crf', '23']))
        encoding_attempts.append(("mpeg4", ['-q:v', '3']))

        last_error = None
        for codec, codec_params in encoding_attempts:
            try:
                try:
                    sub.write_videofile(
                        out_path,
                        codec=codec,
                        audio_codec="aac",
                        logger=None,
                        threads=2,
                        ffmpeg_params=codec_params,
                    )
                except (AttributeError, OSError) as audio_err:
                    print(f"[warn] Audio processing failed for clip {clip_num} with {codec}, retrying without audio: {audio_err}")
                    sub.write_videofile(
                        out_path,
                        codec=codec,
                        audio=False,
                        logger=None,
                        threads=2,
                        ffmpeg_params=codec_params,
                    )
                return out_path
            except Exception as codec_err:
                last_error = codec_err
                msg = str(codec_err).lower()
                print(f"[warn] Codec {codec} failed for clip {clip_num}: {codec_err}")

                # If NVENC failed, disable it globally for the remainder of the process.
                if codec == "h264_nvenc":
                    if "unknown encoder" in msg or "encoder not found" in msg or "error selecting an encoder" in msg:
                        _NVENC_AVAILABLE = False
                        print("[warn] NVENC unavailable at runtime. Falling back to CPU codec (libx264).")
                    elif "broken pipe" in msg:
                        _NVENC_AVAILABLE = False
                        print("[warn] NVENC pipeline unstable (broken pipe). Falling back to CPU codec (libx264).")
                continue

        if last_error is not None:
            raise last_error
        return None
    except Exception as ex:
        print(f"[warn] Failed to write clip {clip_num} ({s:.1f}s - {e:.1f}s): {ex}")
        return None
    finally:
        if sub is not None:
            sub.close()
        if clip is not None:
            clip.close()


def write_subclips(video_path: str, intervals: List[Tuple[float, float]], out_dir: str, max_workers: Optional[int] = None, use_gpu_encoding: bool = True) -> List[str]:
    """Write multiple subclips using parallel processing"""
    if max_workers is None:
        # Use up to 75% of CPU count to avoid overwhelming the system
        max_workers = max(2, int(multiprocessing.cpu_count() * 0.75))

    # Check encoding method
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is None and use_gpu_encoding:
        _NVENC_AVAILABLE = check_nvenc_available()

    encoding_method = "GPU (NVENC)" if _NVENC_AVAILABLE else "CPU (x264)"
    print(f"[performance] Writing {len(intervals)} clips using {max_workers} parallel workers")
    print(f"[performance] Video encoding: {encoding_method}")

    paths = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all clip writing tasks
        future_to_clip = {
            executor.submit(write_single_subclip, video_path, interval, k, out_dir, use_gpu_encoding): k
            for k, interval in enumerate(intervals, start=1)
        }

        # Collect results as they complete with progress bar
        with tqdm(total=len(intervals), desc="Writing clips", unit="clip") as pbar:
            for future in as_completed(future_to_clip):
                clip_num = future_to_clip[future]
                try:
                    result = future.result()
                    if result:
                        paths.append(result)
                except Exception as exc:
                    print(f"[warn] Clip {clip_num} generated an exception: {exc}")
                pbar.update(1)

    # Sort paths by clip number to maintain order
    paths.sort()
    return paths


def _trajectory_to_samples(traj: List[TrackPoint], time_offset_seconds: float = 0.0) -> List[Tuple[float, float, float]]:
    return [
        (float(point.t + time_offset_seconds), float(point.xy[0]), float(point.xy[1]))
        for point in traj
    ]


def _trajectory_to_manifest_points(
    traj: List[TrackPoint],
    trim_offset: float = 0.0,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for point in traj:
        item: Dict[str, float] = {
            "t": round(float(point.t + trim_offset), 3),
            "x": round(float(point.xy[0]), 3),
            "y": round(float(point.xy[1]), 3),
        }
        if point.bbox is not None:
            item.update(
                {
                    "x1": round(float(point.bbox[0]), 3),
                    "y1": round(float(point.bbox[1]), 3),
                    "x2": round(float(point.bbox[2]), 3),
                    "y2": round(float(point.bbox[3]), 3),
                }
            )
        rows.append(item)
    return rows


def write_single_follow_cam_subclip(
    video_path: str,
    interval: Tuple[float, float],
    clip_num: int,
    out_dir: str,
    player_traj: List[TrackPoint],
    ball_traj: List[TrackPoint],
    camera_mode: str = "follow_action",
    zoom_factor: float = 1.6,
    track_time_offset_seconds: float = 0.0,
) -> Optional[str]:
    s, e = interval
    out_path = os.path.join(out_dir, f"highlight_{clip_num:02d}.mp4")
    ball_weight = ball_weight_for_mode(camera_mode)
    try:
        return render_follow_cam_clip(
            video_path=video_path,
            output_path=out_path,
            start_seconds=s,
            end_seconds=e,
            player_track=_trajectory_to_samples(player_traj, time_offset_seconds=track_time_offset_seconds),
            ball_track=_trajectory_to_samples(ball_traj, time_offset_seconds=track_time_offset_seconds),
            zoom_factor=zoom_factor,
            ball_weight=ball_weight,
            smooth_factor=0.24,
            include_audio=True,
        )
    except Exception as ex:
        print(f"[warn] Failed to write follow-cam clip {clip_num} ({s:.1f}s - {e:.1f}s): {ex}")
        return None


def write_follow_cam_subclips(
    video_path: str,
    intervals: List[Tuple[float, float]],
    out_dir: str,
    player_traj: List[TrackPoint],
    ball_traj: List[TrackPoint],
    camera_mode: str = "follow_action",
    zoom_factor: float = 1.6,
    max_workers: Optional[int] = None,
    track_time_offset_seconds: float = 0.0,
) -> List[str]:
    if max_workers is None:
        max_workers = max(1, min(3, int(multiprocessing.cpu_count() * 0.33)))
    else:
        max_workers = max(1, min(int(max_workers), 4))

    print(f"[follow-cam] Writing {len(intervals)} clips using {max_workers} parallel workers")
    print(f"[follow-cam] Camera mode: {camera_mode} | Zoom factor: {zoom_factor:.2f}x")

    paths = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_clip = {
            executor.submit(
                write_single_follow_cam_subclip,
                video_path,
                interval,
                k,
                out_dir,
                player_traj,
                ball_traj,
                camera_mode,
                zoom_factor,
                track_time_offset_seconds,
            ): k
            for k, interval in enumerate(intervals, start=1)
        }

        with tqdm(total=len(intervals), desc="Writing follow-cam clips", unit="clip") as pbar:
            for future in as_completed(future_to_clip):
                clip_num = future_to_clip[future]
                try:
                    result = future.result()
                    if result:
                        paths.append(result)
                except Exception as exc:
                    print(f"[warn] Follow-cam clip {clip_num} generated an exception: {exc}")
                pbar.update(1)

    paths.sort()
    return paths


def write_follow_ball_subclips(
    processing_video: str,
    intervals: List[Tuple[float, float]],
    out_dir: str,
    camera_plan: CameraPlan,
    field_geometry: FieldGeometry,
    max_workers: Optional[int] = None,
) -> List[str]:
    """Render highlight clips from the game-centric camera plan.

    Intervals are in the processing-video timebase (matching the plan).
    Decode + crop runs in Python threads while encoding happens in separate
    ffmpeg processes, so a small pool scales nearly linearly (same policy as
    the legacy follow-cam clip writer).
    """
    if max_workers is None:
        max_workers = max(1, min(3, int(multiprocessing.cpu_count() * 0.33)))
    else:
        max_workers = max(1, min(int(max_workers), 4))
    print(f"[follow-ball] Writing {len(intervals)} clips from the game camera plan ({max_workers} workers)")

    def _render_one(clip_num: int, s: float, e: float) -> Optional[str]:
        out_path = os.path.join(out_dir, f"highlight_{clip_num:02d}.mp4")
        try:
            sub_plan = slice_plan(camera_plan, s, e)
            if len(sub_plan) == 0:
                print(f"[warn] Follow-ball clip {clip_num} has no planned frames ({s:.1f}s - {e:.1f}s)")
                return None
            return render_camera_plan_video(
                video_path=processing_video,
                output_path=out_path,
                plan=sub_plan,
                include_audio=True,
                geometry=field_geometry,
            )
        except Exception as ex:
            print(f"[warn] Failed to write follow-ball clip {clip_num} ({s:.1f}s - {e:.1f}s): {ex}")
            return None

    paths: List[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_render_one, clip_num, s, e): clip_num
            for clip_num, (s, e) in enumerate(intervals, start=1)
        }
        with tqdm(total=len(intervals), desc="Writing follow-ball clips", unit="clip") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    paths.append(result)
                pbar.update(1)
    paths.sort()
    return paths


def write_full_follow_cam_video(
    video_path: str,
    interval: Tuple[float, float],
    out_dir: str,
    player_traj: List[TrackPoint],
    ball_traj: List[TrackPoint],
    camera_mode: str = "follow_action",
    zoom_factor: float = 1.6,
    track_time_offset_seconds: float = 0.0,
    progress_callback: Optional[ProgressCallback] = None,
) -> Optional[str]:
    start_s, end_s = interval
    if end_s <= start_s:
        print(f"[warn] Full follow-cam interval is empty: {start_s:.2f}s - {end_s:.2f}s")
        return None

    safe_mode = str(camera_mode or "follow_action").strip().lower()
    out_path = os.path.join(out_dir, f"full_{safe_mode}_zoom.mp4")
    ball_weight = ball_weight_for_mode(safe_mode)
    duration_s = float(end_s - start_s)
    print(
        f"[follow-cam] Rendering full zoom movie: {format_time(start_s)} - {format_time(end_s)} "
        f"({duration_s:.1f}s), mode={safe_mode}, zoom={zoom_factor:.2f}x"
    )

    def _render_progress(written_frames: int, total_frames: int) -> None:
        if total_frames <= 0:
            return
        fraction = min(1.0, max(0.0, float(written_frames) / float(total_frames)))
        emit_progress(
            progress_callback,
            "rendering_full_zoom",
            0.955 + (0.03 * fraction),
            "Rendering full zoom movie",
            {
                "written_frames": int(written_frames),
                "total_frames": int(total_frames),
                "camera_mode": safe_mode,
                "zoom_factor": round(float(zoom_factor), 3),
                "output_path": out_path,
            },
        )

    try:
        return render_follow_cam_clip(
            video_path=video_path,
            output_path=out_path,
            start_seconds=start_s,
            end_seconds=end_s,
            player_track=_trajectory_to_samples(player_traj, time_offset_seconds=track_time_offset_seconds),
            ball_track=_trajectory_to_samples(ball_traj, time_offset_seconds=track_time_offset_seconds),
            zoom_factor=zoom_factor,
            ball_weight=ball_weight,
            smooth_factor=0.24,
            include_audio=True,
            progress_callback=_render_progress,
        )
    except Exception as ex:
        print(f"[warn] Failed to render full follow-cam movie ({start_s:.1f}s - {end_s:.1f}s): {ex}")
        return None


def draw_single_spotlight_overlay(video_path: str, traj: List[TrackPoint], interval: Tuple[float, float],
                                   clip_num: int, out_dir: str, radius: int = 35) -> Optional[str]:
    """Draw spotlight overlay for a single clip (used for parallel processing)"""
    s, e = interval
    t_arr = np.array([p.t for p in traj])
    xy_arr = np.array([p.xy for p in traj])

    def pos_at(t: float) -> Tuple[int, int]:
        # nearest neighbor in time
        idx = int(np.argmin(np.abs(t_arr - t)))
        x, y = xy_arr[idx]
        return int(round(x)), int(round(y))

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for overlay {clip_num}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Seek to start
        seek_success = cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000.0)
        if not seek_success:
            print(f"[warn] Failed to seek to {s}s for overlay {clip_num}, starting from beginning")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        temp_path = os.path.join(out_dir, f"highlight_{clip_num:02d}_spotlight_temp.mp4")
        out_path = os.path.join(out_dir, f"highlight_{clip_num:02d}_spotlight.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            print(f"[warn] Could not open writer for {temp_path}, trying alternate codec...")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

        frames_needed = int((e - s) * fps)
        for _ in range(frames_needed):
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            cx, cy = pos_at(t)
            # Draw soft circle
            cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 2)
            cv2.circle(frame, (cx, cy), radius + 6, (0, 0, 0), 2)
            writer.write(frame)

        writer.release()
        cap.release()

        # Add audio using moviepy
        try:
            VideoFileClip, _ = _import_moviepy()
            with VideoFileClip(video_path) as source_clip:
                with VideoFileClip(temp_path) as video_only:
                    # Extract audio from the same time interval
                    try:
                        audio_subclip = source_clip.subclip(s, e)
                    except AttributeError:
                        audio_subclip = source_clip.subclipped(s, e)
                    audio_clip = audio_subclip.audio
                    if audio_clip is not None:
                        final_clip = video_only.set_audio(audio_clip)
                        final_clip.write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None)
                        final_clip.close()
                    else:
                        # No audio in source, just rename temp file
                        os.rename(temp_path, out_path)
            # Clean up temp file if it still exists
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return out_path
        except Exception as e:
            print(f"[warn] Could not add audio to overlay {clip_num}: {e}. Using video-only version.")
            if os.path.exists(temp_path):
                os.rename(temp_path, out_path)
            return out_path
    except Exception as ex:
        print(f"[warn] Failed to create overlay for clip {clip_num}: {ex}")
        return None


def draw_spotlight_overlay(video_path: str, traj: List[TrackPoint], intervals: List[Tuple[float, float]],
                           out_dir: str, radius: int = 35, max_workers: Optional[int] = None):
    """Draw spotlight overlays using parallel processing"""
    if max_workers is None:
        # Use up to 50% of CPU count for overlays (memory intensive)
        max_workers = max(2, int(multiprocessing.cpu_count() * 0.5))

    print(f"[performance] Rendering {len(intervals)} overlays using {max_workers} parallel workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all overlay rendering tasks
        future_to_clip = {
            executor.submit(draw_single_spotlight_overlay, video_path, traj, interval, k, out_dir, radius): k
            for k, interval in enumerate(intervals, start=1)
        }

        # Collect results as they complete with progress bar
        with tqdm(total=len(intervals), desc="Rendering overlays", unit="clip") as pbar:
            for future in as_completed(future_to_clip):
                clip_num = future_to_clip[future]
                try:
                    result = future.result()
                    if not result:
                        print(f"[warn] Failed to render overlay for clip {clip_num}")
                except Exception as exc:
                    print(f"[warn] Overlay {clip_num} generated an exception: {exc}")
                pbar.update(1)




def process_video_highlights(
    video_path: str,
    output_dir: str,
    select_player: bool = False,
    pre_seconds: float = 2.0,
    post_seconds: float = 6.0,
    min_clip_duration: float = 4.0,
    no_audio: bool = False,
    overlay: bool = False,
    trim_start: Optional[float] = None,
    trim_end: Optional[float] = None,
    threads: Optional[int] = None,
    require_gpu: bool = False,
    speed_sensitivity: float = 2.0,
    audio_sensitivity: float = 2.0,
    focus_event_types: Optional[List[str]] = None,
    model_version: Optional[str] = None,
    analysis_only: bool = False,
    camera_mode: str = "wide",
    zoom_factor: float = 1.6,
    render_full_follow_cam: bool = False,
    player_roi: Optional[Dict[str, float]] = None,
    yolo_model: str = "yolo26s.pt",
    tracker_config: str = "botsort.yaml",
    inference_imgsz: int = 960,
    detection_conf: float = 0.18,
    vid_stride: int = 1,
    progress_callback: Optional[ProgressCallback] = None,
    debug: bool = False,
    log_file: Optional[str] = None,
    debug_video: bool = False,
    dump_training_data: bool = False,
    goal_box_left: Optional[Dict[str, float]] = None,
    goal_box_right: Optional[Dict[str, float]] = None,
    detect_cards: bool = True,
) -> bool:
    """Public pipeline entry point used by the CLI, GUI, and API worker.

    Configures per-run logging, then delegates to the implementation; the
    run's log-file handler is always detached and closed on exit so
    long-lived host processes do not leak file descriptors.
    """
    run_log_handler = setup_logging(debug=debug, log_file=log_file)
    try:
        return _process_video_highlights_impl(
            video_path=video_path,
            output_dir=output_dir,
            select_player=select_player,
            pre_seconds=pre_seconds,
            post_seconds=post_seconds,
            min_clip_duration=min_clip_duration,
            no_audio=no_audio,
            overlay=overlay,
            trim_start=trim_start,
            trim_end=trim_end,
            threads=threads,
            require_gpu=require_gpu,
            speed_sensitivity=speed_sensitivity,
            audio_sensitivity=audio_sensitivity,
            focus_event_types=focus_event_types,
            model_version=model_version,
            analysis_only=analysis_only,
            camera_mode=camera_mode,
            zoom_factor=zoom_factor,
            render_full_follow_cam=render_full_follow_cam,
            player_roi=player_roi,
            yolo_model=yolo_model,
            tracker_config=tracker_config,
            inference_imgsz=inference_imgsz,
            detection_conf=detection_conf,
            vid_stride=vid_stride,
            progress_callback=progress_callback,
            debug=debug,
            log_file=log_file,
            debug_video=debug_video,
            dump_training_data=dump_training_data,
            goal_box_left=goal_box_left,
            goal_box_right=goal_box_right,
            detect_cards=detect_cards,
        )
    finally:
        teardown_run_logging(run_log_handler)


def _preflight_dependencies(camera_mode: str, analysis_only: bool, no_audio: bool) -> Optional[str]:
    """Check optional heavy dependencies BEFORE any expensive work.

    Returns an error message when the run cannot possibly succeed; prints
    warnings for degradations (skipped montage, disabled audio detection).
    """
    import importlib.util

    if importlib.util.find_spec("ultralytics") is None:
        return (
            "ultralytics is not installed - player/ball tracking cannot run. "
            "Install with: pip install ultralytics"
        )
    if not analysis_only and importlib.util.find_spec("moviepy") is None:
        if camera_mode == "wide":
            return (
                "moviepy is not installed - wide-mode highlight clips cannot be rendered. "
                "Install with: pip install moviepy (or use a follow camera mode)"
            )
        print(
            "[warn] moviepy is not installed: highlight clips will render, "
            "but the montage will be skipped. Install with: pip install moviepy"
        )
    if not no_audio and importlib.util.find_spec("librosa") is None:
        print(
            "[warn] librosa is not installed: audio peak detection is disabled. "
            "Install with: pip install librosa soundfile"
        )
    return None


def _process_video_highlights_impl(
    video_path: str,
    output_dir: str,
    select_player: bool = False,
    pre_seconds: float = 2.0,
    post_seconds: float = 6.0,
    min_clip_duration: float = 4.0,
    no_audio: bool = False,
    overlay: bool = False,
    trim_start: Optional[float] = None,
    trim_end: Optional[float] = None,
    threads: Optional[int] = None,
    require_gpu: bool = False,
    speed_sensitivity: float = 2.0,
    audio_sensitivity: float = 2.0,
    focus_event_types: Optional[List[str]] = None,
    model_version: Optional[str] = None,
    analysis_only: bool = False,
    camera_mode: str = "wide",
    zoom_factor: float = 1.6,
    render_full_follow_cam: bool = False,
    player_roi: Optional[Dict[str, float]] = None,
    yolo_model: str = "yolo26s.pt",
    tracker_config: str = "botsort.yaml",
    inference_imgsz: int = 960,
    detection_conf: float = 0.18,
    vid_stride: int = 1,
    progress_callback: Optional[ProgressCallback] = None,
    debug: bool = False,
    log_file: Optional[str] = None,
    debug_video: bool = False,
    dump_training_data: bool = False,
    goal_box_left: Optional[Dict[str, float]] = None,
    goal_box_right: Optional[Dict[str, float]] = None,
    detect_cards: bool = True,
) -> bool:
    """
    Core video highlights processing function.
    This function is used by both CLI and GUI interfaces.

    Args:
        video_path: Path to input video file
        output_dir: Directory for output clips
        select_player: Whether to manually select player on first frame
        pre_seconds: Seconds before event to include
        post_seconds: Seconds after event to include
        min_clip_duration: Minimum clip duration after merging
        no_audio: Disable audio-based peak detection
        overlay: Render spotlight overlay clips
        trim_start: Start time in seconds (None for beginning)
        trim_end: End time in seconds (None for end)
        threads: Number of parallel threads for clip writing
        require_gpu: Require GPU acceleration (fail if not available)
        focus_event_types: Optional event targets to bias tuning and record run intent
        model_version: Optional model version label for run traceability
        analysis_only: Run analysis/bookmark generation without writing highlight clips
        camera_mode: wide | follow_player | follow_action
        zoom_factor: Crop zoom level for follow-cam modes
        render_full_follow_cam: Export one continuous zoomed follow-cam movie for the processed window/full source
        player_roi: Optional normalized/pixel ROI used to lock onto one player without an interactive popup
        yolo_model: Ultralytics detector weights used for player/ball tracking
        tracker_config: Ultralytics tracker config, usually botsort.yaml or bytetrack.yaml
        inference_imgsz: Inference image size; higher values use more GPU and preserve small players better
        detection_conf: Detection confidence threshold
        vid_stride: Analyze every Nth video frame
        progress_callback: Optional callback for realtime stage/progress telemetry
        debug: Print full debug diagnostics to the console
        log_file: Also write a full DEBUG log (with timestamps) to this path
        debug_video: Render an annotated wide "camera decisions" review video
            showing the center of the game, the crop box, the ball, and why
        dump_training_data: Write camera_decisions.jsonl and ball_track.csv for
            model tuning/training
        goal_box_left: Optional manual left goal box {x1,y1,x2,y2} (pixels or normalized)
        goal_box_right: Optional manual right goal box {x1,y1,x2,y2} (pixels or normalized)

    Returns:
        True if processing succeeded, False otherwise
    """
    # Check GPU requirement
    emit_progress(progress_callback, "initializing", 0.02, "Validating runtime and GPU requirements")
    if require_gpu:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: GPU acceleration is required but no CUDA-capable GPU was detected.")
            print("Please ensure:")
            print("  1. You have an NVIDIA GPU installed")
            print("  2. CUDA drivers are installed (run 'nvidia-smi' to verify)")
            print("  3. PyTorch with CUDA support is installed:")
            print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130")
            emit_progress(progress_callback, "failed", 1.0, "GPU is required but CUDA is not available")
            return False

    # Validate paths
    video_path = os.path.abspath(os.path.expanduser(video_path))
    output_dir = os.path.abspath(os.path.expanduser(output_dir))

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        emit_progress(progress_callback, "failed", 1.0, "Source video file was not found")
        return False

    if not os.path.isfile(video_path):
        print(f"Error: Path is not a file: {video_path}")
        emit_progress(progress_callback, "failed", 1.0, "Source path is not a file")
        return False

    camera_mode = str(camera_mode or "wide").strip().lower()
    if camera_mode not in FOLLOW_CAM_MODES:
        print(f"Error: Unsupported camera_mode '{camera_mode}'. Valid options: {', '.join(sorted(FOLLOW_CAM_MODES))}")
        emit_progress(progress_callback, "failed", 1.0, "Unsupported camera mode")
        return False
    zoom_factor = max(1.0, float(zoom_factor or 1.0))

    # Fail fast on missing dependencies BEFORE trimming/tracking, with an
    # actionable message (a missing package used to surface hours in, or
    # silently degrade the run).
    preflight_error = _preflight_dependencies(camera_mode, analysis_only, no_audio)
    if preflight_error:
        print(f"ERROR: {preflight_error}")
        emit_progress(progress_callback, "failed", 1.0, preflight_error)
        return False

    # Print configuration
    print(f"\nProcessing video: {video_path}")
    print(f"Output directory: {output_dir}")
    if model_version:
        print(f"Model version: {model_version}")
    requested_targets: List[str] = [str(item).strip().lower() for item in (focus_event_types or []) if str(item).strip()]
    if requested_targets:
        print(f"Focus event targets: {', '.join(requested_targets)}")
        # If sensitivity values are defaults, adapt them slightly for requested targets.
        # This does not replace event classification; it tunes highlight candidate generation.
        explosive_targets = {"goal", "shot", "penalty_kick", "save"}
        restart_targets = {"corner_kick", "free_kick", "goal_kick", "kickoff"}
        foul_targets = {"foul"}
        target_set = set(requested_targets)
        if speed_sensitivity == 2.0 and target_set.intersection(explosive_targets):
            speed_sensitivity = 1.8
        if audio_sensitivity == 2.0 and target_set.intersection(foul_targets):
            audio_sensitivity = 1.7
        if target_set.intersection(restart_targets):
            pre_seconds = max(pre_seconds, 3.0)
            post_seconds = max(post_seconds, 7.0)
    if trim_start is not None or trim_end is not None:
        print(f"Trim range: {format_time(trim_start or 0)} to {format_time(trim_end) if trim_end else 'end'}")
    print(f"Pre-event buffer: {pre_seconds}s")
    print(f"Post-event buffer: {post_seconds}s")
    print(f"Manual selection: {'Yes' if select_player else 'No'}")
    print(f"Configured player ROI: {'Yes' if player_roi else 'No'}")
    print(f"Camera mode: {camera_mode}")
    if camera_mode != "wide":
        print(f"Zoom factor: {zoom_factor:.2f}x")
    print(f"Full follow-cam movie: {'Yes' if render_full_follow_cam else 'No'}")
    print(f"Detector: {yolo_model} | Tracker: {tracker_config} | imgsz={int(inference_imgsz)} | conf={float(detection_conf):.2f} | stride={int(vid_stride)}")
    print(f"Spotlight overlay: {'Yes' if overlay else 'No'}")
    print(f"Analysis-only mode: {'Yes' if analysis_only else 'No'}")
    print(f"Debug logging: {'Yes' if debug else 'No'}{f' (log file: {log_file})' if log_file else ''}")
    print(f"Camera-decision debug video: {'Yes' if debug_video else 'No'}")
    print(f"Training data dump: {'Yes' if dump_training_data else 'No'}")
    if goal_box_left or goal_box_right:
        print(f"Manual goal boxes: left={goal_box_left or 'auto'} right={goal_box_right or 'auto'}")
    print()

    ensure_dir(output_dir)
    emit_progress(
        progress_callback,
        "source",
        0.06,
        "Source video and processing configuration resolved",
        {
            "video_path": video_path,
            "output_dir": output_dir,
            "camera_mode": camera_mode,
            "yolo_model": yolo_model,
            "tracker_config": tracker_config,
            "inference_imgsz": int(inference_imgsz),
            "vid_stride": int(vid_stride),
        },
    )

    try:
        # Create trimmed video if needed
        original_video = video_path
        emit_progress(progress_callback, "trimming", 0.10, "Preparing working video window")
        processing_video, trim_offset = create_trimmed_video(video_path, output_dir, trim_start, trim_end)
        emit_progress(
            progress_callback,
            "trimming",
            0.18,
            "Working video window ready",
            {"processing_video": processing_video, "trim_offset_seconds": round(trim_offset, 3)},
        )

        print("[1/5] Tracking players and ball (YOLO + ByteTrack)...")
        tracks, ball_traj, fps, (W, H), selection_metadata = track_video(
            processing_video,
            select_roi=select_player,
            player_roi=player_roi,
            yolo_model=yolo_model,
            tracker_config=tracker_config,
            inference_imgsz=inference_imgsz,
            detection_conf=detection_conf,
            vid_stride=vid_stride,
            progress_callback=progress_callback,
            progress_start=0.20,
            progress_end=0.65,
        )
        target_id = int(selection_metadata.get("target_track_id") or list(tracks.keys())[0])
        traj = tracks[target_id]
        emit_progress(
            progress_callback,
            "tracking",
            0.67,
            "Target player track selected",
            {
                "target_track_id": target_id,
                "stitched_track_count": int(selection_metadata.get("stitched_track_count") or 1),
                "ball_detection_count": len(ball_traj),
            },
        )

        print("[2/5] Computing multi-factor highlights (speed + ball proximity + direction changes)...")
        emit_progress(progress_callback, "scoring", 0.70, "Computing speed, ball proximity, and direction-change signals")
        times, speed = compute_speed_series(traj, fps)

        # Compute ball proximity scores
        prox_times, ball_proximity = compute_ball_proximity_score(traj, ball_traj, proximity_threshold=200.0)

        # Compute direction changes
        dir_times, direction_changes = compute_direction_changes(traj, fps)

        # Use multi-factor detection that combines speed, ball proximity, and direction changes
        speed_intervals = detect_highlights_multi_factor(
            times, speed, ball_proximity, direction_changes,
            pre=pre_seconds, post=post_seconds,
            speed_weight=0.5, proximity_weight=0.3, direction_weight=0.2,
            k=speed_sensitivity
        )

        audio_intervals = []
        if not no_audio:
            print("[3/5] Detecting audio peaks...")
            emit_progress(progress_callback, "audio", 0.76, "Detecting audio peaks")
            audio_intervals = detect_audio_peaks(processing_video, pre=pre_seconds, post=post_seconds, k=audio_sensitivity)
        else:
            emit_progress(progress_callback, "audio", 0.78, "Audio analysis skipped by configuration")

        print("[4/5] Merging and pruning intervals...")
        emit_progress(progress_callback, "bookmarks", 0.82, "Merging detection intervals into review candidates")
        LOGGER.debug(f"Total intervals before merge: multi-factor={len(speed_intervals)}, audio={len(audio_intervals)}")
        intervals = merge_intervals(speed_intervals + audio_intervals)
        fallback_intervals: List[Tuple[float, float]] = []
        if not intervals:
            print("[info] No threshold-based highlights found. Selecting strongest motion windows for review.")
            # Always-visible diagnostics for the "found nothing" case so a
            # multi-hour run does not need repeating with --debug to see why.
            LOGGER.info(
                "detection diagnostics: speed_samples=%d (median=%.1f max=%.1f px/s), "
                "audio_intervals=%d, sensitivity: speed=%.2f audio=%.2f",
                len(speed),
                float(np.median(speed)) if len(speed) else 0.0,
                float(speed.max()) if len(speed) else 0.0,
                len(audio_intervals),
                speed_sensitivity,
                audio_sensitivity,
            )
            fallback_intervals = detect_review_candidate_intervals(
                times,
                speed,
                direction_changes,
                pre=pre_seconds,
                post=post_seconds,
                max_candidates=3,
            )
            intervals = merge_intervals(fallback_intervals)
        LOGGER.debug(f"Total intervals after final merge: {len(intervals)}")

        # Get video duration to clamp intervals
        cap_check = cv2.VideoCapture(processing_video)
        video_duration = cap_check.get(cv2.CAP_PROP_FRAME_COUNT) / cap_check.get(cv2.CAP_PROP_FPS) if cap_check.isOpened() else float('inf')
        cap_check.release()

        analysis_end_s = video_duration
        if not math.isfinite(analysis_end_s) or analysis_end_s <= 0:
            analysis_end_s = max(
                traj[-1].t if traj else 0.0,
                ball_traj[-1].t if ball_traj else 0.0,
                1.0,
            )

        print("[4b] Analyzing ball track, field geometry, and game states...")
        emit_progress(progress_callback, "game_analysis", 0.83, "Building ball track and game-state timeline")
        player_positions = selection_metadata.get("player_positions")
        ball_track = build_ball_track(ball_traj, (W, H))
        field_geometry = estimate_field_geometry(
            player_positions, (W, H),
            goal_box_left=goal_box_left, goal_box_right=goal_box_right,
        )
        goal_events = detect_goal_events(ball_track, field_geometry, 0.0, analysis_end_s)
        game_segments = analyze_game_states(
            ball_track, field_geometry, 0.0, analysis_end_s, goal_events=goal_events
        )
        set_piece_events = detect_set_pieces(ball_track, field_geometry, 0.0, analysis_end_s)
        game_segments = overlay_set_piece_states(game_segments, set_piece_events)
        for sp in set_piece_events:
            print(f"[game] {sp.kind} at {format_time(sp.t_kick + trim_offset)}: {sp.reason}")

        card_events = []
        if detect_cards:
            emit_progress(progress_callback, "game_analysis", 0.842, "Scanning stopped play for referee cards")
            try:
                card_events = detect_card_events(
                    processing_video,
                    stopped_play_windows(game_segments),
                    debug_dir=os.path.join(output_dir, "card_crops"),
                    ball_track=ball_track,
                )
            except Exception as card_exc:
                print(f"[warn] Card detection failed: {card_exc}")
            for card in card_events:
                print(
                    f"[game] {card.kind.replace('_', ' ').upper()} flagged at "
                    f"{format_time(card.t + trim_offset)} (confidence {card.confidence:.2f})"
                )

        state_summary = summarize_states(game_segments)
        ball_coverage = ball_track.coverage_fraction(0.0, analysis_end_s)
        print(
            f"[game] Ball visible {ball_coverage * 100.0:.1f}% of the window | "
            f"states: {state_summary} | goals flagged: {len(goal_events)}"
        )
        for goal in goal_events:
            print(
                f"[game] GOAL flagged at {format_time(goal.t + trim_offset)} "
                f"({goal.side} goal, confidence {goal.confidence:.2f}): {goal.reason}"
            )
        emit_progress(
            progress_callback,
            "game_analysis",
            0.845,
            "Game-state analysis complete",
            {
                "ball_coverage_fraction": round(ball_coverage, 3),
                "goal_event_count": len(goal_events),
                "state_summary": state_summary,
                "field_geometry_source": field_geometry.source,
            },
        )

        # Guarantee every flagged goal gets a highlight interval, even when
        # motion/audio thresholds missed it.
        goal_intervals = [
            (max(0.0, goal.t - max(pre_seconds, 4.0)), min(analysis_end_s, goal.t + max(post_seconds, 6.0)))
            for goal in goal_events
        ]
        threat_set_pieces = [
            sp for sp in set_piece_events
            if sp.kind in {"corner_kick", "penalty_kick"}
            or (sp.kind == "free_kick" and sp.side is not None)
        ]
        event_intervals = goal_intervals + [
            (max(0.0, sp.t_kick - 3.0), min(analysis_end_s, sp.t_kick + 7.0))
            for sp in threat_set_pieces
        ] + [
            (max(0.0, card.t - 4.0), min(analysis_end_s, card.t + 6.0))
            for card in card_events
        ]
        if event_intervals:
            intervals = merge_intervals(sorted(intervals + event_intervals))
            LOGGER.debug(
                "intervals after adding %d goal/%d set-piece/%d card interval(s): %d",
                len(goal_intervals), len(threat_set_pieces), len(card_events), len(intervals),
            )

        # Enforce minimum clip length and clamp to video duration
        clamped_intervals = []
        for i, (s, e) in enumerate(intervals):
            duration = e - s
            if duration < min_clip_duration:
                e = min(s + min_clip_duration, video_duration)
                LOGGER.debug(f"Interval {i+1}: extended from {duration:.2f}s to {e-s:.2f}s (min={min_clip_duration}s)")
            else:
                e = min(e, video_duration)
            if e > s:
                clamped_intervals.append((s, e))
                LOGGER.debug(f"Interval {i+1}: [{s:.2f}s - {e:.2f}s] duration={e-s:.2f}s")
        intervals = clamped_intervals

        LOGGER.debug(f"Final intervals after clamping: {len(intervals)}")

        # Adjust intervals back to original video timestamps if trimmed
        original_intervals = [(s + trim_offset, e + trim_offset) for s, e in intervals]
        original_speed_intervals = [(s + trim_offset, e + trim_offset) for s, e in speed_intervals]
        original_audio_intervals = [(s + trim_offset, e + trim_offset) for s, e in audio_intervals]
        LOGGER.debug(f"Original video intervals (with trim offset +{trim_offset:.2f}s): {len(original_intervals)}")

        if trim_offset > 0:
            print(f"[info] Found {len(intervals)} highlights. Adjusting timestamps to original video (offset: +{format_time(trim_offset)})")

        # Shared game-analysis payloads (original-video timebase), referenced
        # by both the tracking manifest and analysis_game_states.json so the
        # two files can never drift apart.
        ball_track_stats = {
            **{key: float(value) for key, value in ball_track.stats.items()},
            "coverage_fraction": round(ball_coverage, 4),
        }
        original_goal_events = [
            {**goal.to_dict(), "t": round(goal.t + trim_offset, 3)} for goal in goal_events
        ]
        original_set_pieces = [
            {
                **sp.to_dict(),
                "t_start": round(sp.t_start + trim_offset, 3),
                "t_kick": round(sp.t_kick + trim_offset, 3),
            }
            for sp in set_piece_events
        ]
        original_card_events = [
            {**card.to_dict(), "t": round(card.t + trim_offset, 3)} for card in card_events
        ]

        tracking_manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_video_path": original_video,
            "processing_video_path": processing_video,
            "output_dir": output_dir,
            "trim_offset_seconds": round(trim_offset, 3),
            "camera": {
                "mode": camera_mode,
                "zoom_factor": round(zoom_factor, 3),
                "render_full_follow_cam": bool(render_full_follow_cam),
            },
            "detector": {
                "yolo_model": yolo_model,
                "tracker_config": tracker_config,
                "inference_imgsz": int(inference_imgsz),
                "detection_conf": round(float(detection_conf), 3),
                "vid_stride": int(vid_stride),
            },
            "selection": {
                "manual_select_popup": bool(select_player),
                "player_roi": dict(player_roi or {}),
                "stitched_track_ids": list(selection_metadata.get("stitched_track_ids") or [int(target_id)]),
                "stitched_track_count": int(selection_metadata.get("stitched_track_count") or 1),
            },
            "video": {
                "fps": round(float(fps), 3),
                "frame_width": int(W),
                "frame_height": int(H),
            },
            "tracking": {
                "target_track_id": int(target_id),
                "target_track_ids": list(selection_metadata.get("stitched_track_ids") or [int(target_id)]),
                "target_track": _trajectory_to_manifest_points(traj, trim_offset=trim_offset),
                "ball_track": _trajectory_to_manifest_points(ball_traj, trim_offset=trim_offset),
            },
            "game_analysis": {
                "field_geometry": field_geometry.to_dict(),
                "ball_track_stats": ball_track_stats,
                "state_summary_s": state_summary,
                "goal_events": original_goal_events,
            },
        }
        tracking_manifest_path = write_tracking_manifest(output_dir, tracking_manifest)

        original_game_states = [
            {
                **seg.to_dict(),
                "start_s": round(seg.start_s + trim_offset, 3),
                "end_s": round(seg.end_s + trim_offset, 3),
            }
            for seg in game_segments
        ]
        game_states_path = os.path.join(output_dir, "analysis_game_states.json")
        with open(game_states_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "trim_offset_seconds": round(trim_offset, 3),
                    "field_geometry": field_geometry.to_dict(),
                    "ball_track_stats": ball_track_stats,
                    "state_summary_s": state_summary,
                    "segments": original_game_states,
                    "goal_events": original_goal_events,
                    "set_piece_events": original_set_pieces,
                    "card_events": original_card_events,
                },
                handle,
                indent=2,
            )
        print(f"[analysis] Game-state manifest: {game_states_path}")
        print(f"[analysis] Tracking manifest: {tracking_manifest_path}")
        emit_progress(
            progress_callback,
            "tracking_manifest",
            0.86,
            "Tracking manifest written",
            {"tracking_manifest_path": tracking_manifest_path},
        )

        live_manifest_path = os.path.join(output_dir, "analysis_bookmarks.json")
        live_manifest_context: Dict[str, object] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "video_path": original_video,
            "processing_video_path": processing_video,
            "output_dir": output_dir,
            "analysis_only": analysis_only,
            "model_version": model_version,
            "focus_event_types": requested_targets,
            "trim_offset_seconds": round(trim_offset, 3),
            "tracking_manifest_path": tracking_manifest_path,
            "settings": {
                "pre_seconds": pre_seconds,
                "post_seconds": post_seconds,
                "min_clip_duration": min_clip_duration,
                "speed_sensitivity": speed_sensitivity,
                "audio_sensitivity": audio_sensitivity,
                "camera_mode": camera_mode,
                "zoom_factor": round(zoom_factor, 3),
                "render_full_follow_cam": bool(render_full_follow_cam),
                "no_audio": no_audio,
                "overlay": overlay,
                "threads": threads,
                "yolo_model": yolo_model,
                "tracker_config": tracker_config,
                "inference_imgsz": int(inference_imgsz),
                "detection_conf": round(float(detection_conf), 3),
                "vid_stride": int(vid_stride),
                "debug": bool(debug),
                "debug_video": bool(debug_video),
                "dump_training_data": bool(dump_training_data),
                "goal_box_left": dict(goal_box_left or {}),
                "goal_box_right": dict(goal_box_right or {}),
            },
            "stats": {
                "speed_interval_count": len(speed_intervals),
                "audio_interval_count": len(audio_intervals),
                "fallback_interval_count": len(fallback_intervals),
                "merged_interval_count": len(intervals),
                "goal_event_count": len(goal_events),
                "bookmark_count": 0,
            },
        }

        bookmarks = build_analysis_bookmarks(
            original_intervals=original_intervals,
            speed_intervals=original_speed_intervals,
            audio_intervals=original_audio_intervals,
            requested_targets=requested_targets,
            live_manifest_path=live_manifest_path,
            live_manifest_context=live_manifest_context,
            goal_events=original_goal_events,
            game_states=original_game_states,
            card_events=original_card_events,
            set_piece_events=original_set_pieces,
        )
        emit_progress(
            progress_callback,
            "bookmarks",
            0.90,
            "Bookmark candidates built",
            {"bookmark_count": len(bookmarks), "fallback_interval_count": len(fallback_intervals)},
        )
        manifest: Dict[str, object] = {
            **live_manifest_context,
            "stats": {
                "speed_interval_count": len(speed_intervals),
                "audio_interval_count": len(audio_intervals),
                "fallback_interval_count": len(fallback_intervals),
                "merged_interval_count": len(intervals),
                "goal_event_count": len(goal_events),
                "bookmark_count": len(bookmarks),
            },
            "goal_events": original_goal_events,
            "game_states_path": game_states_path,
            "bookmarks": bookmarks,
        }
        manifest_path, csv_path = write_analysis_bookmark_files(output_dir, manifest)
        print(f"[analysis] Bookmark manifest: {manifest_path}")
        print(f"[analysis] Bookmark table: {csv_path}")
        emit_progress(
            progress_callback,
            "bookmarks",
            0.94,
            "Bookmark manifest and table written",
            {"analysis_manifest_path": manifest_path, "analysis_table_csv_path": csv_path, "bookmark_count": len(bookmarks)},
        )

        # --- Game-centric camera plan (center of the game + why) ---
        camera_plan: Optional[CameraPlan] = None
        need_camera_plan = camera_mode == "follow_ball" or debug_video or dump_training_data
        if need_camera_plan:
            emit_progress(progress_callback, "camera_plan", 0.945, "Planning game-centric camera path")
            camera_plan = plan_camera(
                ball_track=ball_track,
                player_positions=player_positions,
                geometry=field_geometry,
                segments=game_segments,
                start_seconds=0.0,
                end_seconds=analysis_end_s,
                fps=fps,
                frame_size=(W, H),
                base_zoom=zoom_factor if camera_mode != "wide" else 1.6,
            )
            print(f"[camera] Plan summary: {camera_plan.summary()}")

        if dump_training_data:
            decisions_path = os.path.join(output_dir, "camera_decisions.jsonl")
            camera_plan.write_jsonl(
                decisions_path,
                transform=lambda row: {**row, "t_source": round(float(row["t"]) + trim_offset, 3)},
            )
            ball_csv_path = os.path.join(output_dir, "ball_track.csv")
            with open(ball_csv_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["t", "t_source", "x", "y"])
                writer.writeheader()
                for row in ball_track.to_rows():
                    writer.writerow({**row, "t_source": round(row["t"] + trim_offset, 3)})
            print(f"[training] Camera decisions: {decisions_path}")
            print(f"[training] Ball track table: {ball_csv_path}")
            emit_progress(
                progress_callback,
                "training_dump",
                0.948,
                "Training data written",
                {"camera_decisions_path": decisions_path, "ball_track_csv_path": ball_csv_path},
            )

        if debug_video:
            debug_path = os.path.join(output_dir, "debug_camera_wide.mp4")
            LOGGER.debug("Rendering annotated wide camera-decision video (center of the game + why)...")
            emit_progress(progress_callback, "debug_video", 0.95, "Rendering camera-decision debug video")
            try:
                rendered_debug = render_camera_plan_video(
                    video_path=processing_video,
                    output_path=debug_path,
                    plan=camera_plan,
                    include_audio=True,
                    debug_wide=True,
                    geometry=field_geometry,
                )
                LOGGER.debug(f"Camera-decision video: {rendered_debug}")
            except Exception as exc:
                print(f"[warn] Debug video render failed: {exc}")

        if analysis_only:
            print("[analysis] Analysis-only run complete. Skipped clip rendering.")
            emit_progress(progress_callback, "completed", 0.98, "Analysis-only run complete")
            return True

        full_follow_cam_path: Optional[str] = None
        if render_full_follow_cam:
            if camera_mode == "wide":
                print("[warn] Full follow-cam movie requested, but camera mode is wide. Skipping full zoom export.")
            elif camera_mode == "follow_ball":
                emit_progress(
                    progress_callback,
                    "rendering_full_zoom",
                    0.955,
                    "Rendering full game-camera movie",
                    {"camera_mode": camera_mode, "zoom_factor": round(float(zoom_factor), 3)},
                )

                def _full_ball_progress(written_frames: int, total_frames: int) -> None:
                    if total_frames <= 0:
                        return
                    fraction = min(1.0, max(0.0, float(written_frames) / float(total_frames)))
                    emit_progress(
                        progress_callback,
                        "rendering_full_zoom",
                        0.955 + (0.03 * fraction),
                        "Rendering full game-camera movie",
                        {"written_frames": int(written_frames), "total_frames": int(total_frames)},
                    )

                try:
                    full_follow_cam_path = render_camera_plan_video(
                        video_path=processing_video,
                        output_path=os.path.join(output_dir, "full_follow_ball_zoom.mp4"),
                        plan=camera_plan,
                        include_audio=True,
                        debug_wide=False,
                        geometry=field_geometry,
                        progress_callback=_full_ball_progress,
                    )
                    print(f"[follow-cam] Full game-camera movie: {full_follow_cam_path}")
                except Exception as exc:
                    print(f"[warn] Full game-camera render failed: {exc}")
                emit_progress(
                    progress_callback,
                    "rendering_full_zoom",
                    0.985,
                    "Full zoom movie rendered" if full_follow_cam_path else "Full zoom movie render did not produce output",
                    {"full_follow_cam_path": full_follow_cam_path or ""},
                )
            else:
                full_interval = (float(trim_offset), float(trim_offset + video_duration))
                emit_progress(
                    progress_callback,
                    "rendering_full_zoom",
                    0.955,
                    "Rendering full zoom movie",
                    {
                        "start_seconds": round(full_interval[0], 3),
                        "end_seconds": round(full_interval[1], 3),
                        "camera_mode": camera_mode,
                        "zoom_factor": round(float(zoom_factor), 3),
                    },
                )
                full_follow_cam_path = write_full_follow_cam_video(
                    original_video,
                    full_interval,
                    output_dir,
                    traj,
                    ball_traj,
                    camera_mode=camera_mode,
                    zoom_factor=zoom_factor,
                    track_time_offset_seconds=trim_offset,
                    progress_callback=progress_callback,
                )
                emit_progress(
                    progress_callback,
                    "rendering_full_zoom",
                    0.985,
                    "Full zoom movie rendered" if full_follow_cam_path else "Full zoom movie render did not produce output",
                    {"full_follow_cam_path": full_follow_cam_path or ""},
                )

        if not intervals:
            print("No highlight intervals found. Bookmark table generated for manual review.")
            if full_follow_cam_path:
                return True
            emit_progress(progress_callback, "failed", 0.98, "Analysis finished with no highlight clip intervals")
            return False

        print("[5/5] Writing subclips...")
        emit_progress(progress_callback, "rendering", 0.95, "Rendering highlight clips")
        if camera_mode == "wide":
            clip_paths = write_subclips(original_video, original_intervals, output_dir, max_workers=threads)
        elif camera_mode == "follow_ball":
            clip_paths = write_follow_ball_subclips(
                processing_video,
                intervals,
                output_dir,
                camera_plan,
                field_geometry,
                max_workers=threads,
            )
        else:
            clip_paths = write_follow_cam_subclips(
                original_video,
                original_intervals,
                output_dir,
                traj,
                ball_traj,
                camera_mode=camera_mode,
                zoom_factor=zoom_factor,
                max_workers=threads,
                track_time_offset_seconds=trim_offset,
            )

        # Montage (best-effort: clips are the deliverable, so a montage
        # failure - e.g. moviepy missing on this host - must not fail a run
        # whose clips all rendered successfully).
        if clip_paths:
            try:
                VideoFileClip, concatenate_videoclips = _import_moviepy()
                clips = []
                try:
                    clips = [VideoFileClip(p) for p in clip_paths]
                    montage = concatenate_videoclips(clips, method="compose")
                    montage_path = os.path.join(output_dir, "highlights_montage.mp4")
                    montage.write_videofile(montage_path, codec="libx264", audio_codec="aac")
                    montage.close()
                finally:
                    for c in clips:
                        c.close()
                print(f"Wrote {len(clip_paths)} clips and a montage to: {output_dir}")
            except Exception as montage_exc:
                print(f"[warn] Montage skipped ({montage_exc}). {len(clip_paths)} clips are in: {output_dir}")
        emit_progress(progress_callback, "rendering", 0.98, "Clip rendering complete", {"clip_count": len(clip_paths)})

        # Optional overlay rendering
        if overlay:
            print("[overlay] Rendering spotlight overlays (this can take a while)...")
            overlay_workers = min(2, threads) if threads else None
            draw_spotlight_overlay(original_video, traj, original_intervals, output_dir, max_workers=overlay_workers)
            print("[overlay] Done.")
            emit_progress(progress_callback, "overlay", 0.99, "Spotlight overlays rendered")

        return True

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        emit_progress(progress_callback, "failed", 1.0, "Processing failed", {"error": str(e)})
        return False


def main():
    ap = argparse.ArgumentParser(description="Soccer highlight generator (YOLO+ByteTrack + audio peaks)")
    ap.add_argument("--video", help="Input video path (iPhone recording)")
    ap.add_argument("--out", help="Output directory for highlights")
    ap.add_argument("--select", action="store_true", help="Interactively select your player's box on first frame")
    ap.add_argument("--pre", type=float, default=2.0, help="Seconds before event")
    ap.add_argument("--post", type=float, default=6.0, help="Seconds after event")
    ap.add_argument("--min-clip", type=float, default=4.0, help="Minimum clip duration (after merging)")
    ap.add_argument("--no-audio", action="store_true", help="Disable audio-based peak detection")
    ap.add_argument("--overlay", action="store_true", help="Render spotlight overlay clips (slower)")
    ap.add_argument("--trim-start", type=str, help="Trim video start time (format: MM:SS or HH:MM:SS or seconds)")
    ap.add_argument("--trim-end", type=str, help="Trim video end time (format: MM:SS or HH:MM:SS or seconds)")
    ap.add_argument("--threads", type=int, default=None, help="Number of parallel threads for clip writing (default: auto, max 4)")
    ap.add_argument("--speed-sensitivity", type=float, default=2.0, help="Speed detection sensitivity (lower = more sensitive, default: 2.0, old default was 3.0)")
    ap.add_argument("--audio-sensitivity", type=float, default=2.0, help="Audio peak detection sensitivity (lower = more sensitive, default: 2.0, old default was 3.0)")
    ap.add_argument("--analysis-only", action="store_true", help="Run detection/bookmark analysis without writing highlight clips")
    ap.add_argument("--camera-mode", choices=sorted(FOLLOW_CAM_MODES), default="wide", help="Video framing mode for rendered clips (follow_ball = game-centric camera that tracks the ball)")
    ap.add_argument("--zoom-factor", type=float, default=1.6, help="Zoom factor for follow camera modes")
    ap.add_argument("--render-full-follow-cam", action="store_true", help="Also render one continuous zoomed follow-cam movie for the processed window/full source")
    ap.add_argument("--debug", action="store_true", help="Print full debug diagnostics to the console")
    ap.add_argument("--log-file", type=str, default=None, help="Write a full DEBUG log (with timestamps) to this file")
    ap.add_argument("--debug-video", action="store_true", help="Render debug_camera_wide.mp4: annotated wide video showing the center of the game, crop box, ball trail, and why each camera decision was made")
    ap.add_argument("--dump-training-data", action="store_true", help="Write camera_decisions.jsonl and ball_track.csv for tuning/training")
    ap.add_argument("--goal-box-left", type=str, default=None, help="Manual left goal box as x1,y1,x2,y2 (normalized 0-1 or pixels); overrides auto estimate")
    ap.add_argument("--goal-box-right", type=str, default=None, help="Manual right goal box as x1,y1,x2,y2 (normalized 0-1 or pixels); overrides auto estimate")
    ap.add_argument("--no-card-detection", action="store_true", help="Disable yellow/red card flagging (enabled by default)")
    args = ap.parse_args()

    def _parse_goal_box(raw: Optional[str], flag: str) -> Optional[Dict[str, float]]:
        if not raw:
            return None
        parts = [item.strip() for item in raw.split(",")]
        if len(parts) != 4:
            print(f"Error: {flag} expects x1,y1,x2,y2 (got: {raw})")
            sys.exit(1)
        try:
            x1, y1, x2, y2 = (float(item) for item in parts)
        except ValueError:
            print(f"Error: {flag} values must be numbers (got: {raw})")
            sys.exit(1)
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    goal_box_left = _parse_goal_box(args.goal_box_left, "--goal-box-left")
    goal_box_right = _parse_goal_box(args.goal_box_right, "--goal-box-right")

    # Interactive mode if video or output not provided
    if not args.video:
        print("\n=== Video Highlights Generator ===\n")
        args.video = input("Enter the path to your video file: ").strip()
        if not args.video:
            print("Error: Video path is required")
            sys.exit(1)

    if not args.out:
        default_out = "./highlights_output"
        out_input = input(f"Enter output directory (default: {default_out}): ").strip()
        args.out = out_input if out_input else default_out

    # Validate and normalize paths
    args.video = os.path.abspath(os.path.expanduser(args.video))
    args.out = os.path.abspath(os.path.expanduser(args.out))

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    if not os.path.isfile(args.video):
        print(f"Error: Path is not a file: {args.video}")
        sys.exit(1)

    # Validate video file extension
    valid_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.MP4', '.MOV', '.AVI', '.MKV', '.M4V'}
    if not any(args.video.endswith(ext) for ext in valid_extensions):
        print(f"Warning: File extension may not be a valid video format: {args.video}")

    # Ask about player selection if not already set
    if not args.select and sys.stdin.isatty():
        select_input = input("Do you want to manually select your player on the first frame? (y/N): ").strip().lower()
        args.select = select_input in ['y', 'yes']

    # Ask about overlay if not already set
    if not args.overlay and sys.stdin.isatty():
        overlay_input = input("Do you want to render spotlight overlay clips? (slower) (y/N): ").strip().lower()
        args.overlay = overlay_input in ['y', 'yes']

    # Ask about analysis-only mode if not already set
    if not args.analysis_only and sys.stdin.isatty():
        analysis_input = input("Run analysis-only mode (bookmarks table, no clip rendering)? (y/N): ").strip().lower()
        args.analysis_only = analysis_input in ['y', 'yes']

    # Ask about trimming if not already set
    trim_start_seconds = None
    trim_end_seconds = None

    if args.trim_start:
        try:
            trim_start_seconds = parse_time(args.trim_start)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    if args.trim_end:
        try:
            trim_end_seconds = parse_time(args.trim_end)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Interactive trim prompts
    if not args.trim_start and not args.trim_end and sys.stdin.isatty():
        trim_input = input("Do you want to trim the video to a specific time range? (y/N): ").strip().lower()
        if trim_input in ['y', 'yes']:
            start_input = input("Enter start time (MM:SS, HH:MM:SS, or seconds) [press Enter for beginning]: ").strip()
            if start_input:
                try:
                    trim_start_seconds = parse_time(start_input)
                except ValueError as e:
                    print(f"Error: {e}")
                    sys.exit(1)

            end_input = input("Enter end time (MM:SS, HH:MM:SS, or seconds) [press Enter for end]: ").strip()
            if end_input:
                try:
                    trim_end_seconds = parse_time(end_input)
                except ValueError as e:
                    print(f"Error: {e}")
                    sys.exit(1)

    # Call the core processing function
    success = process_video_highlights(
        video_path=args.video,
        output_dir=args.out,
        select_player=args.select,
        pre_seconds=args.pre,
        post_seconds=args.post,
        min_clip_duration=args.min_clip,
        no_audio=args.no_audio,
        overlay=args.overlay,
        trim_start=trim_start_seconds,
        trim_end=trim_end_seconds,
        threads=args.threads,
        require_gpu=False,  # CLI doesn't require GPU by default
        speed_sensitivity=args.speed_sensitivity,
        audio_sensitivity=args.audio_sensitivity,
        analysis_only=args.analysis_only,
        camera_mode=args.camera_mode,
        zoom_factor=args.zoom_factor,
        render_full_follow_cam=args.render_full_follow_cam,
        debug=args.debug,
        log_file=args.log_file,
        debug_video=args.debug_video,
        dump_training_data=args.dump_training_data,
        goal_box_left=goal_box_left,
        goal_box_right=goal_box_right,
        detect_cards=not args.no_card_detection,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
