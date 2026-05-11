from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .ffmpeg_tools import ffmpeg_available, ffmpeg_exe
from ..utils import ensure_dir

TrackSample = Tuple[float, float, float]


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for video follow-cam rendering") from exc
    return cv2


def _normalize_track(track: Optional[Iterable[TrackSample]]) -> np.ndarray:
    if not track:
        return np.empty((0, 3), dtype=np.float32)

    rows: List[Tuple[float, float, float]] = []
    for item in track:
        try:
            t, x, y = item
            rows.append((float(t), float(x), float(y)))
        except Exception:
            continue

    if not rows:
        return np.empty((0, 3), dtype=np.float32)

    rows.sort(key=lambda value: value[0])
    return np.asarray(rows, dtype=np.float32)


def _interpolate_track_point(track: np.ndarray, t: float, max_gap_seconds: Optional[float] = None) -> Optional[Tuple[float, float]]:
    if track.size == 0:
        return None

    times = track[:, 0]
    idx = int(np.searchsorted(times, t))

    if idx <= 0:
        nearest_gap = abs(float(t) - float(times[0]))
        if max_gap_seconds is not None and nearest_gap > max_gap_seconds:
            return None
        return float(track[0, 1]), float(track[0, 2])

    if idx >= len(track):
        nearest_gap = abs(float(t) - float(times[-1]))
        if max_gap_seconds is not None and nearest_gap > max_gap_seconds:
            return None
        return float(track[-1, 1]), float(track[-1, 2])

    left = track[idx - 1]
    right = track[idx]
    left_gap = abs(float(t) - float(left[0]))
    right_gap = abs(float(right[0]) - float(t))
    nearest_gap = min(left_gap, right_gap)
    if max_gap_seconds is not None and nearest_gap > max_gap_seconds:
        return None

    span = float(right[0] - left[0])
    if span <= 1e-6:
        return float(left[1]), float(left[2])

    alpha = float((t - left[0]) / span)
    x = float(left[1] + (right[1] - left[1]) * alpha)
    y = float(left[2] + (right[2] - left[2]) * alpha)
    return x, y


def _clamp_center(center: Tuple[float, float], frame_size: Tuple[int, int], zoom_factor: float) -> Tuple[float, float]:
    frame_w, frame_h = frame_size
    crop_w = max(2.0, float(frame_w) / max(1.0, float(zoom_factor)))
    crop_h = max(2.0, float(frame_h) / max(1.0, float(zoom_factor)))
    half_w = crop_w / 2.0
    half_h = crop_h / 2.0
    x = min(max(center[0], half_w), max(half_w, float(frame_w) - half_w))
    y = min(max(center[1], half_h), max(half_h, float(frame_h) - half_h))
    return x, y


def build_follow_cam_centers(
    player_track: Sequence[TrackSample],
    ball_track: Optional[Sequence[TrackSample]],
    start_seconds: float,
    end_seconds: float,
    fps: float,
    frame_size: Tuple[int, int],
    zoom_factor: float = 1.6,
    ball_weight: float = 0.0,
    smooth_factor: float = 0.2,
    max_player_gap_seconds: float = 0.75,
    max_ball_gap_seconds: float = 0.35,
) -> List[Tuple[float, float]]:
    if end_seconds <= start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds")
    if fps <= 0:
        raise ValueError("fps must be positive")

    player = _normalize_track(player_track)
    ball = _normalize_track(ball_track)
    frame_w, frame_h = frame_size
    frame_count = max(1, int(math.ceil((end_seconds - start_seconds) * fps)))

    centers: List[Tuple[float, float]] = []
    previous: Optional[Tuple[float, float]] = None
    max_step = max(frame_w, frame_h) / max(12.0, zoom_factor * 6.0)

    for index in range(frame_count):
        t = float(start_seconds + (index / fps))
        player_point = _interpolate_track_point(player, t, max_gap_seconds=max_player_gap_seconds)
        ball_point = None
        if ball_weight > 0.0:
            ball_point = _interpolate_track_point(ball, t, max_gap_seconds=max_ball_gap_seconds)

        if player_point is not None:
            focus_x, focus_y = player_point
            if ball_point is not None:
                focus_x = (focus_x * (1.0 - ball_weight)) + (ball_point[0] * ball_weight)
                focus_y = (focus_y * (1.0 - ball_weight)) + (ball_point[1] * ball_weight)
        elif ball_point is not None:
            focus_x, focus_y = ball_point
        else:
            focus_x, focus_y = frame_w / 2.0, frame_h / 2.0

        raw_center = _clamp_center((focus_x, focus_y), frame_size, zoom_factor)
        if previous is None:
            smoothed = raw_center
        else:
            dx = raw_center[0] - previous[0]
            dy = raw_center[1] - previous[1]
            distance = math.hypot(dx, dy)
            if distance > max_step > 0:
                scale = max_step / distance
                dx *= scale
                dy *= scale
            smoothed = (
                previous[0] + (dx * smooth_factor),
                previous[1] + (dy * smooth_factor),
            )
            smoothed = _clamp_center(smoothed, frame_size, zoom_factor)

        centers.append(smoothed)
        previous = smoothed

    return centers


def crop_frame_to_center(
    frame: np.ndarray,
    center: Tuple[float, float],
    zoom_factor: float,
    output_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    frame_h, frame_w = frame.shape[:2]
    target_w, target_h = output_size or (frame_w, frame_h)
    crop_w = max(2, min(frame_w, int(round(frame_w / max(1.0, float(zoom_factor))))))
    crop_h = max(2, min(frame_h, int(round(frame_h / max(1.0, float(zoom_factor))))))

    clamped_x, clamped_y = _clamp_center(center, (frame_w, frame_h), zoom_factor)
    x1 = int(round(clamped_x - (crop_w / 2.0)))
    y1 = int(round(clamped_y - (crop_h / 2.0)))
    x1 = min(max(0, x1), max(0, frame_w - crop_w))
    y1 = min(max(0, y1), max(0, frame_h - crop_h))
    cropped = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
    if cropped.size == 0:
        cropped = frame
    try:
        cv2 = _import_cv2()
        return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    except RuntimeError:
        y_idx = np.linspace(0, cropped.shape[0] - 1, target_h).astype(int)
        x_idx = np.linspace(0, cropped.shape[1] - 1, target_w).astype(int)
        return cropped[np.ix_(y_idx, x_idx)]


def _mux_audio(
    source_video_path: str,
    temp_video_path: str,
    output_path: str,
    start_seconds: float,
    end_seconds: float,
) -> bool:
    cmd = [
        ffmpeg_exe(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_seconds:.3f}",
        "-to",
        f"{end_seconds:.3f}",
        "-i",
        str(source_video_path),
        "-i",
        str(temp_video_path),
        "-map",
        "1:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.returncode == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 0


def render_follow_cam_clip(
    video_path: str,
    output_path: str,
    start_seconds: float,
    end_seconds: float,
    player_track: Sequence[TrackSample],
    ball_track: Optional[Sequence[TrackSample]] = None,
    zoom_factor: float = 1.6,
    ball_weight: float = 0.0,
    smooth_factor: float = 0.2,
    include_audio: bool = True,
) -> str:
    start_s = max(0.0, float(start_seconds))
    end_s = float(end_seconds)
    if end_s <= start_s:
        raise ValueError("end_seconds must be greater than start_seconds")

    out_file = Path(output_path)
    ensure_dir(str(out_file.parent))
    temp_file = out_file.with_name(f"{out_file.stem}_temp_video.mp4")

    cv2 = _import_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frame_w <= 0 or frame_h <= 0:
        cap.release()
        raise RuntimeError(f"Could not determine frame size for: {video_path}")

    centers = build_follow_cam_centers(
        player_track=player_track,
        ball_track=ball_track,
        start_seconds=start_s,
        end_seconds=end_s,
        fps=fps,
        frame_size=(frame_w, frame_h),
        zoom_factor=zoom_factor,
        ball_weight=ball_weight,
        smooth_factor=smooth_factor,
    )

    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_file), fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open follow-cam writer for: {temp_file}")

    written = 0
    try:
        for center in centers:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(crop_frame_to_center(frame, center, zoom_factor, output_size=(frame_w, frame_h)))
            written += 1
    finally:
        writer.release()
        cap.release()

    if written <= 0 or not temp_file.exists() or temp_file.stat().st_size <= 0:
        temp_file.unlink(missing_ok=True)
        raise RuntimeError(f"Follow-cam render produced no frames for: {video_path}")

    if include_audio and ffmpeg_available():
        if _mux_audio(video_path, str(temp_file), str(out_file), start_s, end_s):
            temp_file.unlink(missing_ok=True)
            return str(out_file.resolve())

    temp_file.replace(out_file)
    return str(out_file.resolve())
