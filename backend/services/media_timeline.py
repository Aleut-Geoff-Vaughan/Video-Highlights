from __future__ import annotations

import base64
import math
import subprocess
from typing import Any, Dict, List

import numpy as np

from .ffmpeg_tools import ffmpeg_exe


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to generate timeline thumbnails") from exc
    return cv2


def _video_probe(video_path: str) -> Dict[str, Any]:
    cv2 = _import_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for timeline: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    return {
        "fps": fps,
        "frame_count": int(frame_count),
        "duration_seconds": float(duration),
        "width": width,
        "height": height,
    }


def _sample_times(duration_seconds: float, count: int) -> List[float]:
    duration = max(0.0, float(duration_seconds))
    item_count = max(1, min(48, int(count or 18)))
    if duration <= 0.1:
        return [0.0]
    if item_count == 1:
        return [min(duration - 0.05, duration / 2.0)]
    start = min(0.25, duration / 4.0)
    end = max(start, duration - min(0.25, duration / 4.0))
    return [float(value) for value in np.linspace(start, end, item_count)]


def generate_timeline_thumbnails(
    video_path: str,
    *,
    duration_seconds: float,
    count: int = 18,
    width: int = 160,
) -> List[Dict[str, Any]]:
    cv2 = _import_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for timeline thumbnails: {video_path}")

    thumbnails: List[Dict[str, Any]] = []
    target_width = max(80, min(320, int(width or 160)))
    try:
        for t in _sample_times(duration_seconds, count):
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame_h, frame_w = frame.shape[:2]
            if frame_w <= 0 or frame_h <= 0:
                continue
            target_height = max(45, int(round(target_width * (frame_h / frame_w))))
            resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 76])
            if not ok:
                continue
            data = base64.b64encode(encoded.tobytes()).decode("ascii")
            thumbnails.append(
                {
                    "t": round(float(t), 3),
                    "width": target_width,
                    "height": target_height,
                    "data_url": f"data:image/jpeg;base64,{data}",
                }
            )
    finally:
        cap.release()
    return thumbnails


def generate_waveform_peaks(
    video_path: str,
    *,
    bins: int = 96,
    sample_rate: int = 8000,
    duration_seconds: float = 0.0,
) -> Dict[str, Any]:
    target_bins = max(16, min(360, int(bins or 96)))
    timeout = max(30.0, min(240.0, float(duration_seconds or 0.0) * 0.35))
    cmd = [
        ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=False, timeout=timeout)
    except FileNotFoundError:
        return {"peaks": [], "error": "ffmpeg was not found on PATH"}
    except subprocess.TimeoutExpired:
        return {"peaks": [], "error": "Timed out while reading audio waveform"}
    except Exception as exc:
        return {"peaks": [], "error": str(exc)}

    if result.returncode != 0 or not result.stdout:
        detail = (result.stderr or b"").decode("utf-8", errors="ignore").strip()
        return {"peaks": [], "error": detail or "No audio waveform could be decoded"}

    samples = np.frombuffer(result.stdout, dtype=np.float32)
    if samples.size == 0:
        return {"peaks": [], "error": "No audio samples found"}
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)

    chunk_size = int(math.ceil(samples.size / target_bins))
    peaks: List[float] = []
    for index in range(target_bins):
        start = index * chunk_size
        end = min(samples.size, start + chunk_size)
        if start >= samples.size:
            peaks.append(0.0)
            continue
        peaks.append(float(np.max(np.abs(samples[start:end]))))

    max_peak = max(peaks) if peaks else 0.0
    if max_peak > 0.0:
        peaks = [round(min(1.0, peak / max_peak), 4) for peak in peaks]
    return {"peaks": peaks, "error": None}


def build_media_timeline(
    video_path: str,
    *,
    thumbnail_count: int = 18,
    waveform_bins: int = 96,
) -> Dict[str, Any]:
    probe = _video_probe(video_path)
    duration = float(probe.get("duration_seconds") or 0.0)
    thumbnails = generate_timeline_thumbnails(
        video_path,
        duration_seconds=duration,
        count=thumbnail_count,
    )
    waveform = generate_waveform_peaks(
        video_path,
        bins=waveform_bins,
        duration_seconds=duration,
    )
    return {
        "video": probe,
        "thumbnails": thumbnails,
        "waveform": waveform,
    }
