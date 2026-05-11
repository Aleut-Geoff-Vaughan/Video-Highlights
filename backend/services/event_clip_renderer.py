from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

from .ffmpeg_tools import ffmpeg_exe
from ..utils import ensure_dir


def _ffmpeg_encoder_available(encoder_name: str) -> bool:
    try:
        result = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=6,
            check=False,
        )
    except Exception:
        return False

    output = f"{result.stdout}\n{result.stderr}".lower()
    token = encoder_name.strip().lower()
    return f" {token}" in output or f"{token} " in output


def _codec_attempts(prefer_gpu: bool) -> List[Tuple[str, List[str]]]:
    attempts: List[Tuple[str, List[str]]] = []
    if prefer_gpu and _ffmpeg_encoder_available("h264_nvenc"):
        attempts.append(("h264_nvenc", ["-preset", "fast", "-b:v", "5M"]))
    attempts.append(("libx264", ["-preset", "veryfast", "-crf", "22"]))
    attempts.append(("mpeg4", ["-q:v", "3"]))
    return attempts


def render_clip_ffmpeg(
    video_path: str,
    output_path: str,
    start_seconds: float,
    end_seconds: float,
    include_audio: bool = True,
    prefer_gpu: bool = True,
) -> str:
    start_s = max(0.0, float(start_seconds))
    end_s = float(end_seconds)
    if end_s <= start_s:
        raise ValueError("end_seconds must be greater than start_seconds")

    out_file = Path(output_path)
    ensure_dir(str(out_file.parent))

    base_cmd = [
        ffmpeg_exe(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-ss",
        f"{start_s:.3f}",
        "-to",
        f"{end_s:.3f}",
        "-movflags",
        "+faststart",
    ]
    audio_args = ["-c:a", "aac"] if include_audio else ["-an"]

    last_error = ""
    for codec, codec_args in _codec_attempts(prefer_gpu=prefer_gpu):
        cmd = base_cmd + ["-c:v", codec] + codec_args + audio_args + [str(out_file)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0 and out_file.exists() and out_file.stat().st_size > 0:
            return str(out_file.resolve())
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or f"ffmpeg return code {result.returncode}"
        last_error = f"codec={codec} detail={detail}"

    raise RuntimeError(f"Failed to render clip with ffmpeg ({last_error})")


def concat_clips_ffmpeg(
    clip_paths: List[str],
    output_path: str,
    include_audio: bool = True,
) -> str:
    if not clip_paths:
        raise ValueError("clip_paths must not be empty")

    out_file = Path(output_path)
    ensure_dir(str(out_file.parent))

    # ffmpeg concat demuxer requires a text file.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        list_path = Path(handle.name)
        for path in clip_paths:
            escaped = str(Path(path).resolve()).replace("'", r"'\''")
            handle.write(f"file '{escaped}'\n")

    try:
        cmd = [
            ffmpeg_exe(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
        ]
        if include_audio:
            cmd += ["-c:a", "aac"]
        else:
            cmd += ["-an"]
        cmd += ["-movflags", "+faststart", str(out_file)]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not out_file.exists() or out_file.stat().st_size <= 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            detail = stderr or stdout or f"ffmpeg return code {result.returncode}"
            raise RuntimeError(f"Failed to concat clips ({detail})")
        return str(out_file.resolve())
    finally:
        try:
            list_path.unlink(missing_ok=True)
        except Exception:
            pass
