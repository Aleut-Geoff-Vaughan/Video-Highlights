from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional

from ..config import settings
from .ffmpeg_tools import ffmpeg_exe


AUDIO_EDIT_MODES = {"keep", "remove", "replace", "mix"}
AUDIO_CLEANUP_PROFILES = {"none", "stadium_clean", "wind_reduce", "speech_reduce", "ai_rnnoise"}


def _clamp_volume(value: float) -> float:
    return min(2.0, max(0.0, float(value)))


def _ffmpeg_filter_path(path: str) -> str:
    # ffmpeg filter arguments treat ":" as a separator, so Windows drive letters need escaping.
    return Path(path).resolve().as_posix().replace(":", "\\:")


def _cleanup_filter_chain(cleanup_profile: str) -> List[str]:
    profile = str(cleanup_profile or "none").strip().lower()
    if profile == "none":
        return []
    if profile == "stadium_clean":
        return ["highpass=f=110", "lowpass=f=12500", "afftdn=nf=-25", "dynaudnorm=f=150:g=7"]
    if profile == "wind_reduce":
        return ["highpass=f=180", "afftdn=nf=-30", "dynaudnorm=f=150:g=6"]
    if profile == "speech_reduce":
        return [
            "highpass=f=120",
            "lowpass=f=10000",
            "afftdn=nf=-26",
            "equalizer=f=950:t=q:w=1.1:g=-5",
            "equalizer=f=2500:t=q:w=1.2:g=-7",
            "dynaudnorm=f=150:g=6",
        ]
    if profile == "ai_rnnoise":
        model_path = str(settings.rnnoise_model_path or "").strip()
        if not model_path:
            raise ValueError("AI audio cleanup requires VH_RNNOISE_MODEL_PATH to point to a local RNNoise model file.")
        if not Path(model_path).exists():
            raise ValueError(f"RNNoise model file was not found: {model_path}")
        return [f"arnndn=m={_ffmpeg_filter_path(model_path)}", "dynaudnorm=f=150:g=6"]
    raise ValueError(f"Unsupported cleanup_profile '{cleanup_profile}'.")


def build_audio_edit_command(
    *,
    source_video_path: str,
    output_path: str,
    mode: str,
    cleanup_profile: str = "none",
    external_audio_path: Optional[str] = None,
    original_volume: float = 1.0,
    music_volume: float = 0.35,
    loop_external_audio: bool = True,
) -> List[str]:
    edit_mode = str(mode or "keep").strip().lower()
    if edit_mode not in AUDIO_EDIT_MODES:
        raise ValueError(f"Unsupported audio edit mode '{mode}'.")

    cleanup = str(cleanup_profile or "none").strip().lower()
    if cleanup not in AUDIO_CLEANUP_PROFILES:
        raise ValueError(f"Unsupported cleanup_profile '{cleanup_profile}'.")

    cmd: List[str] = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(source_video_path)]

    if edit_mode in {"replace", "mix"}:
        if not external_audio_path:
            raise ValueError(f"Audio edit mode '{edit_mode}' requires an uploaded MP3/audio file.")
        if loop_external_audio:
            cmd.extend(["-stream_loop", "-1"])
        cmd.extend(["-i", str(external_audio_path)])

    if edit_mode == "remove":
        return cmd + ["-map", "0:v:0", "-c:v", "copy", "-an", "-movflags", "+faststart", str(output_path)]

    original_filters = _cleanup_filter_chain(cleanup)
    original_filters.append(f"volume={_clamp_volume(original_volume):.3f}")
    music_filter = f"volume={_clamp_volume(music_volume):.3f}"

    if edit_mode == "keep":
        cmd.extend(["-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy"])
        if original_filters:
            cmd.extend(["-af", ",".join(original_filters)])
        return cmd + ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output_path)]

    if edit_mode == "replace":
        cmd.extend(["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-af", music_filter])
        return cmd + ["-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(output_path)]

    filter_complex = (
        f"[0:a:0]{','.join(original_filters)}[a0];"
        f"[1:a:0]{music_filter}[a1];"
        "[a0][a1]amix=inputs=2:duration=first:dropout_transition=2[aout]"
    )
    return cmd + [
        "-filter_complex",
        filter_complex,
        "-map",
        "0:v:0",
        "-map",
        "[aout]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def render_audio_edit(
    *,
    source_video_path: str,
    output_path: str,
    mode: str,
    cleanup_profile: str = "none",
    external_audio_path: Optional[str] = None,
    original_volume: float = 1.0,
    music_volume: float = 0.35,
    loop_external_audio: bool = True,
) -> str:
    command = build_audio_edit_command(
        source_video_path=source_video_path,
        output_path=output_path,
        mode=mode,
        cleanup_profile=cleanup_profile,
        external_audio_path=external_audio_path,
        original_volume=original_volume,
        music_volume=music_volume,
        loop_external_audio=loop_external_audio,
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    out_file = Path(output_path)
    if result.returncode != 0 or not out_file.exists() or out_file.stat().st_size <= 0:
        detail = (result.stderr or result.stdout or f"ffmpeg return code {result.returncode}").strip()
        raise RuntimeError(f"Audio edit failed: {detail}")
    return str(out_file.resolve())
