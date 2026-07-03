"""Render videos from a :class:`CameraPlan`.

Two output styles:

* **Follow render** - crop/zoom every frame to the planned camera center with
  per-frame zoom (unlike the legacy follow-cam path, zoom is dynamic), with an
  optional thin status banner explaining the current camera decision.
* **Debug render** - keep the wide frame and draw the full story on top of it:
  the crop rectangle, the camera center crosshair, the ball with a motion
  trail, the estimated field and goal boxes, and a banner stating the game
  state and the reason for the current camera decision. This is the "show me
  the center of the game and why" video used for review and training.

Encoding reuses the follow-cam ffmpeg pipe (NVENC -> libx264 -> OpenCV
fallback) and the audio mux helper.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils import ensure_dir
from .ffmpeg_tools import ffmpeg_available, ffmpeg_exe
from .follow_cam import (
    RenderProgressCallback,
    _ffmpeg_encoder_available,
    _import_cv2,
    _mux_audio,
    crop_frame_to_center,
)
from .camera_planner import CameraDecision, CameraPlan
from .game_tracking import FieldGeometry, GOAL_STATES, RESTART_STATES

LOGGER = logging.getLogger("videohighlights.camera_render")

FrameDecorator = Callable[[np.ndarray, CameraDecision], np.ndarray]

_STATE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "in_play": (80, 200, 80),
    "ball_lost": (60, 160, 230),
    "restart_left": (0, 170, 255),
    "restart_right": (0, 170, 255),
    "restart_touchline": (0, 170, 255),
    "goal_left": (60, 60, 240),
    "goal_right": (60, 60, 240),
}

_BALL_TRAIL_SECONDS = 1.5


def _state_color(state: str) -> Tuple[int, int, int]:
    return _STATE_COLORS.get(state, (200, 200, 200))


def _format_clock(t: float) -> str:
    minutes = int(t // 60)
    seconds = t - minutes * 60
    return f"{minutes:02d}:{seconds:04.1f}"


def annotate_wide_frame(
    frame: np.ndarray,
    decision: CameraDecision,
    *,
    ball_trail: Optional[Sequence[Tuple[float, float]]] = None,
    geometry: Optional[FieldGeometry] = None,
    player_marker: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Draw the camera decision explanation onto a wide frame (in place)."""
    cv2 = _import_cv2()
    frame_h, frame_w = frame.shape[:2]
    color = _state_color(decision.state)
    thickness = max(1, frame_w // 640)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, frame_w / 2400.0)

    # Field bounds and goal boxes.
    if geometry is not None:
        cv2.rectangle(
            frame,
            (int(geometry.x_min), int(geometry.y_min)),
            (int(geometry.x_max), int(geometry.y_max)),
            (128, 128, 128),
            thickness,
        )
        for goal in (geometry.left_goal, geometry.right_goal):
            goal_color = (255, 160, 0)
            if decision.state in RESTART_STATES.union(GOAL_STATES) and decision.focus == f"goal_{goal.side}":
                goal_color = (0, 80, 255)
            cv2.rectangle(frame, (int(goal.x1), int(goal.y1)), (int(goal.x2), int(goal.y2)),
                          goal_color, thickness + 1)

    # Crop rectangle: what the follow camera would show.
    crop_w = frame_w / max(1.0, decision.zoom)
    crop_h = frame_h / max(1.0, decision.zoom)
    x1 = int(round(decision.center_x - crop_w / 2.0))
    y1 = int(round(decision.center_y - crop_h / 2.0))
    cv2.rectangle(frame, (x1, y1), (int(x1 + crop_w), int(y1 + crop_h)), color, thickness + 1)

    # Camera center crosshair ("the center of the game").
    cx, cy = int(round(decision.center_x)), int(round(decision.center_y))
    arm = max(10, frame_w // 96)
    cv2.line(frame, (cx - arm, cy), (cx + arm, cy), (255, 0, 255), thickness + 1)
    cv2.line(frame, (cx, cy - arm), (cx, cy + arm), (255, 0, 255), thickness + 1)

    # Raw target before smoothing (small dot) helps tuning.
    if decision.target_x is not None and decision.target_y is not None:
        cv2.circle(frame, (int(decision.target_x), int(decision.target_y)),
                   max(3, thickness * 2), (255, 255, 0), -1)

    # Ball trail + marker.
    if ball_trail and len(ball_trail) >= 2:
        points = np.asarray(ball_trail, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 255), thickness=thickness)
    if decision.ball_x is not None and decision.ball_y is not None:
        bx, by = int(round(decision.ball_x)), int(round(decision.ball_y))
        radius = max(6, frame_w // 160)
        if decision.ball_source == "detected":
            cv2.circle(frame, (bx, by), radius, (0, 255, 255), -1)
        else:
            cv2.circle(frame, (bx, by), radius, (0, 255, 255), thickness + 1)

    if player_marker is not None:
        px, py = int(round(player_marker[0])), int(round(player_marker[1]))
        cv2.drawMarker(frame, (px, py), (255, 128, 0), cv2.MARKER_TRIANGLE_UP,
                       max(12, frame_w // 120), thickness + 1)

    # Banner with state + reason ("why").
    banner_h = max(34, int(frame_h * 0.075))
    overlay = frame[0:banner_h, :].copy()
    overlay[:] = (20, 20, 20)
    frame[0:banner_h, :] = cv2.addWeighted(overlay, 0.65, frame[0:banner_h, :], 0.35, 0)
    line1 = (
        f"t={_format_clock(decision.t)}  state={decision.state.upper()}  "
        f"focus={decision.focus}  zoom={decision.zoom:.2f}x  conf={decision.confidence:.2f}"
    )
    line2 = f"why: {decision.reason}"
    cv2.putText(frame, line1, (10, int(banner_h * 0.42)), font, font_scale, color, thickness)
    cv2.putText(frame, line2, (10, int(banner_h * 0.85)), font, font_scale, (235, 235, 235), thickness)
    if decision.state in GOAL_STATES:
        cv2.putText(frame, "GOAL!", (frame_w - int(180 * font_scale * 2), int(banner_h * 0.7)),
                    font, font_scale * 1.8, (60, 60, 240), thickness + 2)
    return frame


def annotate_zoomed_banner(frame: np.ndarray, decision: CameraDecision) -> np.ndarray:
    """Small status strip at the bottom of a follow-camera frame."""
    cv2 = _import_cv2()
    frame_h, frame_w = frame.shape[:2]
    banner_h = max(22, int(frame_h * 0.05))
    y0 = frame_h - banner_h
    overlay = frame[y0:frame_h, :].copy()
    overlay[:] = (20, 20, 20)
    frame[y0:frame_h, :] = cv2.addWeighted(overlay, 0.6, frame[y0:frame_h, :], 0.4, 0)
    font_scale = max(0.4, frame_w / 2600.0)
    text = f"{_format_clock(decision.t)} | {decision.state} | {decision.reason}"
    cv2.putText(frame, text[:160], (8, frame_h - int(banner_h * 0.3)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, _state_color(decision.state),
                max(1, frame_w // 800))
    return frame


def _open_ffmpeg_writer(output_path: Path, frame_size: Tuple[int, int], fps: float,
                        encoder: str) -> subprocess.Popen:
    frame_w, frame_h = frame_size
    if encoder == "h264_nvenc":
        encoder_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
    else:
        encoder_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22"]
    cmd = [
        ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{frame_w}x{frame_h}", "-r", f"{fps:.6f}",
        "-i", "pipe:0", "-an", *encoder_args,
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)


def _iter_encoders() -> List[str]:
    encoders = []
    for encoder in ("h264_nvenc", "libx264"):
        if _ffmpeg_encoder_available(encoder):
            encoders.append(encoder)
    return encoders


def render_camera_plan_video(
    *,
    video_path: str,
    output_path: str,
    plan: CameraPlan,
    include_audio: bool = True,
    debug_wide: bool = False,
    overlay_banner: bool = False,
    geometry: Optional[FieldGeometry] = None,
    max_debug_width: int = 1280,
    progress_callback: Optional[RenderProgressCallback] = None,
) -> str:
    """Render the plan to a video file.

    ``debug_wide=False`` produces the polished follow camera (per-frame crop +
    zoom). ``debug_wide=True`` produces the annotated wide review video.
    """
    cv2 = _import_cv2()
    out_file = Path(output_path)
    ensure_dir(str(out_file.parent))
    temp_file = out_file.with_name(f"{out_file.stem}_temp_video.mp4")

    frame_w, frame_h = plan.frame_size
    if debug_wide and frame_w > max_debug_width:
        scale = max_debug_width / frame_w
        out_w = max_debug_width
        out_h = int(round(frame_h * scale))
    else:
        out_w, out_h = frame_w, frame_h
    # Even dimensions keep yuv420p encoders happy.
    out_w -= out_w % 2
    out_h -= out_h % 2

    trail_frames = int(_BALL_TRAIL_SECONDS * plan.fps)

    def _decorated_frames(cap):
        trail: List[Tuple[float, float]] = []
        for decision in plan.decisions:
            ok, frame = cap.read()
            if not ok:
                break
            if decision.ball_x is not None and decision.ball_y is not None:
                trail.append((decision.ball_x, decision.ball_y))
            if len(trail) > trail_frames:
                trail[:] = trail[-trail_frames:]
            if debug_wide:
                annotate_wide_frame(frame, decision, ball_trail=trail, geometry=geometry)
                if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            else:
                frame = crop_frame_to_center(frame, (decision.center_x, decision.center_y),
                                             decision.zoom, output_size=(out_w, out_h))
                if overlay_banner:
                    annotate_zoomed_banner(frame, decision)
            yield frame

    def _open_capture():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        if plan.start_seconds > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, plan.start_seconds * 1000.0)
        return cap

    total_frames = len(plan.decisions)
    written = 0
    encoder_used = ""

    for encoder in _iter_encoders():
        cap = _open_capture()
        temp_file.unlink(missing_ok=True)
        process = _open_ffmpeg_writer(temp_file, (out_w, out_h), plan.fps, encoder)
        written = 0
        last_emit = 0.0
        try:
            for frame in _decorated_frames(cap):
                if process.stdin is None:
                    break
                process.stdin.write(frame.tobytes())
                written += 1
                if progress_callback is not None:
                    now = time.monotonic()
                    if written == 1 or written >= total_frames or (now - last_emit) >= 2.0:
                        progress_callback(written, total_frames)
                        last_emit = now
        except BrokenPipeError:
            pass
        except Exception:
            process.kill()
            cap.release()
            raise
        finally:
            cap.release()
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
        stderr = process.stderr.read().decode("utf-8", errors="ignore") if process.stderr else ""
        code = process.wait()
        if code == 0 and written > 0 and temp_file.exists() and temp_file.stat().st_size > 0:
            encoder_used = encoder
            break
        LOGGER.warning("camera-plan encode with %s failed (%s); trying next encoder",
                       encoder, stderr.strip()[:400])
    else:
        # OpenCV writer fallback.
        cap = _open_capture()
        temp_file.unlink(missing_ok=True)
        writer = cv2.VideoWriter(str(temp_file), cv2.VideoWriter_fourcc(*"mp4v"),
                                 plan.fps, (out_w, out_h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open video writer for: {temp_file}")
        written = 0
        last_emit = 0.0
        try:
            for frame in _decorated_frames(cap):
                writer.write(frame)
                written += 1
                if progress_callback is not None:
                    now = time.monotonic()
                    if written == 1 or written >= total_frames or (now - last_emit) >= 2.0:
                        progress_callback(written, total_frames)
                        last_emit = now
        finally:
            writer.release()
            cap.release()
        encoder_used = "mp4v"

    if written <= 0 or not temp_file.exists() or temp_file.stat().st_size <= 0:
        temp_file.unlink(missing_ok=True)
        raise RuntimeError(f"Camera-plan render produced no frames for: {video_path}")

    LOGGER.info("camera-plan render: %d/%d frames with %s -> %s",
                written, total_frames, encoder_used, out_file.name)

    end_seconds = plan.start_seconds + written / plan.fps
    if include_audio and ffmpeg_available():
        if _mux_audio(str(video_path), str(temp_file), str(out_file), plan.start_seconds, end_seconds):
            temp_file.unlink(missing_ok=True)
            return str(out_file.resolve())

    temp_file.replace(out_file)
    return str(out_file.resolve())
