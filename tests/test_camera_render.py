from __future__ import annotations

import numpy as np
import pytest

from backend.services.camera_planner import CameraDecision, CameraPlan
from backend.services.camera_render import (
    annotate_wide_frame,
    annotate_zoomed_banner,
    render_camera_plan_video,
)
from backend.services.game_tracking import estimate_field_geometry

cv2 = pytest.importorskip("cv2")

WIDTH, HEIGHT = 128, 96
FPS = 5.0


def _write_source_video(path, frames=15):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened()
    try:
        for index in range(frames):
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            # Keep the ball inside the legal crop-center region for zoom=2.0
            # (x in [32, 96]) so the planned center is not clamped away.
            x = 40 + index * 3
            cv2.circle(frame, (x, HEIGHT // 2), 4, (0, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()


def _plan(frames=15, zoom=2.0, debug_state="in_play"):
    plan = CameraPlan(start_seconds=0.0, fps=FPS, frame_size=(WIDTH, HEIGHT), base_zoom=zoom)
    for index in range(frames):
        t = index / FPS
        x = 40.0 + index * 3.0
        plan.decisions.append(
            CameraDecision(
                index=index,
                t=t,
                center_x=x,
                center_y=HEIGHT / 2.0,
                zoom=zoom,
                state=debug_state,
                focus="ball",
                reason="following ball",
                confidence=0.9,
                ball_x=x,
                ball_y=HEIGHT / 2.0,
                ball_source="detected",
                target_x=x,
                target_y=HEIGHT / 2.0,
            )
        )
    return plan


def test_render_camera_plan_video_zoomed_output(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "zoomed.mp4"
    _write_source_video(source)

    rendered = render_camera_plan_video(
        video_path=str(source),
        output_path=str(output),
        plan=_plan(),
        include_audio=False,
    )

    assert rendered == str(output.resolve())
    cap = cv2.VideoCapture(str(output))
    assert cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    cap.release()
    assert ok
    assert frame_count >= 14
    # Ball was centered by the crop: bright pixels near frame center.
    center = first[HEIGHT // 2 - 8 : HEIGHT // 2 + 8, WIDTH // 2 - 8 : WIDTH // 2 + 8]
    assert float(center[:, :, 1].mean()) > 20.0


def test_render_camera_plan_video_debug_wide(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "debug.mp4"
    _write_source_video(source)
    geometry = estimate_field_geometry(None, (WIDTH, HEIGHT))

    rendered = render_camera_plan_video(
        video_path=str(source),
        output_path=str(output),
        plan=_plan(debug_state="restart_right"),
        include_audio=False,
        debug_wide=True,
        geometry=geometry,
    )

    assert rendered == str(output.resolve())
    cap = cv2.VideoCapture(str(output))
    ok, first = cap.read()
    cap.release()
    assert ok
    # The banner darkens/annotates the top strip; it must not be all black
    # and must differ from the raw source frame (which was black up top).
    banner = first[0:10, :]
    assert float(banner.mean()) > 1.0


def test_annotate_wide_frame_draws_overlay_and_banner() -> None:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    geometry = estimate_field_geometry(None, (WIDTH, HEIGHT))
    decision = _plan(frames=1).decisions[0]

    out = annotate_wide_frame(
        frame,
        decision,
        ball_trail=[(5.0, 40.0), (10.0, 45.0), (15.0, 48.0)],
        geometry=geometry,
    )

    assert out is frame
    assert float(frame.mean()) > 0.5  # something was drawn
    # Crosshair arm at the camera center in magenta (sample just outside the
    # yellow ball/target markers that sit on the exact center pixel).
    cx, cy = int(decision.center_x), int(decision.center_y)
    assert frame[cy, cx + 8, 0] > 100 and frame[cy, cx + 8, 2] > 100


def test_annotate_zoomed_banner_writes_reason_strip() -> None:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    decision = _plan(frames=1).decisions[0]
    annotate_zoomed_banner(frame, decision)
    assert float(frame[-int(HEIGHT * 0.05) :, :].mean()) > 0.5


def test_render_fails_cleanly_on_missing_video(tmp_path) -> None:
    with pytest.raises(RuntimeError):
        render_camera_plan_video(
            video_path=str(tmp_path / "missing.mp4"),
            output_path=str(tmp_path / "out.mp4"),
            plan=_plan(),
            include_audio=False,
        )
