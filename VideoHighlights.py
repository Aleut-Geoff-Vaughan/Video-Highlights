"""
Soccer Highlight Agent
----------------------

A no-subscription, local Python pipeline that:
  • Tracks all players with YOLO + ByteTrack
  • Lets you lock onto your child once (interactive box selection)
  • Detects highlight moments from speed/acceleration spikes and audio peaks
  • Exports clean subclips and an optional overlay version with a spotlight circle

Dependencies (install):
    pip install ultralytics==8.* opencv-python numpy tqdm moviepy librosa soundfile

Usage examples:
    python soccer_highlights.py \
        --video /path/to/match.mp4 \
        --out ./highlights_out \
        --select  \
        --pre 2.0 --post 6.0 --min-clip 4.0 --overlay

Notes:
  • --select opens a window on the FIRST frame so you can drag a box over your child. Press ENTER/SPACE to confirm.
  • If you skip --select, the script picks the longest-lived person track (works surprisingly well when your child plays full-time).
  • First run will auto-download YOLO weights.
  • Works best with 1080p/60 or 4K/60 videos recorded from a stable, elevated sideline or halfway-line vantage.
"""

import os
import sys
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
from tqdm import tqdm

# YOLOv8 API
from ultralytics import YOLO

# Audio & video IO
import librosa
try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
except ImportError:
    # moviepy 2.x has different import structure
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips


@dataclass
class TrackPoint:
    t: float  # seconds
    xy: Tuple[float, float]  # center x,y in pixels


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


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


def track_video(video_path: str, fps_hint: Optional[float] = None, select_roi: bool = False) -> Tuple[Dict[int, List[TrackPoint]], float, Tuple[int,int]]:
    """Run YOLO + ByteTrack, return per-ID trajectory, FPS, and frame size.
    Returns: (tracks, fps, (W,H)) where tracks[id] = [TrackPoint, ...]
    """
    # Prepare first frame (for selection)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = fps_hint or cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, first = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read first frame.")
    H, W = first.shape[:2]

    user_box = None  # [x1,y1,x2,y2]
    if select_roi:
        # OpenCV ROI returns (x,y,w,h)
        roi = cv2.selectROI("Select your player", first, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select your player")
        x, y, w, h = roi
        if w > 0 and h > 0:
            user_box = np.array([x, y, x + w, y + h], dtype=np.float32)

    # YOLO tracking (persons + sports ball for potential future use)
    model = YOLO("yolov8n.pt")

    # The stream=True iterator yields per-frame results with .boxes and .boxes.id
    tracks: Dict[int, List[TrackPoint]] = {}

    for frame_idx, result in enumerate(model.track(source=video_path, stream=True, tracker="bytetrack.yaml", classes=[0, 32])):
        # We mostly care about persons (class 0). result.boxes.cls, .id, .xyxy
        if result.boxes is None or result.boxes.id is None:
            continue
        ids = result.boxes.id.cpu().numpy().astype(int)
        cls = result.boxes.cls.cpu().numpy().astype(int)
        xyxy = result.boxes.xyxy.cpu().numpy()

        t = frame_idx / fps
        for idx, (track_id, c) in enumerate(zip(ids, cls)):
            if c != 0:  # person only
                continue
            x1, y1, x2, y2 = xyxy[idx]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            tracks.setdefault(track_id, []).append(TrackPoint(t=t, xy=(float(cx), float(cy))))

    # Choose target ID
    target_id = None
    if user_box is not None:
        # Pick ID whose early boxes overlap the user's box best within first 3 seconds
        best_iou = -1.0
        window_t = 3.0
        for tid, traj in tracks.items():
            # Find the earliest point within window
            early = [p for p in traj if p.t <= window_t]
            if not early:
                continue
            # approximate bbox as a small box around center (fallback if precise boxes unavailable)
            # We'll instead estimate IoU via distance: smaller distance -> higher pseudo IoU
            cxs = np.array([p.xy[0] for p in early])
            cys = np.array([p.xy[1] for p in early])
            cx, cy = np.mean(cxs), np.mean(cys)
            # distance to user box center
            ux = (user_box[0] + user_box[2]) / 2.0
            uy = (user_box[1] + user_box[3]) / 2.0
            dist = math.hypot(cx - ux, cy - uy) + 1e-3
            pseudo_iou = 1.0 / dist
            if pseudo_iou > best_iou:
                best_iou = pseudo_iou
                target_id = tid
    else:
        # default: longest-lived track
        target_id = max(tracks.keys(), key=lambda k: (tracks[k][-1].t - tracks[k][0].t)) if tracks else None

    if target_id is None:
        raise RuntimeError("No player track found. Try using --select on the first frame.")

    return {target_id: tracks[target_id]}, fps, (W, H)


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


def detect_highlights_from_speed(times: np.ndarray, speed: np.ndarray, pre: float, post: float) -> List[Tuple[float, float]]:
    if len(speed) == 0:
        return []
    thr = robust_threshold(speed, k=3.0)
    candidates = np.where(speed >= thr)[0]
    intervals = []
    for idx in candidates:
        t = float(times[idx])
        intervals.append((max(0.0, t - pre), t + post))
    return merge_intervals(intervals)


def detect_audio_peaks(video_path: str, pre: float, post: float) -> List[Tuple[float, float]]:
    try:
        # Load audio at native sampling rate
        y, sr = librosa.load(video_path, sr=None, mono=True)
        # Frame over ~100 ms windows
        hop = int(0.05 * sr)
        win = int(0.10 * sr)
        rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=True).flatten()
        # Map frames to times
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop, n_fft=win)
        thr = robust_threshold(rms, k=3.0)
        peaks = np.where(rms >= thr)[0]
        intervals = [(max(0.0, float(times[i]) - pre), float(times[i]) + post) for i in peaks]
        return merge_intervals(intervals)
    except Exception as e:
        print(f"[warn] audio peak detection failed: {e}")
        return []


def write_subclips(video_path: str, intervals: List[Tuple[float, float]], out_dir: str) -> List[str]:
    paths = []
    with VideoFileClip(video_path) as clip:
        for k, (s, e) in enumerate(intervals, start=1):
            s = max(0.0, s)
            e = min(clip.duration, e)
            if e - s <= 0.25:
                continue
            sub = clip.subclip(s, e)
            out_path = os.path.join(out_dir, f"highlight_{k:02d}.mp4")
            sub.write_videofile(out_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
            paths.append(out_path)
    return paths


def draw_spotlight_overlay(video_path: str, traj: List[TrackPoint], intervals: List[Tuple[float, float]], out_dir: str, radius: int = 35):
    # Index trajectory by time for quick nearest lookup
    t_arr = np.array([p.t for p in traj])
    xy_arr = np.array([p.xy for p in traj])

    def pos_at(t: float) -> Tuple[int, int]:
        # nearest neighbor in time
        idx = int(np.argmin(np.abs(t_arr - t)))
        x, y = xy_arr[idx]
        return int(round(x)), int(round(y))

    for k, (s, e) in enumerate(intervals, start=1):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video during overlay stage.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Seek to start
        cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000.0)

        out_path = os.path.join(out_dir, f"highlight_{k:02d}_spotlight.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            print(f"[warn] Could not open writer for {out_path}, trying alternate codec...")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frames_needed = int((e - s) * fps)
        try:
            for _ in tqdm(range(frames_needed), desc=f"overlay {k}"):
                ok, frame = cap.read()
                if not ok:
                    break
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                cx, cy = pos_at(t)
                # Draw soft circle
                cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 2)
                cv2.circle(frame, (cx, cy), radius + 6, (0, 0, 0), 2)
                writer.write(frame)
        finally:
            writer.release()
            cap.release()


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
    args = ap.parse_args()

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

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    # Ask about player selection if not already set
    if not args.select and sys.stdin.isatty():
        select_input = input("Do you want to manually select your player on the first frame? (y/N): ").strip().lower()
        args.select = select_input in ['y', 'yes']

    # Ask about overlay if not already set
    if not args.overlay and sys.stdin.isatty():
        overlay_input = input("Do you want to render spotlight overlay clips? (slower) (y/N): ").strip().lower()
        args.overlay = overlay_input in ['y', 'yes']

    print(f"\nProcessing video: {args.video}")
    print(f"Output directory: {args.out}")
    print(f"Pre-event buffer: {args.pre}s")
    print(f"Post-event buffer: {args.post}s")
    print(f"Manual selection: {'Yes' if args.select else 'No'}")
    print(f"Spotlight overlay: {'Yes' if args.overlay else 'No'}")
    print()

    ensure_dir(args.out)

    print("[1/5] Tracking players (YOLO + ByteTrack)...")
    tracks, fps, (W, H) = track_video(args.video, select_roi=args.select)
    target_id = list(tracks.keys())[0]
    traj = tracks[target_id]

    print("[2/5] Computing speed-based highlights...")
    times, speed = compute_speed_series(traj, fps)
    speed_intervals = detect_highlights_from_speed(times, speed, pre=args.pre, post=args.post)

    audio_intervals = []
    if not args.no_audio:
        print("[3/5] Detecting audio peaks...")
        audio_intervals = detect_audio_peaks(args.video, pre=args.pre, post=args.post)

    print("[4/5] Merging and pruning intervals...")
    intervals = merge_intervals(speed_intervals + audio_intervals)

    # Enforce minimum clip length
    intervals = [(s, e) if (e - s) >= args.min_clip else (s, s + args.min_clip) for (s, e) in intervals]

    if not intervals:
        print("No highlight intervals found. Try lowering thresholds (edit robust_threshold k) or ensure --select is used.")
        return

    print("[5/5] Writing subclips...")
    clip_paths = write_subclips(args.video, intervals, args.out)

    # Montage
    if clip_paths:
        clips = []
        try:
            clips = [VideoFileClip(p) for p in clip_paths]
            montage = concatenate_videoclips(clips, method="compose")
            montage_path = os.path.join(args.out, "highlights_montage.mp4")
            montage.write_videofile(montage_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
            montage.close()
        finally:
            for c in clips:
                c.close()
        print(f"Wrote {len(clip_paths)} clips and a montage to: {args.out}")

    # Optional overlay rendering
    if args.overlay:
        print("[overlay] Rendering spotlight overlays (this can take a while)...")
        draw_spotlight_overlay(args.video, traj, intervals, args.out)
        print("[overlay] Done.")


if __name__ == "__main__":
    main()
