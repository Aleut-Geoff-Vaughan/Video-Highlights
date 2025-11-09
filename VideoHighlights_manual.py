"""
Modified version of VideoHighlights.py with hardcoded player selection coordinates
This bypasses the interactive cv2.selectROI for headless environments
"""

import os
import sys
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from datetime import datetime

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
    try:
        # moviepy 2.x has different import structure
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips
    except ImportError as e:
        print(f"Error: Could not import moviepy. Please install it with: pip install moviepy")
        print(f"Details: {e}")
        sys.exit(1)


@dataclass
class TrackPoint:
    t: float  # seconds
    xy: Tuple[float, float]  # center x,y in pixels


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


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

    try:
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


def track_video(video_path: str, fps_hint: Optional[float] = None, user_box_coords: Optional[Tuple[int,int,int,int]] = None) -> Tuple[Dict[int, List[TrackPoint]], float, Tuple[int,int]]:
    """Run YOLO + ByteTrack, return per-ID trajectory, FPS, and frame size.
    Returns: (tracks, fps, (W,H)) where tracks[id] = [TrackPoint, ...]
    user_box_coords: (x, y, w, h) for manual player selection
    """
    # Prepare first frame
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
    if user_box_coords is not None:
        x, y, w, h = user_box_coords
        if w > 0 and h > 0:
            user_box = np.array([x, y, x + w, y + h], dtype=np.float32)
            print(f"[manual selection] Using provided coordinates: x={x}, y={y}, w={w}, h={h}")

    # YOLO tracking (persons + sports ball for potential future use)
    model = YOLO("yolov8n.pt")

    # Enable GPU if available and use half precision for faster inference
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[performance] Using device: {device}")
    if device == 'cuda':
        print(f"[performance] GPU: {torch.cuda.get_device_name(0)}")

    # The stream=True iterator yields per-frame results with .boxes and .boxes.id
    tracks: Dict[int, List[TrackPoint]] = {}

    for frame_idx, result in enumerate(model.track(source=video_path, stream=True, tracker="bytetrack.yaml", classes=[0, 32], device=device, half=True if device == 'cuda' else False, verbose=False)):
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

    if not tracks:
        raise RuntimeError("No player tracks detected in video. Ensure the video contains visible people.")

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

        if target_id is None:
            # Fallback if no tracks match the user selection
            print("[warn] No tracks matched your selection. Using longest-lived track instead.")
            target_id = max(tracks.keys(), key=lambda k: (tracks[k][-1].t - tracks[k][0].t))
        else:
            print(f"[manual selection] Matched player to track ID: {target_id}")
    else:
        # default: longest-lived track
        target_id = max(tracks.keys(), key=lambda k: (tracks[k][-1].t - tracks[k][0].t))

    if target_id is None:
        raise RuntimeError("No player track found.")

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
    clip = None
    try:
        clip = VideoFileClip(video_path)
        for k, (s, e) in enumerate(intervals, start=1):
            s = max(0.0, s)
            e = min(clip.duration, e)
            if e - s <= 0.25:
                continue
            sub = None
            try:
                # Try both subclip and subclipped (different moviepy versions)
                try:
                    sub = clip.subclip(s, e)
                except AttributeError:
                    sub = clip.subclipped(s, e)

                out_path = os.path.join(out_dir, f"highlight_{k:02d}.mp4")
                # Try with audio first, fallback to no audio if it fails
                try:
                    sub.write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None)
                except (AttributeError, OSError) as audio_err:
                    print(f"[warn] Audio processing failed for clip {k}, retrying without audio: {audio_err}")
                    sub.write_videofile(out_path, codec="libx264", audio=False, logger=None)
                paths.append(out_path)
            except Exception as ex:
                print(f"[warn] Failed to write clip {k} ({s:.1f}s - {e:.1f}s): {ex}")
            finally:
                if sub is not None:
                    sub.close()
    finally:
        if clip is not None:
            clip.close()
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
        seek_success = cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000.0)
        if not seek_success:
            print(f"[warn] Failed to seek to {s}s for overlay {k}, starting from beginning")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        temp_path = os.path.join(out_dir, f"highlight_{k:02d}_spotlight_temp.mp4")
        out_path = os.path.join(out_dir, f"highlight_{k:02d}_spotlight.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            print(f"[warn] Could not open writer for {temp_path}, trying alternate codec...")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

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

        # Add audio using moviepy
        try:
            with VideoFileClip(video_path) as source_clip:
                with VideoFileClip(temp_path) as video_only:
                    # Extract audio from the same time interval (handle different moviepy versions)
                    try:
                        audio_subclip = source_clip.subclip(s, e)
                    except AttributeError:
                        audio_subclip = source_clip.subclipped(s, e)
                    audio_clip = audio_subclip.audio
                    if audio_clip is not None:
                        final_clip = video_only.set_audio(audio_clip)
                        final_clip.write_videofile(out_path, codec="libx264", audio_codec="aac")
                        final_clip.close()
                    else:
                        # No audio in source, just rename temp file
                        os.rename(temp_path, out_path)
            # Clean up temp file if it still exists
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            print(f"[warn] Could not add audio to overlay {k}: {e}. Using video-only version.")
            if os.path.exists(temp_path):
                os.rename(temp_path, out_path)


def main():
    # Setup timestamped log directory
    ensure_dir("logs")
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/run_{run_timestamp}.log"

    # Redirect stdout and stderr to log file while still printing to console
    import sys
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    log_fp = open(log_file, 'w')
    sys.stdout = TeeOutput(sys.stdout, log_fp)
    sys.stderr = TeeOutput(sys.stderr, log_fp)

    print(f"[log] Run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[log] Log file: {log_file}")

    ap = argparse.ArgumentParser(description="Soccer highlight generator with manual coordinate selection")
    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument("--out", required=True, help="Output directory for highlights")
    ap.add_argument("--box", type=str, help="Manual player box as 'x,y,w,h' (e.g., '100,200,50,100')")
    ap.add_argument("--pre", type=float, default=2.0, help="Seconds before event")
    ap.add_argument("--post", type=float, default=6.0, help="Seconds after event")
    ap.add_argument("--min-clip", type=float, default=4.0, help="Minimum clip duration (after merging)")
    ap.add_argument("--no-audio", action="store_true", help="Disable audio-based peak detection")
    ap.add_argument("--overlay", action="store_true", help="Render spotlight overlay clips (slower)")
    ap.add_argument("--trim-start", type=str, help="Trim video start time (format: MM:SS or HH:MM:SS or seconds)")
    ap.add_argument("--trim-end", type=str, help="Trim video end time (format: MM:SS or HH:MM:SS or seconds)")
    args = ap.parse_args()

    # Parse manual box coordinates if provided
    user_box_coords = None
    if args.box:
        try:
            coords = [int(c.strip()) for c in args.box.split(',')]
            if len(coords) != 4:
                raise ValueError("Box must have 4 values: x,y,w,h")
            user_box_coords = tuple(coords)
            print(f"Using manual player selection: {user_box_coords}")
        except Exception as e:
            print(f"Error parsing --box argument: {e}")
            sys.exit(1)

    # Validate and normalize paths
    args.video = os.path.abspath(os.path.expanduser(args.video))
    args.out = os.path.abspath(os.path.expanduser(args.out))

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    # Parse trim times
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

    print(f"\nProcessing video: {args.video}")
    print(f"Output directory: {args.out}")
    if trim_start_seconds is not None or trim_end_seconds is not None:
        print(f"Trim range: {format_time(trim_start_seconds or 0)} to {format_time(trim_end_seconds) if trim_end_seconds else 'end'}")
    print(f"Pre-event buffer: {args.pre}s")
    print(f"Post-event buffer: {args.post}s")
    print(f"Manual selection: {'Yes' if user_box_coords else 'No (auto-detect longest track)'}")
    print(f"Spotlight overlay: {'Yes' if args.overlay else 'No'}")
    print()

    ensure_dir(args.out)

    # Create trimmed video if needed
    original_video = args.video
    processing_video, trim_offset = create_trimmed_video(args.video, args.out, trim_start_seconds, trim_end_seconds)

    print("[1/5] Tracking players (YOLO + ByteTrack)...")
    tracks, fps, (W, H) = track_video(processing_video, user_box_coords=user_box_coords)
    target_id = list(tracks.keys())[0]
    traj = tracks[target_id]

    print("[2/5] Computing speed-based highlights...")
    times, speed = compute_speed_series(traj, fps)
    speed_intervals = detect_highlights_from_speed(times, speed, pre=args.pre, post=args.post)

    audio_intervals = []
    if not args.no_audio:
        print("[3/5] Detecting audio peaks...")
        audio_intervals = detect_audio_peaks(processing_video, pre=args.pre, post=args.post)

    print("[4/5] Merging and pruning intervals...")
    intervals = merge_intervals(speed_intervals + audio_intervals)

    # Get video duration to clamp intervals (use processing video duration)
    cap_check = cv2.VideoCapture(processing_video)
    video_duration = cap_check.get(cv2.CAP_PROP_FRAME_COUNT) / cap_check.get(cv2.CAP_PROP_FPS) if cap_check.isOpened() else float('inf')
    cap_check.release()

    # Enforce minimum clip length and clamp to video duration
    clamped_intervals = []
    for s, e in intervals:
        if (e - s) < args.min_clip:
            e = min(s + args.min_clip, video_duration)
        else:
            e = min(e, video_duration)
        if e > s:  # Only add valid intervals
            clamped_intervals.append((s, e))
    intervals = clamped_intervals

    if not intervals:
        print("No highlight intervals found. Try lowering thresholds (edit robust_threshold k).")
        return

    # Adjust intervals back to original video timestamps if trimmed
    original_intervals = [(s + trim_offset, e + trim_offset) for s, e in intervals]

    if trim_offset > 0:
        print(f"[info] Found {len(intervals)} highlights. Adjusting timestamps to original video (offset: +{format_time(trim_offset)})")

    print("[5/5] Writing subclips...")
    clip_paths = write_subclips(original_video, original_intervals, args.out)

    # Montage
    if clip_paths:
        clips = []
        try:
            clips = [VideoFileClip(p) for p in clip_paths]
            montage = concatenate_videoclips(clips, method="compose")
            montage_path = os.path.join(args.out, "highlights_montage.mp4")
            montage.write_videofile(montage_path, codec="libx264", audio_codec="aac")
            montage.close()
        finally:
            for c in clips:
                c.close()
        print(f"Wrote {len(clip_paths)} clips and a montage to: {args.out}")

    # Optional overlay rendering
    if args.overlay:
        print("[overlay] Rendering spotlight overlays (this can take a while)...")
        draw_spotlight_overlay(original_video, traj, original_intervals, args.out)
        print("[overlay] Done.")


if __name__ == "__main__":
    main()
