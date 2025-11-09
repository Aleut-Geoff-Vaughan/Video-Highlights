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
  • GPU acceleration requires PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118
"""

import os
import sys
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

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
        # default: longest-lived track
        target_id = max(tracks.keys(), key=lambda k: (tracks[k][-1].t - tracks[k][0].t))

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


def check_nvenc_available() -> bool:
    """Check if NVENC GPU encoding is available"""
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'],
                              capture_output=True, text=True, timeout=5)
        return 'h264_nvenc' in result.stdout
    except:
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

        # Choose codec based on availability
        codec = "h264_nvenc" if _NVENC_AVAILABLE else "libx264"
        codec_params = []
        if _NVENC_AVAILABLE:
            # NVENC parameters for faster encoding
            codec_params = ['-preset', 'fast', '-b:v', '5M']
        else:
            # libx264 parameters for faster encoding
            codec_params = ['-preset', 'faster', '-crf', '23']

        # Try with audio first, fallback to no audio if it fails
        try:
            sub.write_videofile(out_path, codec=codec, audio_codec="aac", logger=None,
                              threads=2, ffmpeg_params=codec_params)
        except (AttributeError, OSError) as audio_err:
            print(f"[warn] Audio processing failed for clip {clip_num}, retrying without audio: {audio_err}")
            sub.write_videofile(out_path, codec=codec, audio=False, logger=None,
                              threads=2, ffmpeg_params=codec_params)
        return out_path
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
    require_gpu: bool = False
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

    Returns:
        True if processing succeeded, False otherwise
    """
    # Check GPU requirement
    if require_gpu:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: GPU acceleration is required but no CUDA-capable GPU was detected.")
            print("Please ensure:")
            print("  1. You have an NVIDIA GPU installed")
            print("  2. CUDA drivers are installed (run 'nvidia-smi' to verify)")
            print("  3. PyTorch with CUDA support is installed:")
            print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            return False

    # Validate paths
    video_path = os.path.abspath(os.path.expanduser(video_path))
    output_dir = os.path.abspath(os.path.expanduser(output_dir))

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return False

    if not os.path.isfile(video_path):
        print(f"Error: Path is not a file: {video_path}")
        return False

    # Print configuration
    print(f"\nProcessing video: {video_path}")
    print(f"Output directory: {output_dir}")
    if trim_start is not None or trim_end is not None:
        print(f"Trim range: {format_time(trim_start or 0)} to {format_time(trim_end) if trim_end else 'end'}")
    print(f"Pre-event buffer: {pre_seconds}s")
    print(f"Post-event buffer: {post_seconds}s")
    print(f"Manual selection: {'Yes' if select_player else 'No'}")
    print(f"Spotlight overlay: {'Yes' if overlay else 'No'}")
    print()

    ensure_dir(output_dir)

    try:
        # Create trimmed video if needed
        original_video = video_path
        processing_video, trim_offset = create_trimmed_video(video_path, output_dir, trim_start, trim_end)

        print("[1/5] Tracking players (YOLO + ByteTrack)...")
        tracks, fps, (W, H) = track_video(processing_video, select_roi=select_player)
        target_id = list(tracks.keys())[0]
        traj = tracks[target_id]

        print("[2/5] Computing speed-based highlights...")
        times, speed = compute_speed_series(traj, fps)
        speed_intervals = detect_highlights_from_speed(times, speed, pre=pre_seconds, post=post_seconds)

        audio_intervals = []
        if not no_audio:
            print("[3/5] Detecting audio peaks...")
            audio_intervals = detect_audio_peaks(processing_video, pre=pre_seconds, post=post_seconds)

        print("[4/5] Merging and pruning intervals...")
        intervals = merge_intervals(speed_intervals + audio_intervals)

        # Get video duration to clamp intervals
        cap_check = cv2.VideoCapture(processing_video)
        video_duration = cap_check.get(cv2.CAP_PROP_FRAME_COUNT) / cap_check.get(cv2.CAP_PROP_FPS) if cap_check.isOpened() else float('inf')
        cap_check.release()

        # Enforce minimum clip length and clamp to video duration
        clamped_intervals = []
        for s, e in intervals:
            if (e - s) < min_clip_duration:
                e = min(s + min_clip_duration, video_duration)
            else:
                e = min(e, video_duration)
            if e > s:
                clamped_intervals.append((s, e))
        intervals = clamped_intervals

        if not intervals:
            print("No highlight intervals found. Try lowering thresholds or ensure --select is used.")
            return False

        # Adjust intervals back to original video timestamps if trimmed
        original_intervals = [(s + trim_offset, e + trim_offset) for s, e in intervals]

        if trim_offset > 0:
            print(f"[info] Found {len(intervals)} highlights. Adjusting timestamps to original video (offset: +{format_time(trim_offset)})")

        print("[5/5] Writing subclips...")
        clip_paths = write_subclips(original_video, original_intervals, output_dir, max_workers=threads)

        # Montage
        if clip_paths:
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

        # Optional overlay rendering
        if overlay:
            print("[overlay] Rendering spotlight overlays (this can take a while)...")
            overlay_workers = min(2, threads) if threads else None
            draw_spotlight_overlay(original_video, traj, original_intervals, output_dir, max_workers=overlay_workers)
            print("[overlay] Done.")

        return True

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
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
        require_gpu=False  # CLI doesn't require GPU by default
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
