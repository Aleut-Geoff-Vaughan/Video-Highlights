# Performance Improvements

This document summarizes performance-related behavior already implemented in the current local pipeline (`VideoHighlights.py`).

## Implemented Optimizations

### 1. CUDA-Enabled Tracking Path

- Detects CUDA availability automatically
- Uses GPU inference when available
- Enables FP16 inference on CUDA

Expected impact:

- Approximate 2x to 3x tracking speedup versus CPU-only mode

### 2. Parallel Clip Rendering

- Uses `ThreadPoolExecutor` for concurrent clip generation
- Auto-sizes worker count relative to CPU core count
- Handles per-clip failures without aborting full job

Expected impact:

- Faster clip generation for multi-highlight outputs

### 3. NVENC Hardware Encoding Fallback Chain

- Detects `h264_nvenc` at runtime
- Uses NVENC when present, falls back to `libx264` otherwise

Expected impact:

- Significant encoding speedup on compatible NVIDIA GPUs

### 4. Parallel Spotlight Overlay Rendering

- Runs overlay tasks concurrently
- Uses a lower worker profile than clip rendering to reduce memory pressure

Expected impact:

- Higher throughput for overlay-enabled jobs

### 5. Resilience Improvements

- Retries clip writing without audio when audio codec errors occur
- Ensures clip resources are closed/cleaned defensively

Expected impact:

- Better completion rate on noisy or inconsistent source media

## Validation Checklist

```bash
python test_performance.py
nvidia-smi
ffmpeg -hide_banner -encoders
```

Useful runtime indicators:

1. `[performance] Using device: cuda`
2. `[performance] Video encoding: GPU (NVENC)`
3. Parallel worker count logs for clip and overlay stages

## Performance Dependencies

Observed runtime depends strongly on:

1. Source resolution and frame rate
2. Number and duration of generated intervals
3. Disk throughput (SSD/NVMe recommended)
4. CPU cores and memory headroom
5. GPU model and codec support