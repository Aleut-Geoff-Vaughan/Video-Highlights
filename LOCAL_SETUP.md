# Local Setup Guide - GPU Acceleration

This guide will help you run the Video Highlights generator on your local machine with NVIDIA GPU acceleration for significantly faster processing.

## Prerequisites

1. **NVIDIA GPU** - Compatible NVIDIA GPU (GTX 10-series or newer recommended)
2. **NVIDIA Drivers** - Latest NVIDIA drivers installed ([Download here](https://www.nvidia.com/Download/index.aspx))
3. **Python 3.8+** - Python 3.8 or higher
4. **Git** - For cloning the repository

## Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd Video-Highlights
```

## Step 2: Create Virtual Environment

### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 3: Install CUDA Toolkit (if not already installed)

Download and install CUDA Toolkit 12.x from:
https://developer.nvidia.com/cuda-downloads

**Verify CUDA installation:**
```bash
nvcc --version
```

## Step 4: Install PyTorch with CUDA Support

```bash
# For CUDA 12.x (most modern GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.x (older GPUs)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify PyTorch can see your GPU:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3080 (or your GPU model)
```

## Step 5: Install Other Dependencies

```bash
pip install -r requirements.txt
```

## Step 6: Verify GPU Acceleration

```bash
nvidia-smi
```

This should show your GPU and its current usage.

## Running with GPU

### Basic Usage:
```bash
python VideoHighlights_manual.py --video path/to/your/video.mp4 --out ./output --box x,y,w,h
```

### With Overlay (Slower):
```bash
python VideoHighlights_manual.py --video path/to/your/video.mp4 --out ./output --box x,y,w,h --overlay
```

### Full Options:
```bash
python VideoHighlights_manual.py \
    --video path/to/your/video.mp4 \
    --out ./output \
    --box 615,470,40,80 \
    --pre 2.0 \
    --post 6.0 \
    --overlay
```

## Performance Expectations

### With NVIDIA GPU:
- **RTX 3080/3090**: ~2-3x faster than CPU (15-20 minutes for 10-minute video)
- **RTX 4080/4090**: ~3-4x faster than CPU (10-15 minutes for 10-minute video)
- **GTX 1080 Ti**: ~1.5-2x faster than CPU (20-25 minutes for 10-minute video)

### Processing Steps:
1. **Player Tracking** (slowest) - YOLO + ByteTrack - GPU helps most here
2. **Speed Analysis** (fast) - CPU-based calculations
3. **Audio Detection** (medium) - CPU-based audio analysis
4. **Clip Writing** (medium) - Video encoding
5. **Overlay Rendering** (slow if enabled) - Frame-by-frame processing

## Troubleshooting

### GPU Not Detected

If the script shows "Using device: cpu" instead of "cuda":

1. **Check CUDA Installation:**
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```

2. **Reinstall PyTorch with CUDA:**
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Update NVIDIA Drivers:**
   - Download latest from https://www.nvidia.com/Download/index.aspx

### Out of Memory Errors

If you get CUDA out of memory errors:

1. Close other GPU-intensive applications
2. Use a smaller batch size (the script already uses optimal settings)
3. Process shorter video segments using `--trim-start` and `--trim-end`

### Example - Process only second half:
```bash
python VideoHighlights_manual.py \
    --video video.mp4 \
    --out ./output \
    --box 615,470,40,80 \
    --trim-start 45:00 \
    --trim-end 1:30:00
```

## Logs

All runs create timestamped log files in the `logs/` directory:
- Format: `logs/run_YYYYMMDD_HHMMSS.log`
- Contains full output including performance metrics and any errors

## Getting Player Coordinates

If you need to find the player coordinates (x, y, width, height):

1. **Extract first frame:**
   ```bash
   python -c "import cv2; cap = cv2.VideoCapture('video.mp4'); ret, frame = cap.read(); cv2.imwrite('first_frame.jpg', frame)"
   ```

2. **Open in image viewer** and note the approximate pixel coordinates of the player
   - x, y = top-left corner of box around player
   - w, h = width and height of box (typically 40-60 pixels wide, 80-120 tall)

3. **Or use the GUI version** on your local machine:
   ```bash
   python VideoHighlights.py --video video.mp4 --select --out ./output
   ```
   This will let you draw a box interactively.

## Alternative: Run Original Script with Interactive Selection

If you prefer the interactive selection (requires display):

```bash
python VideoHighlights.py --video video.mp4 --select --out ./output --overlay
```

This will open a window where you can draw a box around your player.

## Support

For issues or questions, please check the main README.md or create an issue on GitHub.
