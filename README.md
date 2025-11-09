# Video-Highlights

A no-subscription, local Python pipeline for generating soccer highlight reels from game footage.

## Features

- **Player Tracking**: Uses YOLO + ByteTrack to track all players
- **Smart Player Selection**: Lock onto your player with interactive box selection or auto-detect longest-playing player
- **Intelligent Highlight Detection**: Finds exciting moments using:
  - Speed/acceleration spikes from player movement
  - Audio peaks (crowd reactions, whistles, etc.)
- **Video Trimming**: Process only specific portions of long game videos
- **Clean Output**: Exports highlight clips and optional spotlight overlay versions
- **GUI & Command Line**: Choose between a user-friendly GUI or powerful command-line interface

## Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for video processing)

### Quick Start with Auto-Install

The easiest way to get started is to use the provided launcher scripts, which will automatically check for and install dependencies:

**Windows:**
```bash
run_gui.bat
```

**Linux/Mac:**
```bash
chmod +x run_gui.sh
./run_gui.sh
```

These launchers will:
- Check if Python is installed
- Detect missing dependencies and offer to install them
- Optionally install GPU acceleration (PyTorch with CUDA)
- Launch the GUI once everything is ready

### Manual Install Dependencies

If you prefer to install dependencies manually:

```bash
pip install -r requirements.txt
```

### Optional: GPU Acceleration (2-3x faster)

For significantly faster processing with NVIDIA GPUs:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Requirements:**
- NVIDIA GPU with CUDA support
- CUDA drivers installed (verify with `nvidia-smi`)
- 4GB+ VRAM recommended

## Usage

### Option 1: Web Interface (Recommended)

Launch the web-based interface in your browser:

```bash
streamlit run app.py
```

Or use the launcher:
```bash
./run_web.sh
```

The web interface offers:
- 🌐 Works in any browser (no desktop required)
- 📤 Upload videos directly from your computer
- ⚙️ Interactive configuration with sliders and checkboxes
- 📊 Real-time progress tracking
- ⬇️ Download individual clips or full montage
- 📱 Mobile-friendly responsive design
- 💡 Built-in help and tips

### Option 2: Desktop GUI

Launch the graphical interface (requires display):

**Windows:**
```bash
run_gui.bat
```

**Linux/Mac:**
```bash
./run_gui.sh
```

**Or manually:**
```bash
python VideoHighlightsGUI.py
```

The desktop GUI provides:
- Browse and select your video file
- Set trim start/end times with easy time format (MM:SS or HH:MM:SS)
- Configure all detection parameters
- Choose output directory
- Enable/disable options (player selection, overlay, audio detection)
- **GPU requirement checkbox** - ensure GPU is available before processing
- **GPU status indicator** - shows detected GPU or CPU mode
- See real-time processing output
- Get notified when complete

**Note:** Requires a graphical display. Use the web interface for remote/headless servers.

### Option 3: Command Line

For power users and automation:

#### Basic usage
```bash
python VideoHighlights.py --video /path/to/match.mp4 --out ./highlights_out
```

#### With player selection and overlay
```bash
python VideoHighlights.py --video match.mp4 --select --overlay
```

#### Trim long video (2nd half only - 45 min to 90 min)
```bash
python VideoHighlights.py --video match.mp4 --trim-start 45:00 --trim-end 1:30:00
```

#### Interactive mode (prompts for all options)
```bash
python VideoHighlights.py
```

### Command-Line Options

- `--video` - Input video path
- `--out` - Output directory for highlights
- `--select` - Interactively select your player's box on first frame
- `--trim-start` - Trim video start time (format: MM:SS or HH:MM:SS or seconds)
- `--trim-end` - Trim video end time (format: MM:SS or HH:MM:SS or seconds)
- `--pre` - Seconds before event (default: 2.0)
- `--post` - Seconds after event (default: 6.0)
- `--min-clip` - Minimum clip duration in seconds (default: 4.0)
- `--overlay` - Render spotlight overlay clips (slower)
- `--no-audio` - Disable audio-based peak detection

## How It Works

1. **Video Trimming** (Optional): Creates a working copy of the specified time range
2. **Player Tracking**: YOLO detects all people, ByteTrack maintains consistent IDs across frames
3. **Target Selection**: Either you select the player manually, or it picks the longest-lived track
4. **Highlight Detection**:
   - Analyzes player speed/acceleration for action moments
   - Detects audio peaks for crowd reactions
   - Merges nearby events into coherent clips
5. **Clip Generation**: Extracts highlight clips from the original video with proper timestamps
6. **Montage Creation**: Combines all highlights into a single compilation video

## Output

The script generates:
- Individual highlight clips: `highlight_01.mp4`, `highlight_02.mp4`, etc.
- Combined montage: `highlights_montage.mp4`
- Optional overlay versions with spotlight circle: `highlight_XX_spotlight.mp4`
- Temporary trimmed video (if using trim feature): `trimmed_working_video.mp4`

## Tips

- **Player Selection**: Use `--select` if your player isn't on the field the whole game
- **Long Videos**: Use `--trim-start` and `--trim-end` to process only relevant portions (saves time!)
- **Video Quality**: Works best with 1080p/60fps or 4K/60fps from a stable, elevated sideline view
- **First Run**: YOLO weights (~6MB) will auto-download on first use
- **Processing Time**: Expect ~1-2 minutes per minute of video for tracking (varies by hardware)

## Troubleshooting

### No highlights found
- Try lowering detection thresholds by editing `robust_threshold()` k value in the code
- Use `--select` to ensure the correct player is tracked
- Check that video has visible player movement

### Video file not found
- Use absolute paths or ensure relative paths are correct
- Check file extension is supported (.mp4, .mov, .avi, .mkv, .m4v)

### Memory issues with long videos
- Use the trim feature to process segments
- Close other applications to free RAM
- Consider processing in multiple passes

## License

See LICENSE file for details.

## Credits

Built with:
- YOLOv8 (Ultralytics) for object detection
- ByteTrack for multi-object tracking
- MoviePy for video processing
- Librosa for audio analysis
