"""
Video Highlights Generator - Web Interface
------------------------------------------
A Streamlit web app for generating soccer highlight reels.
Run with: streamlit run app.py
"""

import streamlit as st
import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
import time

# Page config
st.set_page_config(
    page_title="Video Highlights Generator",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

def parse_time(time_str):
    """Parse time string to seconds"""
    if not time_str:
        return None
    time_str = time_str.strip()
    try:
        return float(time_str)
    except ValueError:
        parts = time_str.split(':')
        if len(parts) == 2:
            minutes, seconds = map(float, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:
            hours, minutes, seconds = map(float, parts)
            return hours * 3600 + minutes * 60 + seconds
    return None

def format_time(seconds):
    """Format seconds to readable time"""
    if seconds is None:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"

def run_highlights_generator(video_path, output_dir, config):
    """Run the highlights generator with given configuration"""
    cmd = [sys.executable, "VideoHighlights.py"]

    cmd.extend(["--video", video_path])
    cmd.extend(["--out", output_dir])
    cmd.extend(["--pre", str(config['pre_seconds'])])
    cmd.extend(["--post", str(config['post_seconds'])])
    cmd.extend(["--min-clip", str(config['min_clip'])])

    if config['trim_start']:
        cmd.extend(["--trim-start", config['trim_start']])
    if config['trim_end']:
        cmd.extend(["--trim-end", config['trim_end']])

    if config['select_player']:
        cmd.append("--select")
    if config['overlay']:
        cmd.append("--overlay")
    if config['no_audio']:
        cmd.append("--no-audio")

    return cmd

# Main header
st.markdown('<div class="main-header">⚽ Soccer Highlight Generator</div>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Video upload section
st.sidebar.subheader("📹 Video Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload your soccer game video",
    type=['mp4', 'mov', 'avi', 'mkv', 'm4v'],
    help="Select a video file from your computer (up to 50GB supported)"
)

# Or use existing file
use_test_video = st.sidebar.checkbox("Use test video (test_video.mp4)", value=False)

# Trim settings
st.sidebar.subheader("✂️ Video Trimming")
st.sidebar.markdown("*Optional: Process only a portion of the video*")

trim_enabled = st.sidebar.checkbox("Enable trimming", value=False)
trim_start = ""
trim_end = ""

if trim_enabled:
    trim_start = st.sidebar.text_input(
        "Start time",
        placeholder="0:30 or 30 or 0:00:30",
        help="Format: MM:SS, HH:MM:SS, or seconds"
    )
    trim_end = st.sidebar.text_input(
        "End time",
        placeholder="2:00 or 120 or 0:02:00",
        help="Format: MM:SS, HH:MM:SS, or seconds"
    )

# Detection parameters
st.sidebar.subheader("🎯 Detection Parameters")
pre_seconds = st.sidebar.slider("Pre-event buffer (seconds)", 0.5, 10.0, 2.0, 0.5,
                                help="Seconds to include before the highlight event")
post_seconds = st.sidebar.slider("Post-event buffer (seconds)", 1.0, 15.0, 6.0, 0.5,
                                 help="Seconds to include after the highlight event")
min_clip = st.sidebar.slider("Minimum clip duration (seconds)", 1.0, 10.0, 4.0, 0.5,
                             help="Minimum length for each highlight clip")

# Options
st.sidebar.subheader("🎬 Options")
select_player = st.sidebar.checkbox(
    "Manually select player",
    value=False,
    help="⚠️ Requires display - won't work in web mode. Use command line for this feature."
)
overlay = st.sidebar.checkbox(
    "Create spotlight overlay clips",
    value=False,
    help="Slower processing - adds visual spotlight on tracked player"
)
no_audio = st.sidebar.checkbox(
    "Disable audio detection",
    value=False,
    help="Skip audio-based highlight detection (faster)"
)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Processing Status")
    status_container = st.empty()
    progress_bar = st.progress(0)
    output_container = st.empty()

with col2:
    st.subheader("ℹ️ Current Settings")

    if uploaded_file:
        st.info(f"**Video:** {uploaded_file.name}")
    elif use_test_video:
        st.info("**Video:** test_video.mp4 (built-in)")
    else:
        st.warning("**Video:** Not selected")

    st.info(f"""
    **Buffers:**
    - Pre-event: {pre_seconds}s
    - Post-event: {post_seconds}s
    - Min clip: {min_clip}s
    """)

    if trim_enabled and (trim_start or trim_end):
        start_parsed = parse_time(trim_start) if trim_start else 0
        end_parsed = parse_time(trim_end) if trim_end else None
        st.info(f"""
        **Trim Range:**
        - Start: {format_time(start_parsed)}
        - End: {format_time(end_parsed) if end_parsed else 'End of video'}
        """)

    options = []
    if select_player:
        options.append("Manual selection")
    if overlay:
        options.append("Spotlight overlay")
    if no_audio:
        options.append("No audio detection")

    if options:
        st.info("**Options:** " + ", ".join(options))

# Process button
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    process_button = st.button("🚀 Generate Highlights", type="primary", use_container_width=True)

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'output_dir' not in st.session_state:
    st.session_state.output_dir = None

# Process video when button is clicked
if process_button:
    # Validate inputs
    video_path = None

    if uploaded_file:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
    elif use_test_video:
        if os.path.exists("test_video.mp4"):
            video_path = "test_video.mp4"
        else:
            st.error("❌ test_video.mp4 not found in current directory")
    else:
        st.error("❌ Please upload a video or select the test video")

    if video_path:
        # Create output directory
        output_dir = tempfile.mkdtemp(prefix="highlights_")
        st.session_state.output_dir = output_dir

        # Prepare configuration
        config = {
            'pre_seconds': pre_seconds,
            'post_seconds': post_seconds,
            'min_clip': min_clip,
            'trim_start': trim_start if trim_enabled else None,
            'trim_end': trim_end if trim_enabled else None,
            'select_player': select_player,
            'overlay': overlay,
            'no_audio': no_audio
        }

        # Build command
        cmd = run_highlights_generator(video_path, output_dir, config)

        status_container.info("⏳ Processing video... This may take several minutes.")
        progress_bar.progress(10)

        # Run the process
        try:
            with st.spinner("Processing..."):
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                output_lines = []
                for line in process.stdout:
                    output_lines.append(line.rstrip())
                    # Update progress based on output
                    if "[1/5]" in line:
                        progress_bar.progress(20)
                    elif "[2/5]" in line:
                        progress_bar.progress(40)
                    elif "[3/5]" in line:
                        progress_bar.progress(60)
                    elif "[4/5]" in line:
                        progress_bar.progress(80)
                    elif "[5/5]" in line:
                        progress_bar.progress(90)

                process.wait()
                progress_bar.progress(100)

                # Display output
                output_container.code("\n".join(output_lines[-20:]), language="bash")

                if process.returncode == 0:
                    status_container.success("✅ Processing completed successfully!")

                    # List generated files
                    st.success("### 🎉 Highlights Generated!")

                    output_files = list(Path(output_dir).glob("*.mp4"))
                    if output_files:
                        st.write(f"Found {len(output_files)} files:")

                        for file in sorted(output_files):
                            col_file1, col_file2 = st.columns([3, 1])
                            with col_file1:
                                st.write(f"📹 {file.name}")
                            with col_file2:
                                with open(file, 'rb') as f:
                                    st.download_button(
                                        label="⬇️ Download",
                                        data=f,
                                        file_name=file.name,
                                        mime="video/mp4",
                                        key=file.name
                                    )

                        st.info(f"📁 All files saved to: `{output_dir}`")
                    else:
                        st.warning("No output files found. Check the output log above.")
                else:
                    status_container.error(f"❌ Processing failed with error code {process.returncode}")
                    st.error("Check the output log above for details.")

        except Exception as e:
            status_container.error(f"❌ Error: {str(e)}")
            st.exception(e)
        finally:
            # Clean up uploaded temp file
            if uploaded_file and video_path != "test_video.mp4":
                try:
                    os.unlink(video_path)
                except:
                    pass

# Help section
with st.expander("📖 How to Use"):
    st.markdown("""
    ### Getting Started
    1. **Upload a video** or use the test video
    2. **Configure settings** in the sidebar:
       - Set trim times if you only want to process part of the video
       - Adjust detection parameters (buffers, minimum clip length)
       - Enable options like overlay or disable audio detection
    3. **Click "Generate Highlights"** and wait for processing
    4. **Download your highlights** when complete!

    ### Time Format
    For trim times, you can use:
    - Seconds: `90` (90 seconds)
    - MM:SS: `1:30` (1 minute 30 seconds)
    - HH:MM:SS: `1:30:00` (1 hour 30 minutes)

    ### Tips
    - 🚀 **Long videos?** Use trimming to process specific halves or periods
    - ⚡ **Faster processing?** Disable audio detection
    - 🎯 **Better accuracy?** Keep audio detection enabled
    - 🎬 **Cool effects?** Enable spotlight overlay (slower)

    ### Processing Time
    Expect approximately 1-2 minutes per minute of video, depending on:
    - Video resolution
    - Server resources
    - Number of people in the video
    - Overlay enabled/disabled
    """)

with st.expander("🔧 Technical Details"):
    st.markdown("""
    ### How It Works
    1. **Video Trimming** (optional): Creates a working segment
    2. **Player Tracking**: YOLOv8 + ByteTrack detect and track all players
    3. **Target Selection**: Picks the longest-visible player automatically
    4. **Highlight Detection**:
       - Analyzes player speed and acceleration
       - Detects audio peaks (crowd reactions, whistles)
       - Merges nearby events into clips
    5. **Clip Generation**: Extracts highlights with proper timestamps
    6. **Montage Creation**: Combines all highlights into one video

    ### Output Files
    - `highlight_01.mp4`, `highlight_02.mp4`, etc. - Individual clips
    - `highlights_montage.mp4` - All clips combined
    - `highlight_XX_spotlight.mp4` - Overlay versions (if enabled)
    - `trimmed_working_video.mp4` - Temporary trimmed video (if using trim)

    ### Technology Stack
    - **YOLOv8** - Object detection
    - **ByteTrack** - Multi-object tracking
    - **MoviePy** - Video processing
    - **Librosa** - Audio analysis
    - **Streamlit** - Web interface
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    ⚽ Video Highlights Generator | Built with Python, YOLO, and Streamlit
</div>
""", unsafe_allow_html=True)
