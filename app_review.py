"""Review Portal: toggle between original, debug, zoom, and highlight views.

A lightweight Streamlit app for reviewing a processing run side by side:
point it at an output directory and switch instantly between the original
recording, the annotated camera-decision debug video, the full game-camera
zoom, the broadcast reel, and every individual highlight clip - with the
bookmark table, game states, goal/card/set-piece events, and card review
crops alongside.

Run:  streamlit run app_review.py --server.port=8505
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Video Highlights - Review", layout="wide")
st.title("Review Portal")

DEFAULT_ROOT = os.environ.get("VH_OUTPUT_ROOT", "./outputs")


def _find_runs(root: str) -> list[Path]:
    """Directories that look like pipeline outputs (contain a manifest)."""
    runs = []
    root_path = Path(root).expanduser()
    if root_path.is_dir():
        if (root_path / "analysis_bookmarks.json").exists():
            runs.append(root_path)
        for child in sorted(root_path.rglob("analysis_bookmarks.json")):
            if child.parent not in runs:
                runs.append(child.parent)
    return runs[:200]


with st.sidebar:
    st.header("Run")
    root = st.text_input("Search root (or exact output dir)", value=DEFAULT_ROOT)
    runs = _find_runs(root)
    if not runs:
        st.info("No runs found. Point this at a pipeline output directory "
                "(one containing analysis_bookmarks.json).")
        st.stop()
    run_dir = Path(st.selectbox("Output directory", [str(r) for r in runs]))

manifest = {}
states = {}
try:
    manifest = json.loads((run_dir / "analysis_bookmarks.json").read_text())
except Exception:
    pass
try:
    states = json.loads((run_dir / "analysis_game_states.json").read_text())
except Exception:
    pass

# ---------------------------------------------------------------------------
# View toggle
# ---------------------------------------------------------------------------

views: dict[str, Path] = {}
source = str(manifest.get("video_path") or "")
if source and Path(source).exists():
    views["Original"] = Path(source)
trimmed = run_dir / "trimmed_working_video.mp4"
if trimmed.exists():
    views["Original (trimmed window)"] = trimmed
debug_v = run_dir / "debug_camera_wide.mp4"
if debug_v.exists():
    views["Debug (camera decisions + why)"] = debug_v
for full in sorted(run_dir.glob("full_*_zoom.mp4")):
    views[f"Zoom ({full.stem.replace('full_', '').replace('_zoom', '')})"] = full
reel = run_dir / "highlights_reel.mp4"
if reel.exists():
    views["Broadcast reel"] = reel
montage = run_dir / "highlights_montage.mp4"
if montage.exists():
    views["Montage"] = montage
for clip in sorted(run_dir.glob("highlight_*.mp4")):
    if "spotlight" not in clip.name:
        views[f"Clip {clip.stem.split('_')[-1]}"] = clip

if not views:
    st.warning("No videos found in this run directory.")
    st.stop()

col_view, col_info = st.columns([2.2, 1.0])

with col_view:
    choice = st.radio("View", list(views.keys()), horizontal=True)
    video_path = views[choice]
    st.video(str(video_path))
    size_mb = video_path.stat().st_size / 1e6
    st.caption(f"`{video_path}` - {size_mb:.1f} MB")

with col_info:
    stats = manifest.get("stats", {}) or {}
    st.subheader("Run summary")
    summary_cols = st.columns(3)
    summary_cols[0].metric("Bookmarks", stats.get("bookmark_count", 0))
    summary_cols[1].metric("Goals", stats.get("goal_event_count", 0))
    ball_stats = (states.get("ball_track_stats") or {})
    coverage = ball_stats.get("coverage_fraction")
    summary_cols[2].metric("Ball coverage", f"{coverage * 100:.0f}%" if coverage is not None else "-")

    if states.get("state_summary_s"):
        st.caption("Game states (seconds)")
        st.json(states["state_summary_s"], expanded=False)

    for key, label in (
        ("goal_events", "Goals"),
        ("card_events", "Cards"),
        ("set_piece_events", "Set pieces"),
    ):
        rows = states.get(key) or []
        if rows:
            st.caption(f"{label} ({len(rows)})")
            st.json(rows, expanded=False)

# ---------------------------------------------------------------------------
# Bookmarks + card crops
# ---------------------------------------------------------------------------

bookmarks = list(manifest.get("bookmarks", []) or [])
if bookmarks:
    st.subheader("Bookmarks")
    st.dataframe(
        [
            {
                "id": b.get("bookmark_id"),
                "type": b.get("event_type"),
                "label": b.get("label"),
                "conf": b.get("confidence"),
                "at": b.get("occurred_at_s"),
                "start": b.get("start_s"),
                "end": b.get("end_s"),
                "state": b.get("game_state"),
                "sources": ",".join(b.get("sources", []) or []),
            }
            for b in bookmarks
        ],
        use_container_width=True,
        hide_index=True,
    )

crops = sorted((run_dir / "card_crops").glob("*.png")) if (run_dir / "card_crops").is_dir() else []
if crops:
    st.subheader(f"Card review crops ({len(crops)})")
    cols = st.columns(min(6, len(crops)))
    for i, crop in enumerate(crops[:24]):
        with cols[i % len(cols)]:
            st.image(str(crop), caption=crop.stem)

log_file = next(iter(run_dir.glob("*.log")), None)
if log_file:
    with st.expander(f"Run log ({log_file.name})"):
        st.code(log_file.read_text(errors="ignore")[-20000:], language="text")
