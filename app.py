
"""
Video Highlights Portal (SaaS-style Streamlit UI)

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import re
from html import escape
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests
import streamlit as st


EVENT_TARGET_OPTIONS = [
    "goal",
    "shot",
    "corner_kick",
    "penalty_kick",
    "free_kick",
    "goal_kick",
    "kickoff",
    "foul",
    "save",
]

ANNOUNCEMENTS = [
    {
        "title": "Model Upgrade Workflow Active",
        "message": "You can rerun any processed game against newer model versions from the Game Library.",
    },
    {
        "title": "Extreme Logging Available",
        "message": "Use Operations Console to inspect step-by-step job logs and quickly kill sessions when testing.",
    },
    {
        "title": "Targeted Highlight Runs",
        "message": "Configure event target sets per run (goals, corners, shots, fouls, saves, and more).",
    },
]

PIPELINE_STAGES = [
    "queued",
    "claimed",
    "initializing",
    "processing_video",
    "completed",
]


st.set_page_config(
    page_title="Video Highlights Portal",
    page_icon="VH",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Source+Sans+3:wght@400;600;700&display=swap');

:root {
  --bg: #eef2f4;
  --panel: #ffffff;
  --panel-soft: #f7f9fa;
  --ink: #151f29;
  --muted: #5f6f7a;
  --brand: #0a6774;
  --brand-2: #28754c;
  --warn: #a86500;
  --danger: #a33a3a;
  --line: #d7e0e5;
  --line-strong: #b8c6ce;
}

html, body, [class*="css"] {
  font-family: "Source Sans 3", sans-serif;
  color: var(--ink);
}

.stApp {
  background: var(--bg);
}

.block-container {
  padding-top: 1.35rem;
  padding-bottom: 2rem;
  max-width: 1500px;
}

[data-testid="stSidebar"] {
  background: #111a22;
  border-right: 1px solid #25323c;
}

[data-testid="stSidebar"] * {
  color: #edf4f7;
}

[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
  color: var(--ink) !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] * {
  color: var(--ink) !important;
}

[data-testid="stSidebar"] [role="radiogroup"] * {
  color: #edf4f7 !important;
}

[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
  color: #aebbc3 !important;
}

h1, h2, h3, h4 {
  font-family: "Space Grotesk", sans-serif;
  letter-spacing: 0;
}

h2, h3 {
  color: var(--ink);
}

div[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: .75rem .85rem;
  box-shadow: 0 1px 2px rgba(21, 31, 41, 0.05);
}

div[data-testid="stMetric"] label {
  color: var(--muted);
}

.stTabs [data-baseweb="tab-list"] {
  gap: .25rem;
  border-bottom: 1px solid var(--line);
}

.stTabs [data-baseweb="tab"] {
  border-radius: 6px 6px 0 0;
  padding: .45rem .75rem;
}

.stTabs [aria-selected="true"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-bottom-color: var(--panel);
}

.stButton > button,
.stDownloadButton > button {
  border-radius: 6px;
  border: 1px solid var(--line-strong);
  background: var(--panel);
  color: var(--ink);
  box-shadow: 0 1px 2px rgba(21, 31, 41, 0.06);
}

.stButton > button:hover,
.stDownloadButton > button:hover {
  border-color: var(--brand);
  color: var(--brand);
}

.stButton > button[kind="primary"] {
  background: var(--brand);
  border-color: var(--brand);
  color: #ffffff;
}

input, textarea, [data-baseweb="select"] > div {
  border-radius: 6px !important;
}

[data-testid="stFileUploader"] section {
  border-radius: 8px;
  border: 1px dashed var(--line-strong);
  background: var(--panel-soft);
}

.portal-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 1rem;
  padding: .15rem 0 .85rem 0;
  margin-bottom: .35rem;
  border-bottom: 1px solid var(--line);
}

.portal-header h1 {
  margin: 0;
  font-size: 1.75rem;
  line-height: 1.1;
  color: var(--ink);
}

.portal-header p {
  margin: .3rem 0 0 0;
  color: var(--muted);
  font-size: .96rem;
}

.portal-kicker {
  display: inline-flex;
  align-items: center;
  gap: .4rem;
  color: var(--brand);
  font-size: .82rem;
  font-weight: 700;
  letter-spacing: .05em;
  text-transform: uppercase;
}

.portal-statusbar {
  display: flex;
  align-items: center;
  gap: .45rem;
  flex-wrap: wrap;
  justify-content: flex-end;
  color: var(--muted);
  font-size: .84rem;
}

.portal-chip {
  display: inline-flex;
  align-items: center;
  min-height: 1.55rem;
  padding: .18rem .55rem;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: var(--panel);
  color: var(--muted);
  white-space: nowrap;
}

.tile {
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 8px;
  padding: .9rem 1rem;
  min-height: 96px;
  box-shadow: 0 1px 2px rgba(21, 31, 41, 0.05);
}

.tile h4 {
  margin: 0 0 .35rem 0;
  font-family: "Space Grotesk", sans-serif;
  color: var(--ink);
  font-size: 1.02rem;
}

.tile p {
  margin: 0;
  color: var(--muted);
  font-size: .94rem;
}

.announce {
  border: 1px solid var(--line);
  border-left: 4px solid var(--brand);
  border-radius: 8px;
  padding: .65rem .8rem;
  margin: .35rem 0;
  background: var(--panel);
}

.announce b {
  color: var(--ink);
}

.transaction-pill {
  display: inline-block;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: .18rem .55rem;
  margin: 0 .2rem .35rem 0;
  font-size: .8rem;
  color: var(--muted);
  background: #fff;
}

.transaction-pill.active {
  border-color: #1f90aa;
  color: #125a6b;
  background: #e8f8fc;
}

.transaction-pill.done {
  border-color: #2f9f63;
  color: #16663c;
  background: #e8f9ee;
}

.transaction-pill.fail {
  border-color: #bd4141;
  color: #7f2424;
  background: #fcecec;
}

.studio-card {
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 8px;
  padding: .85rem .9rem;
  min-height: 172px;
  box-shadow: 0 1px 3px rgba(21, 31, 41, 0.07);
}

.studio-card h4 {
  margin: 0 0 .3rem 0;
  font-family: "Space Grotesk", sans-serif;
  color: var(--ink);
  font-size: 1.02rem;
}

.studio-muted {
  color: var(--muted);
  font-size: .88rem;
  margin: 0 0 .2rem 0;
}

.studio-chip {
  display: inline-block;
  margin: 0 .35rem .35rem 0;
  padding: .12rem .5rem;
  border-radius: 6px;
  font-size: .76rem;
  border: 1px solid var(--line);
  color: var(--muted);
  background: #fff;
}

.studio-chip.live {
  border-color: #1f90aa;
  color: #125a6b;
  background: #e8f8fc;
}

.studio-chip.good {
  border-color: #2f9f63;
  color: #16663c;
  background: #e8f9ee;
}

.studio-chip.bad {
  border-color: #bd4141;
  color: #7f2424;
  background: #fcecec;
}

@media (max-width: 900px) {
  .portal-header {
    align-items: flex-start;
    flex-direction: column;
  }

  .portal-statusbar {
    justify-content: flex-start;
  }
}
</style>
""",
    unsafe_allow_html=True,
)


def _as_json(response: requests.Response) -> Dict[str, Any]:
    try:
        payload = response.json()
    except Exception:
        payload = {"raw": response.text}
    return {"ok": response.ok, "status_code": response.status_code, "payload": payload}


def api_request(
    method: str,
    api_base: str,
    path: str,
    tenant_id: str,
    token: str,
    json_body: Optional[Dict[str, Any]] = None,
    files: Optional[Dict[str, Any]] = None,
    timeout: int = 180,
) -> Dict[str, Any]:
    headers: Dict[str, str] = {"X-Tenant-Id": tenant_id}
    if token.strip():
        headers["Authorization"] = f"Bearer {token.strip()}"
    url = f"{api_base.rstrip('/')}{path}"
    response = requests.request(
        method=method.upper(),
        url=url,
        headers=headers,
        json=json_body,
        files=files,
        timeout=timeout,
    )
    return _as_json(response)


def list_matches(api_base: str, tenant_id: str, token: str, limit: int = 200) -> List[Dict[str, Any]]:
    result = api_request("GET", api_base, f"/matches?limit={limit}", tenant_id, token)
    if not result["ok"]:
        return []
    payload = result["payload"] or {}
    return list(payload.get("items", []))


def list_match_jobs(api_base: str, tenant_id: str, token: str, match_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    result = api_request("GET", api_base, f"/matches/{match_id}/jobs?limit={limit}", tenant_id, token)
    if not result["ok"]:
        return []
    payload = result["payload"] or {}
    return list(payload.get("items", []))


def list_training_models(api_base: str, tenant_id: str, token: str) -> List[Dict[str, Any]]:
    result = api_request("GET", api_base, "/training/models", tenant_id, token)
    if not result["ok"]:
        return []
    payload = result["payload"] or []
    return list(payload)


def list_match_events(
    api_base: str,
    tenant_id: str,
    token: str,
    match_id: str,
    job_id: Optional[str] = None,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    remaining = max(1, int(limit))
    cursor: Optional[str] = None
    all_items: List[Dict[str, Any]] = []
    while remaining > 0:
        page_size = min(500, remaining)
        query = [f"limit={page_size}"]
        if job_id:
            query.append(f"job_id={job_id}")
        if cursor:
            query.append(f"cursor={cursor}")
        suffix = "&".join(query)
        result = api_request("GET", api_base, f"/matches/{match_id}/events?{suffix}", tenant_id, token)
        if not result["ok"]:
            break
        payload = result["payload"] or {}
        items = list(payload.get("items", []))
        all_items.extend(items)
        remaining -= len(items)
        cursor = payload.get("next_cursor")
        if not cursor or not items:
            break
    return all_items


def list_job_bookmarks(
    api_base: str,
    tenant_id: str,
    token: str,
    job_id: str,
    limit: int = 2000,
) -> Dict[str, Any]:
    result = api_request("GET", api_base, f"/jobs/{job_id}/bookmarks?limit={int(limit)}", tenant_id, token)
    if not result["ok"]:
        return {"source": "none", "status": "unknown", "items": []}
    payload = result["payload"] or {}
    return {
        "source": str(payload.get("source", "none")),
        "status": str(payload.get("status", "unknown")),
        "items": list(payload.get("items", [])),
    }


def _iso_to_short(value: Optional[str]) -> str:
    if not value:
        return "-"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return value


def _latest_job(jobs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not jobs:
        return None
    return jobs[0]


def _focus_targets_from_config(config: Dict[str, Any]) -> str:
    targets = config.get("focus_event_types", [])
    if not isinstance(targets, list) or not targets:
        return "all-events"
    return ", ".join(str(item) for item in targets)


def _seconds_to_clock(value: float) -> str:
    total = max(0, int(round(float(value))))
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _resolve_match_video_source(match: Dict[str, Any]) -> str:
    source_path = str(match.get("source_video_path") or "").strip()
    if source_path:
        return source_path
    metadata = match.get("metadata") or {}
    assets = list(metadata.get("assets", []) or [])
    if assets:
        return str(assets[0].get("path") or "").strip()
    return ""


def _extract_first_frame_payload(video_path: str) -> Dict[str, Any]:
    try:
        import cv2  # type: ignore
    except Exception:
        return {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return {}

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame_rgb.shape[:2]
    return {"image_rgb": frame_rgb, "frame_width": int(width), "frame_height": int(height)}


@st.cache_data(show_spinner=False)
def _extract_first_frame_from_video_path(video_path: str) -> Dict[str, Any]:
    if not video_path or not os.path.exists(video_path):
        return {}
    return _extract_first_frame_payload(video_path)


@st.cache_data(show_spinner=False)
def _extract_first_frame_from_video_bytes(video_bytes: bytes, suffix: str) -> Dict[str, Any]:
    if not video_bytes:
        return {}

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".mp4") as handle:
            handle.write(video_bytes)
            temp_path = handle.name
        return _extract_first_frame_payload(temp_path)
    except Exception:
        return {}
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def _draw_roi_overlay(
    image_rgb: Any,
    x1_norm: float,
    y1_norm: float,
    x2_norm: float,
    y2_norm: float,
) -> Any:
    preview = image_rgb.copy()
    height, width = preview.shape[:2]
    x1 = int(round(width * x1_norm))
    x2 = int(round(width * x2_norm))
    y1 = int(round(height * y1_norm))
    y2 = int(round(height * y2_norm))
    border = max(2, min(width, height) // 120)
    color = [255, 170, 0]

    preview[max(0, y1 - border) : min(height, y1 + border), max(0, x1) : min(width, x2)] = color
    preview[max(0, y2 - border) : min(height, y2 + border), max(0, x1) : min(width, x2)] = color
    preview[max(0, y1) : min(height, y2), max(0, x1 - border) : min(width, x1 + border)] = color
    preview[max(0, y1) : min(height, y2), max(0, x2 - border) : min(width, x2 + border)] = color
    return preview


def _render_player_roi_selector(
    *,
    context_key: str,
    default_enabled: bool,
    existing_roi: Optional[Dict[str, Any]] = None,
    source_video_path: str = "",
    upload_bytes: Optional[bytes] = None,
    upload_name: str = "",
) -> Dict[str, Any]:
    enabled = st.checkbox(
        "Lock onto a player ROI for tracking and follow-cam",
        value=default_enabled,
        key=f"{context_key}_player_roi_enabled",
        help="Use the first-frame box below to keep analysis focused on one player instead of auto-selecting a long-lived track.",
    )
    if not enabled:
        return {"enabled": False, "roi": None}

    suffix = os.path.splitext(upload_name or source_video_path or "preview.mp4")[1] or ".mp4"
    preview_payload: Dict[str, Any] = {}
    if upload_bytes:
        preview_payload = _extract_first_frame_from_video_bytes(upload_bytes, suffix)
    elif source_video_path:
        preview_payload = _extract_first_frame_from_video_path(source_video_path)

    if not preview_payload:
        st.warning("Could not extract the first frame for player selection. Uncheck ROI lock or use a readable local source video.")
        return {"enabled": True, "roi": None}

    image_rgb = preview_payload["image_rgb"]
    existing = existing_roi or {}
    x1_default = int(round(float(existing.get("x1_norm", 0.35)) * 100))
    y1_default = int(round(float(existing.get("y1_norm", 0.25)) * 100))
    x2_default = int(round(float(existing.get("x2_norm", 0.45)) * 100))
    y2_default = int(round(float(existing.get("y2_norm", 0.75)) * 100))
    x1_default = min(max(0, x1_default), 98)
    y1_default = min(max(0, y1_default), 98)
    x2_default = min(max(x1_default + 1, x2_default), 100)
    y2_default = min(max(y1_default + 1, y2_default), 100)

    x1_key = f"{context_key}_player_roi_x1"
    y1_key = f"{context_key}_player_roi_y1"
    x2_key = f"{context_key}_player_roi_x2"
    y2_key = f"{context_key}_player_roi_y2"
    st.session_state.setdefault(x1_key, x1_default)
    st.session_state.setdefault(y1_key, y1_default)
    st.session_state.setdefault(x2_key, x2_default)
    st.session_state.setdefault(y2_key, y2_default)

    st.session_state[x1_key] = min(max(int(st.session_state[x1_key]), 0), 98)
    st.session_state[y1_key] = min(max(int(st.session_state[y1_key]), 0), 98)
    st.session_state[x2_key] = min(max(int(st.session_state[x2_key]), st.session_state[x1_key] + 1), 100)
    st.session_state[y2_key] = min(max(int(st.session_state[y2_key]), st.session_state[y1_key] + 1), 100)

    roi_col1, roi_col2 = st.columns([1.5, 1.0])
    with roi_col1:
        preview_image = _draw_roi_overlay(
            image_rgb,
            st.session_state[x1_key] / 100.0,
            st.session_state[y1_key] / 100.0,
            st.session_state[x2_key] / 100.0,
            st.session_state[y2_key] / 100.0,
        )
        st.image(preview_image, caption="First-frame player lock preview", use_container_width=True)
    with roi_col2:
        x1_pct = st.slider("Left %", 0, 98, int(st.session_state[x1_key]), 1, key=x1_key)
        y1_pct = st.slider("Top %", 0, 98, int(st.session_state[y1_key]), 1, key=y1_key)
        x2_pct = st.slider("Right %", x1_pct + 1, 100, max(int(st.session_state[x2_key]), x1_pct + 1), 1, key=x2_key)
        y2_pct = st.slider("Bottom %", y1_pct + 1, 100, max(int(st.session_state[y2_key]), y1_pct + 1), 1, key=y2_key)
        st.caption(
            f"Frame size: `{preview_payload['frame_width']}x{preview_payload['frame_height']}` | "
            f"ROI: `{x1_pct}%,{y1_pct}%` to `{x2_pct}%,{y2_pct}%`"
        )

    roi = {
        "normalized": True,
        "x1_norm": round(x1_pct / 100.0, 4),
        "y1_norm": round(y1_pct / 100.0, 4),
        "x2_norm": round(x2_pct / 100.0, 4),
        "y2_norm": round(y2_pct / 100.0, 4),
    }
    return {"enabled": True, "roi": roi}


def _resolve_clip_playback_source(path: str, download_url: str) -> str:
    raw_url = str(download_url or "").strip()
    if raw_url.lower().startswith("http://") or raw_url.lower().startswith("https://"):
        return raw_url
    raw_path = str(path or "").strip()
    return raw_path or raw_url


def _build_bookmark_rows(
    job: Dict[str, Any],
    events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if events:
        events_sorted = sorted(events, key=lambda item: int(item.get("occurred_at_ms", 0)))
        for item in events_sorted:
            occurred_s = float(item.get("occurred_at_ms", 0)) / 1000.0
            start_s = float(item.get("start_ms", 0)) / 1000.0
            end_s = float(item.get("end_ms", 0)) / 1000.0
            source = item.get("source", {}) if isinstance(item.get("source", {}), dict) else {}
            source_list = source.get("sources", [])
            if isinstance(source_list, list) and source_list:
                sensed_by = ", ".join(str(entry) for entry in source_list)
            else:
                sensed_by = str(source.get("detector") or "n/a")
            explanations = list(item.get("explanations", []) or [])
            signal_texts: List[str] = []
            for exp in explanations[:3]:
                if not isinstance(exp, dict):
                    continue
                signal = str(exp.get("signal") or "").strip()
                value = exp.get("value")
                if signal:
                    try:
                        signal_texts.append(f"{signal}:{float(value):.2f}")
                    except Exception:
                        signal_texts.append(signal)
            rows.append(
                {
                    "event_id": item.get("event_id"),
                    "time": _seconds_to_clock(occurred_s),
                    "occurred_s": round(occurred_s, 2),
                    "window": f"{_seconds_to_clock(start_s)} - {_seconds_to_clock(end_s)}",
                    "event_type": item.get("event_type"),
                    "confidence": round(float(item.get("confidence", 0.0)), 3),
                    "sensed_by": sensed_by,
                    "signals": "; ".join(signal_texts),
                    "status": item.get("status"),
                }
            )
        return rows

    bookmarks = list((job.get("result", {}) or {}).get("bookmarks", []) or [])
    for item in bookmarks:
        occurred_s = float(item.get("occurred_at_s", 0.0) or 0.0)
        start_s = float(item.get("start_s", occurred_s) or occurred_s)
        end_s = float(item.get("end_s", occurred_s) or occurred_s)
        rows.append(
            {
                "event_id": item.get("bookmark_id"),
                "time": _seconds_to_clock(occurred_s),
                "occurred_s": round(occurred_s, 2),
                "window": f"{_seconds_to_clock(start_s)} - {_seconds_to_clock(end_s)}",
                "event_type": item.get("event_type", "candidate"),
                "confidence": round(float(item.get("confidence", 0.0) or 0.0), 3),
                "sensed_by": ", ".join(str(source) for source in list(item.get("sources", []) or [])),
                "signals": "; ".join(
                    f"{key}:{value}" for key, value in (item.get("signals", {}) or {}).items()
                ),
                "status": "bookmark_only",
            }
        )
    return rows


def _bookmark_rows_from_live_items(items: List[Dict[str, Any]], source: str = "manifest") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if "occurred_at_ms" in item:
            occurred_s = float(item.get("occurred_at_ms", 0)) / 1000.0
            start_s = float(item.get("start_ms", 0)) / 1000.0
            end_s = float(item.get("end_ms", 0)) / 1000.0
            rows.append(
                {
                    "event_id": item.get("event_id"),
                    "time": _seconds_to_clock(occurred_s),
                    "occurred_s": round(occurred_s, 2),
                    "window": f"{_seconds_to_clock(start_s)} - {_seconds_to_clock(end_s)}",
                    "event_type": item.get("event_type"),
                    "confidence": round(float(item.get("confidence", 0.0) or 0.0), 3),
                    "sensed_by": str((item.get("source", {}) or {}).get("detector", "n/a")),
                    "signals": "",
                    "status": item.get("status"),
                }
            )
        else:
            occurred_s = float(item.get("occurred_at_s", 0.0) or 0.0)
            start_s = float(item.get("start_s", occurred_s) or occurred_s)
            end_s = float(item.get("end_s", occurred_s) or occurred_s)
            rows.append(
                {
                    "event_id": item.get("bookmark_id"),
                    "time": _seconds_to_clock(occurred_s),
                    "occurred_s": round(occurred_s, 2),
                    "window": f"{_seconds_to_clock(start_s)} - {_seconds_to_clock(end_s)}",
                    "event_type": item.get("event_type", "candidate"),
                    "confidence": round(float(item.get("confidence", 0.0) or 0.0), 3),
                    "sensed_by": ", ".join(str(value) for value in list(item.get("sources", []) or [])),
                    "signals": "; ".join(
                        f"{k}:{v}" for k, v in (item.get("signals", {}) or {}).items()
                    ),
                    "status": item.get("status", f"{source}_bookmark"),
                }
            )
    return rows


def _build_stage_tracker(current_stage: Optional[str], status: Optional[str]) -> str:
    stage = (current_stage or "").strip().lower()
    stat = (status or "").strip().lower()
    chips: List[str] = []
    done_cutoff = -1
    if stage in PIPELINE_STAGES:
        done_cutoff = PIPELINE_STAGES.index(stage)
    if stat in {"completed"}:
        done_cutoff = len(PIPELINE_STAGES) - 1

    for idx, item in enumerate(PIPELINE_STAGES):
        classes = ["transaction-pill"]
        if stat in {"failed"} and item == "processing_video":
            classes.append("fail")
        elif idx <= done_cutoff:
            classes.append("done")
        elif item == stage:
            classes.append("active")
        chips.append(f"<span class=\"{' '.join(classes)}\">{_h(item)}</span>")

    if stat in {"failed", "canceled", "cancel_requested"}:
        chips.append(f"<span class=\"transaction-pill fail\">{_h(stat)}</span>")
    return "".join(chips)


def _profile_defaults(profile_name: str, selected_targets: List[str]) -> Dict[str, Any]:
    profile = profile_name.lower()
    defaults: Dict[str, Any] = {
        "pre_seconds": 2.0,
        "post_seconds": 6.0,
        "min_clip_duration": 4.0,
        "speed_sensitivity": 2.0,
        "audio_sensitivity": 2.0,
    }
    if profile == "offense focus":
        defaults.update({"pre_seconds": 2.0, "post_seconds": 7.0, "speed_sensitivity": 1.8})
        if not selected_targets:
            defaults["focus_event_types"] = ["goal", "shot", "penalty_kick", "save"]
    elif profile == "set piece focus":
        defaults.update({"pre_seconds": 3.5, "post_seconds": 8.0, "speed_sensitivity": 2.2})
        if not selected_targets:
            defaults["focus_event_types"] = ["corner_kick", "free_kick", "goal_kick", "kickoff", "penalty_kick"]
    elif profile == "discipline review":
        defaults.update({"pre_seconds": 3.0, "post_seconds": 7.0, "audio_sensitivity": 1.7})
        if not selected_targets:
            defaults["focus_event_types"] = ["foul"]
    return defaults


def _recommended_job_config() -> Dict[str, Any]:
    return {
        "profile_name": "Balanced",
        "model_version": "event-v0",
        "focus_event_types": [],
        "pre_seconds": 2.0,
        "post_seconds": 6.0,
        "min_clip_duration": 4.0,
        "speed_sensitivity": 2.0,
        "audio_sensitivity": 2.0,
        "camera_mode": "wide",
        "zoom_factor": 1.6,
        "overlay": False,
        "no_audio": False,
        "require_gpu": False,
        "analysis_only": True,
    }


def _render_portal_header() -> None:
    st.markdown(
        """
<div class="portal-header">
  <div>
    <div class="portal-kicker">Video Highlights</div>
    <h1>Match Operations Studio</h1>
    <p>Ingest, analysis, review, exports, and run diagnostics.</p>
  </div>
  <div class="portal-statusbar">
    <span class="portal-chip">Tenant workspace</span>
    <span class="portal-chip">API connected on demand</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def safe_rerun() -> None:
    rerun_fn = getattr(st, "rerun", None)
    if callable(rerun_fn):
        rerun_fn()
        return
    legacy_fn = getattr(st, "experimental_rerun", None)
    if callable(legacy_fn):
        legacy_fn()


def _h(value: Any) -> str:
    return escape(str(value or ""), quote=True)


def _flatten_query_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        return str(value[0])
    return str(value)


def _read_query_params() -> Dict[str, str]:
    modern = getattr(st, "query_params", None)
    if modern is not None:
        try:
            return {
                str(key): _flatten_query_value(value)
                for key, value in dict(modern).items()
            }
        except Exception:
            pass

    legacy_get = getattr(st, "experimental_get_query_params", None)
    if callable(legacy_get):
        try:
            raw = legacy_get()
            return {
                str(key): _flatten_query_value(value)
                for key, value in dict(raw).items()
            }
        except Exception:
            pass
    return {}


def _write_query_params(params: Dict[str, str]) -> None:
    cleaned = {
        str(key): str(value).strip()
        for key, value in params.items()
        if str(value).strip()
    }
    current = _read_query_params()
    if current == cleaned:
        return

    modern = getattr(st, "query_params", None)
    if modern is not None:
        try:
            modern.clear()
            for key, value in cleaned.items():
                modern[key] = value
            return
        except Exception:
            pass

    legacy_set = getattr(st, "experimental_set_query_params", None)
    if callable(legacy_set):
        legacy_set(**cleaned)


def _slugify_text(value: str, fallback: str = "item") -> str:
    lowered = (value or "").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or fallback


def _build_entity_ref(entity_id: str, label: str = "") -> str:
    clean_id = str(entity_id or "").strip()
    if not clean_id:
        return ""
    if not label.strip():
        return f"{clean_id.split('_')[0]}--{clean_id}"
    prefix = _slugify_text(label, fallback=clean_id.split("_")[0])
    return f"{prefix}--{clean_id}"


def _extract_entity_id_from_ref(reference: str, prefix: str) -> str:
    raw = str(reference or "").strip()
    if not raw:
        return ""
    if raw.startswith(f"{prefix}_"):
        return raw
    match = re.search(rf"{re.escape(prefix)}_[a-zA-Z0-9]+", raw)
    if match:
        return match.group(0)
    return ""


def _apply_url_state_to_session() -> None:
    query_params = _read_query_params()
    signature = "&".join(
        f"{key}={query_params[key]}"
        for key in sorted(query_params.keys())
    )
    if signature == str(st.session_state.get("portal_url_applied_signature", "")):
        return
    st.session_state.portal_url_applied_signature = signature
    if not query_params:
        return

    mode_value = str(query_params.get("mode", "")).strip().lower()
    if mode_value in {"technical", "tech"}:
        st.session_state.portal_experience_mode = "Technical"
    elif mode_value in {"user", "friendly", "user-friendly", "user_friendly"}:
        st.session_state.portal_experience_mode = "User Friendly"

    tenant_value = str(
        query_params.get("tenant")
        or query_params.get("tenant_id")
        or ""
    ).strip()
    if tenant_value:
        st.session_state.portal_tenant = tenant_value

    api_value = str(query_params.get("api", "")).strip()
    if api_value:
        st.session_state.portal_api_base = api_value

    view_value = str(query_params.get("view", "")).strip().lower()
    view_map = {
        "home": "Portal Home",
        "studio": "Portal Home",
        "upload": "New Processing Run",
        "library": "Game Library",
        "monitor": "Operations Console",
        "operations": "Operations Console",
        "run_monitor": "Operations Console",
        "run-monitor": "Operations Console",
    }
    target_nav = view_map.get(view_value, "")
    if target_nav:
        if target_nav == "Game Library":
            st.session_state.portal_experience_mode = "Technical"
            st.session_state.portal_nav = target_nav
        elif str(st.session_state.get("portal_experience_mode", "User Friendly")) == "Technical":
            st.session_state.portal_nav = target_nav
        else:
            user_nav_aliases = {
                "Portal Home": "Studio",
                "New Processing Run": "Upload",
                "Operations Console": "Run Monitor",
            }
            st.session_state.portal_user_nav_main = user_nav_aliases.get(target_nav, "Studio")

    match_ref = str(query_params.get("match") or query_params.get("match_id") or "").strip()
    match_id = _extract_entity_id_from_ref(match_ref, "match")
    if match_id:
        st.session_state.selected_match_id = match_id
        if "--" in match_ref:
            raw_label = match_ref.split("--", 1)[0]
            if raw_label and raw_label not in {"match", "run", "job"}:
                st.session_state.selected_match_label = raw_label.replace("-", " ").strip()

    job_ref = str(query_params.get("job") or query_params.get("job_id") or "").strip()
    job_id = _extract_entity_id_from_ref(job_ref, "job")
    if job_id:
        st.session_state.selected_job_id = job_id
        st.session_state.portal_auto_fetch_job_id = job_id

    seek_ref = str(query_params.get("seek") or "").strip()
    if seek_ref:
        try:
            st.session_state.portal_video_seek_s = max(0, int(float(seek_ref)))
        except Exception:
            pass

    bookmark_ref = str(query_params.get("bookmark") or "").strip()
    if bookmark_ref:
        st.session_state.portal_deeplink_bookmark = bookmark_ref


def _view_code_from_nav(nav_key: str, is_technical: bool) -> str:
    if nav_key == "New Processing Run":
        return "upload"
    if nav_key == "Game Library":
        return "library"
    if nav_key == "Operations Console":
        return "monitor"
    return "home" if is_technical else "studio"


def _build_portal_query_params(
    mode_code: str,
    view_code: str,
    tenant_id: str,
    match_id: str,
    match_label: str,
    job_id: str,
    seek_s: int,
    bookmark_ref: str,
) -> Dict[str, str]:
    params: Dict[str, str] = {
        "mode": mode_code,
        "view": view_code,
    }
    tenant_clean = str(tenant_id or "").strip()
    if tenant_clean:
        params["tenant"] = tenant_clean
    match_clean = str(match_id or "").strip()
    if match_clean:
        params["match"] = _build_entity_ref(match_clean, match_label)
    job_clean = str(job_id or "").strip()
    if job_clean:
        params["job"] = _build_entity_ref(job_clean, "run")
    if int(seek_s or 0) > 0:
        params["seek"] = str(int(seek_s))
    bookmark_clean = str(bookmark_ref or "").strip()
    if bookmark_clean:
        params["bookmark"] = bookmark_clean
    return params


def _build_portal_share_link(params: Dict[str, str]) -> str:
    encoded = urlencode(params)
    base_url = str(os.getenv("VH_PORTAL_SHARE_BASE_URL", "")).strip()
    if base_url:
        delimiter = "&" if "?" in base_url else "?"
        return f"{base_url}{delimiter}{encoded}" if encoded else base_url
    return f"?{encoded}" if encoded else "?"


def _bookmark_default_index(
    labels: List[str],
    rows_by_label: Dict[str, Dict[str, Any]],
    bookmark_hint: str,
) -> int:
    hint = str(bookmark_hint or "").strip()
    if not hint:
        return 0
    for idx, label in enumerate(labels):
        row = rows_by_label.get(label) or {}
        event_id = str(row.get("event_id") or "").strip()
        if event_id and event_id == hint:
            return idx
        if str(row.get("time") or "").strip() == hint:
            return idx
        try:
            if str(int(float(row.get("occurred_s", 0.0)))) == hint:
                return idx
        except Exception:
            continue
    return 0


def _render_match_bookmark_review(
    *,
    context_key: str,
    api_base: str,
    tenant_id: str,
    token: str,
    selected_match: Optional[Dict[str, Any]],
    selected_match_id: str,
    jobs: List[Dict[str, Any]],
    is_technical: bool,
    mode_code: str,
    view_code: str,
) -> None:
    st.subheader("Bookmark Review")
    if not selected_match:
        st.info("Unable to resolve selected match details.")
        return
    if not jobs:
        st.info("Run at least one processing job to populate bookmarks for this match.")
        return

    selected_match_label = str(selected_match.get("name") or selected_match_id).strip()
    st.session_state.selected_match_label = selected_match_label

    job_picker = {
        f"{job['job_id']} | {job.get('status')} | {_iso_to_short(job.get('updated_at'))}": job
        for job in jobs
    }
    labels = list(job_picker.keys())
    previous_job_id = str(st.session_state.get("selected_job_id", "")).strip()
    default_job_id = previous_job_id
    default_index = 0
    if default_job_id:
        for idx, label in enumerate(labels):
            if str(job_picker[label].get("job_id")) == default_job_id:
                default_index = idx
                break

    selected_job_label = st.selectbox(
        "Bookmark Source Job",
        labels,
        index=default_index,
        key=f"{context_key}_review_job",
    )
    selected_job = job_picker[selected_job_label]
    selected_job_id = str(selected_job.get("job_id"))
    if previous_job_id and selected_job_id != previous_job_id:
        st.session_state.portal_selected_bookmark_ref = ""
        st.session_state.portal_deeplink_bookmark = ""
    st.session_state.selected_job_id = selected_job_id

    review_auto_refresh = st.checkbox(
        "Auto-refresh bookmarks while job is running",
        value=True,
        key=f"{context_key}_review_auto_refresh_{selected_match_id}",
    )
    live_payload = list_job_bookmarks(
        api_base=api_base,
        tenant_id=tenant_id,
        token=token,
        job_id=selected_job_id,
        limit=2000,
    )
    live_source = str(live_payload.get("source", "none"))
    live_items = list(live_payload.get("items", []))
    if live_source == "events":
        bookmark_rows = _build_bookmark_rows(selected_job, live_items)
    else:
        bookmark_rows = _bookmark_rows_from_live_items(live_items, source=live_source)
        if not bookmark_rows:
            fallback_events = list_match_events(
                api_base=api_base,
                tenant_id=tenant_id,
                token=token,
                match_id=selected_match_id,
                job_id=selected_job_id,
                limit=2000,
            )
            bookmark_rows = _build_bookmark_rows(selected_job, fallback_events)

    player_col, table_col = st.columns([1.5, 1.0])
    with player_col:
        source_video = _resolve_match_video_source(selected_match)
        if source_video:
            seek_default = int(st.session_state.portal_video_seek_s or 0)
            seek_seconds = st.number_input(
                "Seek To (seconds)",
                min_value=0,
                value=max(0, seek_default),
                step=1,
                key=f"{context_key}_seek_seconds_{selected_match_id}",
            )
            st.session_state.portal_video_seek_s = int(seek_seconds)
            st.caption("Playback uses the full source file. Jump to any bookmark and continue watching beyond clip boundaries.")
            st.video(source_video, start_time=int(seek_seconds))
        else:
            st.warning("No playable source video path found on the selected match.")

    with table_col:
        st.metric("Bookmarks", len(bookmark_rows))
        st.metric("Detected Events", len([row for row in bookmark_rows if not str(row.get("event_id", "")).startswith("bm_")]))
        st.caption(f"Bookmark source: `{live_source}`")

        if bookmark_rows:
            jump_rows = {
                f"{row['time']} | {row['event_type']} | conf={row['confidence']}": row
                for row in bookmark_rows
            }
            jump_labels = list(jump_rows.keys())
            jump_default_idx = _bookmark_default_index(
                labels=jump_labels,
                rows_by_label=jump_rows,
                bookmark_hint=str(st.session_state.get("portal_deeplink_bookmark", "")),
            )
            jump_label = st.selectbox(
                "Jump To Bookmark",
                jump_labels,
                index=jump_default_idx,
                key=f"{context_key}_jump_select_{selected_match_id}_{selected_job_id}",
            )
            selected_row = jump_rows[jump_label]
            selected_event_id = str(selected_row.get("event_id") or "").strip()
            if selected_event_id:
                st.session_state.portal_selected_bookmark_ref = selected_event_id

            if st.button(
                "Jump in Full Video",
                key=f"{context_key}_jump_btn_{selected_match_id}_{selected_job_id}",
            ):
                st.session_state.portal_video_seek_s = int(float(selected_row["occurred_s"]))
                safe_rerun()

            share_params = _build_portal_query_params(
                mode_code=mode_code,
                view_code=view_code,
                tenant_id=tenant_id,
                match_id=selected_match_id,
                match_label=selected_match_label,
                job_id=selected_job_id,
                seek_s=int(st.session_state.portal_video_seek_s or 0),
                bookmark_ref="",
            )
            bookmark_params = dict(share_params)
            bookmark_params["seek"] = str(int(float(selected_row.get("occurred_s", 0.0))))
            if selected_event_id:
                bookmark_params["bookmark"] = selected_event_id

            st.caption("Share Links")
            st.code(_build_portal_share_link(share_params), language="text")
            st.code(_build_portal_share_link(bookmark_params), language="text")

            st.caption("On-demand bookmark clip (frame-accurate extract from source video)")
            clip_col1, clip_col2 = st.columns(2)
            clip_pre = clip_col1.slider(
                "Clip Pre (s)",
                0.0,
                20.0,
                1.5,
                0.5,
                key=f"{context_key}_clip_pre",
            )
            clip_post = clip_col2.slider(
                "Clip Post (s)",
                0.0,
                30.0,
                5.0,
                0.5,
                key=f"{context_key}_clip_post",
            )
            clip_col3, clip_col4 = st.columns(2)
            clip_anchor = clip_col3.selectbox(
                "Clip Anchor",
                ["event_window", "occurred_at"],
                index=0,
                key=f"{context_key}_clip_anchor",
            )
            clip_audio = clip_col4.checkbox(
                "Include Audio",
                value=True,
                key=f"{context_key}_clip_audio",
            )

            render_disabled = (not selected_event_id) or selected_event_id.startswith("bm_")
            if st.button(
                "Render Clip On Demand",
                key=f"{context_key}_clip_render_btn",
                disabled=render_disabled,
            ):
                clip_result = api_request(
                    "POST",
                    api_base,
                    f"/matches/{selected_match_id}/events/{selected_event_id}/clip-on-demand",
                    tenant_id,
                    token,
                    json_body={
                        "pre_seconds": float(clip_pre),
                        "post_seconds": float(clip_post),
                        "anchor": clip_anchor,
                        "include_audio": bool(clip_audio),
                        "prefer_gpu": True,
                        "force_rebuild": False,
                    },
                    timeout=300,
                )
                if not clip_result["ok"]:
                    st.error(f"Clip generation failed: {clip_result['payload']}")
                else:
                    payload = clip_result["payload"]
                    clip_source = _resolve_clip_playback_source(
                        path=str(payload.get("path") or ""),
                        download_url=str(payload.get("download_url") or ""),
                    )
                    st.session_state.portal_preview_clip_source = clip_source
                    st.session_state.portal_preview_clip_summary = payload
                    reuse_text = "reused cached clip" if payload.get("reused_existing") else "rendered new clip"
                    st.success(f"Clip ready ({reuse_text}).")
            if render_disabled:
                st.caption("Clip render requires persisted event IDs. Re-run newer jobs if this row is bookmark-only.")

            selectable = [
                row
                for row in bookmark_rows
                if str(row.get("event_id", "")).strip() and not str(row.get("event_id", "")).startswith("bm_")
            ]
            if selectable:
                export_options = {
                    f"{row['time']} | {row['event_type']} | {row['event_id']}": str(row["event_id"])
                    for row in selectable
                }
                selected_export_labels = st.multiselect(
                    "Select Bookmarks For Highlight Export",
                    list(export_options.keys()),
                    key=f"{context_key}_export_selection_{selected_match_id}",
                )
                export_title = st.text_input(
                    "Export Title",
                    value="Selected Highlights",
                    key=f"{context_key}_export_title_{selected_match_id}",
                )
                export_col1, export_col2 = st.columns(2)
                export_pre = export_col1.slider(
                    "Export Pre (s)",
                    0.0,
                    20.0,
                    1.0,
                    0.5,
                    key=f"{context_key}_export_pre_{selected_match_id}",
                )
                export_post = export_col2.slider(
                    "Export Post (s)",
                    0.0,
                    30.0,
                    3.0,
                    0.5,
                    key=f"{context_key}_export_post_{selected_match_id}",
                )
                if st.button(
                    "Export Highlight Reel From Selected Bookmarks",
                    key=f"{context_key}_export_btn_{selected_match_id}",
                    disabled=not selected_export_labels,
                ):
                    event_ids = [export_options[label] for label in selected_export_labels]
                    export_result = api_request(
                        "POST",
                        api_base,
                        f"/matches/{selected_match_id}/exports/highlights",
                        tenant_id,
                        token,
                        json_body={
                            "event_ids": event_ids,
                            "pre_seconds": float(export_pre),
                            "post_seconds": float(export_post),
                            "anchor": "event_window",
                            "include_audio": True,
                            "prefer_gpu": True,
                            "title": export_title.strip() or "Selected Highlights",
                        },
                        timeout=600,
                    )
                    if not export_result["ok"]:
                        st.error(f"Highlight export failed: {export_result['payload']}")
                    else:
                        payload = export_result["payload"]
                        export_source = _resolve_clip_playback_source(
                            path=str(payload.get("path") or ""),
                            download_url=str(payload.get("download_url") or ""),
                        )
                        st.session_state.portal_export_video_source = export_source
                        st.session_state.portal_export_summary = payload
                        st.success("Highlight reel export completed.")
                if st.session_state.portal_export_video_source:
                    st.caption("Latest Highlight Export")
                    st.video(st.session_state.portal_export_video_source)
                    if is_technical:
                        st.json(st.session_state.portal_export_summary)

            st.dataframe(bookmark_rows, use_container_width=True, hide_index=True)
            if st.session_state.portal_preview_clip_source:
                st.caption("Latest Bookmark Clip")
                st.video(st.session_state.portal_preview_clip_source)
                if is_technical:
                    st.json(st.session_state.portal_preview_clip_summary)
        else:
            st.info("No bookmarks available yet for this job. Run worker and refresh.")

    selected_status = str(selected_job.get("status", "")).lower()
    if review_auto_refresh and selected_status in {"queued", "claimed", "running", "cancel_requested"}:
        time.sleep(1.5)
        safe_rerun()


def _render_match_assistant(
    *,
    api_base: str,
    tenant_id: str,
    token: str,
    selected_match_id: str,
) -> None:
    st.subheader("Match Assistant")
    st.caption("Use the configured LLM provider to summarize the match or explain a selected detected event.")

    query_key = f"assistant_query_{selected_match_id}"
    if query_key not in st.session_state:
        st.session_state[query_key] = "Summarize key events and likely missing high-impact moments."

    preset_cols = st.columns(3)
    if preset_cols[0].button("Summary Prompt", key=f"assistant_preset_summary_{selected_match_id}", use_container_width=True):
        st.session_state[query_key] = "Summarize the match, the strongest detected moments, and the overall event pattern."
    if preset_cols[1].button("Missed Moments", key=f"assistant_preset_missed_{selected_match_id}", use_container_width=True):
        st.session_state[query_key] = "Based on the detected event timeline, what likely high-impact moments or review areas should I inspect manually?"
    if preset_cols[2].button("Coaching Notes", key=f"assistant_preset_coach_{selected_match_id}", use_container_width=True):
        st.session_state[query_key] = "Turn this event timeline into short coaching notes with caveats about uncertainty."

    question = st.text_area(
        "Match Question",
        value=str(st.session_state.get(query_key, "")),
        height=120,
        key=query_key,
    )
    if st.button("Analyze Match", key=f"assistant_query_btn_{selected_match_id}", use_container_width=True):
        result = api_request(
            "POST",
            api_base,
            f"/matches/{selected_match_id}/agent/query",
            tenant_id,
            token,
            json_body={"query": question, "include_event_limit": 100},
            timeout=120,
        )
        if not result["ok"]:
            st.error(f"Assistant query failed: {result['payload']}")
        else:
            st.session_state[f"assistant_query_result_{selected_match_id}"] = result["payload"]

    query_result = st.session_state.get(f"assistant_query_result_{selected_match_id}", {})
    if query_result:
        st.caption(
            f"Provider: `{query_result.get('provider', 'fallback')}` | Model: `{query_result.get('model') or '-'}'"
        )
        st.write(str(query_result.get("answer") or ""))
        referenced = list(query_result.get("referenced_event_ids", []) or [])
        if referenced:
            st.caption(f"Referenced events: `{len(referenced)}`")

    st.divider()

    selected_event_id = str(st.session_state.get("portal_selected_bookmark_ref", "")).strip()
    explain_disabled = (not selected_event_id) or selected_event_id.startswith("bm_")
    if explain_disabled:
        st.caption("Pick a detected event in Review & Bookmarks to enable event explanation.")
    else:
        st.caption(f"Selected event: `{selected_event_id}`")

    explain_key = f"assistant_explain_{selected_match_id}"
    if explain_key not in st.session_state:
        st.session_state[explain_key] = "Explain why this event was detected and note any confidence caveats."

    explain_preset_cols = st.columns(2)
    if explain_preset_cols[0].button(
        "Why Detected?",
        key=f"assistant_explain_preset_reason_{selected_match_id}",
        use_container_width=True,
        disabled=explain_disabled,
    ):
        st.session_state[explain_key] = "Explain why this event was detected and note any confidence caveats."
    if explain_preset_cols[1].button(
        "False Positive Risk",
        key=f"assistant_explain_preset_risk_{selected_match_id}",
        use_container_width=True,
        disabled=explain_disabled,
    ):
        st.session_state[explain_key] = "Does this look like a possible false positive based on the available signals? Explain why."

    explain_question = st.text_area(
        "Event Question",
        value=str(st.session_state.get(explain_key, "")),
        height=100,
        key=explain_key,
        disabled=explain_disabled,
    )
    if st.button(
        "Explain Selected Event",
        key=f"assistant_explain_btn_{selected_match_id}",
        use_container_width=True,
        disabled=explain_disabled,
    ):
        result = api_request(
            "POST",
            api_base,
            f"/matches/{selected_match_id}/agent/explain/{selected_event_id}",
            tenant_id,
            token,
            json_body={"question": explain_question},
            timeout=120,
        )
        if not result["ok"]:
            st.error(f"Event explanation failed: {result['payload']}")
        else:
            st.session_state[f"assistant_explain_result_{selected_match_id}"] = result["payload"]

    explain_result = st.session_state.get(f"assistant_explain_result_{selected_match_id}", {})
    if explain_result:
        st.caption(
            f"Provider: `{explain_result.get('provider', 'fallback')}` | Model: `{explain_result.get('model') or '-'}'"
        )
        st.write(str(explain_result.get("answer") or ""))


if "selected_match_id" not in st.session_state:
    st.session_state.selected_match_id = ""
if "selected_job_id" not in st.session_state:
    st.session_state.selected_job_id = ""
if "selected_match_label" not in st.session_state:
    st.session_state.selected_match_label = ""
if "portal_video_seek_s" not in st.session_state:
    st.session_state.portal_video_seek_s = 0
if "portal_preview_clip_source" not in st.session_state:
    st.session_state.portal_preview_clip_source = ""
if "portal_preview_clip_summary" not in st.session_state:
    st.session_state.portal_preview_clip_summary = {}
if "portal_export_video_source" not in st.session_state:
    st.session_state.portal_export_video_source = ""
if "portal_export_summary" not in st.session_state:
    st.session_state.portal_export_summary = {}
if "portal_auto_fetch_job_id" not in st.session_state:
    st.session_state.portal_auto_fetch_job_id = ""
if "portal_flash_message" not in st.session_state:
    st.session_state.portal_flash_message = ""
if "portal_launch_cooldown_until" not in st.session_state:
    st.session_state.portal_launch_cooldown_until = 0.0
if "portal_pending_nav" not in st.session_state:
    st.session_state.portal_pending_nav = ""
if "portal_user_nav_main" not in st.session_state:
    st.session_state.portal_user_nav_main = "Studio"
if "portal_url_applied_signature" not in st.session_state:
    st.session_state.portal_url_applied_signature = ""
if "portal_deeplink_bookmark" not in st.session_state:
    st.session_state.portal_deeplink_bookmark = ""
if "portal_selected_bookmark_ref" not in st.session_state:
    st.session_state.portal_selected_bookmark_ref = ""
if "portal_studio_focus_review" not in st.session_state:
    st.session_state.portal_studio_focus_review = False

_apply_url_state_to_session()

with st.sidebar:
    st.header("Workspace")
    api_base_default = os.getenv("VH_PORTAL_API_BASE", "http://api:8000/v1")
    tenant_default = os.getenv("VH_PORTAL_TENANT", "sandbox")
    api_base = st.text_input("API Base URL", value=api_base_default, key="portal_api_base")
    tenant_id = st.text_input("Tenant", value=tenant_default, key="portal_tenant")
    token = st.text_input("Bearer Token", value="", type="password", key="portal_token")
    experience_mode = st.radio(
        "Experience",
        ["User Friendly", "Technical"],
        index=0,
        key="portal_experience_mode",
    )
    pending_nav = str(st.session_state.portal_pending_nav or "").strip()
    if pending_nav:
        if experience_mode == "Technical":
            st.session_state.portal_nav = pending_nav
        else:
            user_nav_aliases = {
                "Portal Home": "Studio",
                "New Processing Run": "Upload",
                "Game Library": "Studio",
                "Operations Console": "Run Monitor",
                "Studio Home": "Studio",
                "Upload & Run": "Upload",
                "Match Studio": "Studio",
            }
            st.session_state.portal_user_nav_main = user_nav_aliases.get(pending_nav, pending_nav)
        st.session_state.portal_pending_nav = ""

    nav = "Portal Home"
    if experience_mode == "Technical":
        nav = st.radio(
            "Portal",
            ["Portal Home", "New Processing Run", "Game Library", "Operations Console"],
            index=0,
            key="portal_nav",
        )
    if experience_mode == "Technical":
        st.caption("Tip: Keep this screen open while worker runs. Use Operations Console for live logs and kill controls.")
    else:
        st.caption("Tip: Use Studio for a YouTube-style game library. Start runs there without re-uploading.")

is_technical = experience_mode == "Technical"
if experience_mode == "User Friendly":
    user_nav_options = ["Studio", "Upload", "Run Monitor"]
    if str(st.session_state.portal_user_nav_main or "") not in user_nav_options:
        st.session_state.portal_user_nav_main = "Studio"
    user_nav = st.radio(
        "Workspace",
        user_nav_options,
        horizontal=True,
        key="portal_user_nav_main",
    )
    nav_map = {
        "Studio": "Portal Home",
        "Upload": "New Processing Run",
        "Run Monitor": "Operations Console",
    }
    nav_key = nav_map.get(user_nav, "Portal Home")
else:
    nav_key = nav

monitor_nav_label = "Operations Console" if is_technical else "Run Monitor"
ops_job_id_for_url = ""


_render_portal_header()
st.write("")
flash_message = str(st.session_state.portal_flash_message or "").strip()
if flash_message:
    st.success(flash_message)
    st.session_state.portal_flash_message = ""
if nav_key == "Portal Home":
    matches = list_matches(api_base, tenant_id, token, limit=200)
    snapshots: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = []
    running_count = 0
    completed_count = 0
    failed_count = 0

    for match in matches[:60]:
        jobs = list_match_jobs(api_base, tenant_id, token, match["match_id"], limit=1)
        latest = _latest_job(jobs)
        snapshots.append((match, latest))
        if not latest:
            continue
        status = str(latest.get("status", ""))
        if status in {"running", "claimed", "queued", "cancel_requested"}:
            running_count += 1
        elif status == "completed":
            completed_count += 1
        elif status in {"failed", "canceled"}:
            failed_count += 1

    if is_technical:
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Games In Library", len(matches))
        col_b.metric("Completed Runs", completed_count)
        col_c.metric("Active Transactions", running_count)
        col_d.metric("Failed/Canceled", failed_count)

        st.write("")
        feature_cols = st.columns(3)
        feature_cols[0].markdown(
            """
<div class="tile">
  <h4>Targeted Processing</h4>
  <p>Configure event target sets (goals, corners, shots, fouls, saves) for each run profile.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        feature_cols[1].markdown(
            """
<div class="tile">
  <h4>Rerun By Model Version</h4>
  <p>Rerun games using newer model versions and compare outputs without re-uploading video.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        feature_cols[2].markdown(
            """
<div class="tile">
  <h4>Deep Diagnostics</h4>
  <p>Inspect basic, detailed, or extreme logs per job and kill sessions immediately when needed.</p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.subheader("Announcements")
        for note in ANNOUNCEMENTS:
            st.markdown(
                f"<div class='announce'><b>{_h(note['title'])}</b><br/>{_h(note['message'])}</div>",
                unsafe_allow_html=True,
            )

        st.subheader("Recent Games")
        recent_rows: List[Dict[str, Any]] = []
        for match, latest in snapshots[:12]:
            recent_rows.append(
                {
                    "match_id": match.get("match_id"),
                    "game": match.get("name") or match.get("match_id"),
                    "match_date": match.get("match_date") or "-",
                    "teams": f"{match.get('home_team_name') or '?'} vs {match.get('away_team_name') or '?'}",
                    "latest_status": latest.get("status") if latest else "no_jobs",
                    "model_version": (latest or {}).get("config", {}).get("model_version", "-"),
                    "camera": (latest or {}).get("config", {}).get("camera_mode", "wide"),
                    "targets": _focus_targets_from_config((latest or {}).get("config", {})),
                    "updated_at": _iso_to_short((latest or {}).get("updated_at")),
                }
            )
        st.dataframe(recent_rows, use_container_width=True, hide_index=True)
    else:
        st.subheader("Studio")

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Games", len(matches))
        col_b.metric("Completed Runs", completed_count)
        col_c.metric("Live Runs", running_count)
        col_d.metric("Needs Attention", failed_count)

        filters_col1, filters_col2 = st.columns([2.0, 1.0])
        library_query = filters_col1.text_input(
            "Search Library",
            value="",
            placeholder="Search by game, team, date, or match id",
            key="portal_studio_search",
        )
        status_filter = filters_col2.selectbox(
            "Status Filter",
            ["all", "live", "completed", "failed", "no_runs"],
            index=0,
            key="portal_studio_status_filter",
        )

        filtered_snapshots: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = []
        query = library_query.strip().lower()
        for match, latest in snapshots:
            status = str((latest or {}).get("status", "no_runs")).lower()
            status_bucket = "no_runs"
            if status in {"running", "claimed", "queued", "cancel_requested"}:
                status_bucket = "live"
            elif status == "completed":
                status_bucket = "completed"
            elif status in {"failed", "canceled"}:
                status_bucket = "failed"

            if status_filter != "all" and status_bucket != status_filter:
                continue
            if query:
                haystack = " ".join(
                    [
                        str(match.get("name") or ""),
                        str(match.get("match_id") or ""),
                        str(match.get("home_team_name") or ""),
                        str(match.get("away_team_name") or ""),
                        str(match.get("match_date") or ""),
                        str(status_bucket),
                    ]
                ).lower()
                if query not in haystack:
                    continue
            filtered_snapshots.append((match, latest))

        if not filtered_snapshots:
            st.info("No games match the current filters.")
        else:
            st.subheader("Game Library")
            card_cols = st.columns(3)
            for idx, (match, latest) in enumerate(filtered_snapshots[:90]):
                match_id = str(match.get("match_id"))
                game_name = str(match.get("name") or match_id)
                teams = f"{match.get('home_team_name') or '?'} vs {match.get('away_team_name') or '?'}"
                match_date = str(match.get("match_date") or "-")
                latest_status = str((latest or {}).get("status", "no_runs"))
                latest_model = str((latest or {}).get("config", {}).get("model_version", "-"))
                latest_camera = str((latest or {}).get("config", {}).get("camera_mode", "wide"))
                latest_targets = _focus_targets_from_config((latest or {}).get("config", {}))
                latest_bookmarks = int((latest or {}).get("result", {}).get("bookmarks_count", 0) or 0)

                chip_classes = "studio-chip"
                if latest_status in {"running", "claimed", "queued", "cancel_requested"}:
                    chip_classes = "studio-chip live"
                elif latest_status == "completed":
                    chip_classes = "studio-chip good"
                elif latest_status in {"failed", "canceled"}:
                    chip_classes = "studio-chip bad"

                with card_cols[idx % 3]:
                    st.markdown(
                        f"""
<div class="studio-card">
  <h4>{_h(game_name)}</h4>
  <p class="studio-muted">{_h(teams)}</p>
  <p class="studio-muted">Date: {_h(match_date)}</p>
  <span class="{chip_classes}">{_h(latest_status)}</span>
  <span class="studio-chip">model: {_h(latest_model)}</span>
  <span class="studio-chip">camera: {_h(latest_camera)}</span>
  <span class="studio-chip">bookmarks: {latest_bookmarks}</span>
  <p class="studio-muted">targets: {_h(latest_targets)}</p>
</div>
""",
                        unsafe_allow_html=True,
                    )
                    action_col1, action_col2, action_col3 = st.columns(3)
                    if action_col1.button("Open", key=f"studio_open_match_{match_id}"):
                        st.session_state.selected_match_id = match_id
                        st.session_state.selected_match_label = game_name
                        st.session_state.portal_studio_focus_review = False
                        safe_rerun()

                    if action_col2.button("Review", key=f"studio_review_match_{match_id}"):
                        st.session_state.selected_match_id = match_id
                        st.session_state.selected_match_label = game_name
                        latest_job_id = str((latest or {}).get("job_id") or "").strip()
                        if latest_job_id:
                            st.session_state.selected_job_id = latest_job_id
                        st.session_state.portal_studio_focus_review = True
                        st.session_state.portal_flash_message = "Match selected. Opening Review & Bookmarks."
                        safe_rerun()

                    source_video = _resolve_match_video_source(match)
                    if action_col3.button(
                        "Analyze",
                        key=f"studio_new_run_{match_id}",
                        disabled=not bool(source_video),
                    ):
                        base_config = dict((latest or {}).get("config", {}) or _recommended_job_config())
                        base_config["run_created_at"] = datetime.utcnow().isoformat()
                        base_config["run_created_from"] = "studio_card_quick_launch"
                        create_result = api_request(
                            "POST",
                            api_base,
                            f"/matches/{match_id}/jobs",
                            tenant_id,
                            token,
                            json_body={"config": base_config},
                        )
                        if not create_result["ok"]:
                            st.error(f"Failed to queue analysis for {game_name}: {create_result['payload']}")
                        else:
                            new_job_id = str(create_result["payload"]["job_id"])
                            st.session_state.selected_match_id = match_id
                            st.session_state.selected_match_label = game_name
                            st.session_state.selected_job_id = new_job_id
                            st.session_state.portal_studio_focus_review = False
                            st.session_state.portal_auto_fetch_job_id = new_job_id
                            st.session_state.portal_pending_nav = monitor_nav_label
                            st.session_state.portal_flash_message = f"Analysis queued for {game_name} ({new_job_id})."
                            safe_rerun()

            selector = {
                f"{(m.get('name') or m.get('match_id'))} | {m.get('home_team_name') or '?'} vs {m.get('away_team_name') or '?'} | {m.get('match_date') or '-'}": m.get("match_id")
                for m, _ in filtered_snapshots
            }
            selector_labels = list(selector.keys())
            default_match_id = str(st.session_state.selected_match_id or "")
            default_idx = 0
            if default_match_id:
                for idx, label in enumerate(selector_labels):
                    if str(selector[label]) == default_match_id:
                        default_idx = idx
                        break
            selected_label = st.selectbox(
                "Active Match Workspace",
                selector_labels,
                index=default_idx,
                key="portal_studio_active_match",
            )
            selected_match_id = str(selector[selected_label])
            st.session_state.selected_match_id = selected_match_id
            selected_match = next((m for m, _ in filtered_snapshots if str(m.get("match_id")) == selected_match_id), None)
            if selected_match:
                st.session_state.selected_match_label = str(selected_match.get("name") or selected_match_id)
            jobs = list_match_jobs(api_base, tenant_id, token, selected_match_id, limit=200)

            if selected_match:
                st.subheader(f"Match Analysis: {selected_match.get('name') or selected_match_id}")
                workspace_tabs = st.tabs(["Overview", "Live Analysis", "Runs", "Exports", "Assistant"])
                latest_job = _latest_job(jobs)
                source_video = _resolve_match_video_source(selected_match)

                if bool(st.session_state.portal_studio_focus_review):
                    st.subheader("Review & Bookmarks")
                    review_col1, review_col2 = st.columns([1.3, 1.0])
                    review_col1.caption("Direct review mode. Jump bookmarks, render clips, and export without nested tabs.")
                    if review_col2.button(
                        "Close Review Mode",
                        key=f"studio_review_close_{selected_match_id}",
                        use_container_width=True,
                    ):
                        st.session_state.portal_studio_focus_review = False
                        safe_rerun()

                    _render_match_bookmark_review(
                        context_key="studio_review_focus",
                        api_base=api_base,
                        tenant_id=tenant_id,
                        token=token,
                        selected_match=selected_match,
                        selected_match_id=selected_match_id,
                        jobs=jobs,
                        is_technical=is_technical,
                        mode_code="technical" if is_technical else "user",
                        view_code="studio",
                    )
                    st.divider()

                with workspace_tabs[0]:
                    ov_col1, ov_col2, ov_col3, ov_col4 = st.columns(4)
                    ov_col1.metric("Total Runs", len(jobs))
                    ov_col2.metric(
                        "Active Runs",
                        len(
                            [
                                job
                                for job in jobs
                                if str(job.get("status", "")).lower()
                                in {"queued", "claimed", "running", "cancel_requested"}
                            ]
                        ),
                    )
                    ov_col3.metric("Latest Status", str((latest_job or {}).get("status", "no_runs")))
                    ov_col4.metric("Latest Bookmarks", int((latest_job or {}).get("result", {}).get("bookmarks_count", 0) or 0))

                    action_col1, action_col2, action_col3 = st.columns(3)
                    if action_col1.button(
                        "Start New Analysis For This Match",
                        key=f"studio_workspace_start_{selected_match_id}",
                        disabled=not bool(source_video),
                        use_container_width=True,
                    ):
                        next_config = dict((latest_job or {}).get("config", {}) or _recommended_job_config())
                        next_config["run_created_at"] = datetime.utcnow().isoformat()
                        next_config["run_created_from"] = "studio_workspace"
                        create_result = api_request(
                            "POST",
                            api_base,
                            f"/matches/{selected_match_id}/jobs",
                            tenant_id,
                            token,
                            json_body={"config": next_config},
                        )
                        if not create_result["ok"]:
                            st.error(f"Failed to queue analysis: {create_result['payload']}")
                        else:
                            new_job_id = str(create_result["payload"]["job_id"])
                            st.session_state.selected_job_id = new_job_id
                            st.session_state.portal_studio_focus_review = False
                            st.session_state.portal_auto_fetch_job_id = new_job_id
                            st.session_state.portal_pending_nav = monitor_nav_label
                            st.session_state.portal_flash_message = f"Analysis queued. job_id={new_job_id}"
                            safe_rerun()
                    if action_col2.button(
                        "Open Run Monitor",
                        key=f"studio_workspace_monitor_{selected_match_id}",
                        use_container_width=True,
                    ):
                        st.session_state.portal_pending_nav = monitor_nav_label
                        safe_rerun()
                    if action_col3.button(
                        "Open Review & Bookmarks",
                        key=f"studio_workspace_review_{selected_match_id}",
                        use_container_width=True,
                        disabled=not bool(jobs),
                    ):
                        st.session_state.portal_studio_focus_review = True
                        safe_rerun()

                    if source_video:
                        seek_seconds = st.number_input(
                            "Video Start (seconds)",
                            min_value=0,
                            value=int(st.session_state.portal_video_seek_s or 0),
                            step=1,
                            key=f"studio_workspace_seek_{selected_match_id}",
                        )
                        st.session_state.portal_video_seek_s = int(seek_seconds)
                        st.video(source_video, start_time=int(seek_seconds))
                    else:
                        st.warning("This match has no uploaded source video yet. Upload from the Upload page.")

                    if latest_job:
                        latest_artifacts = list((latest_job.get("result", {}) or {}).get("artifacts", []) or [])
                        if latest_artifacts:
                            st.caption("Latest run artifacts")
                            st.dataframe(
                                [{"path": path} for path in latest_artifacts],
                                use_container_width=True,
                                hide_index=True,
                            )

                with workspace_tabs[1]:
                    if not jobs:
                        st.info("No runs yet for this match.")
                    else:
                        active_job = next(
                            (
                                job
                                for job in jobs
                                if str(job.get("status", "")).lower()
                                in {"queued", "claimed", "running", "cancel_requested"}
                            ),
                            None,
                        )
                        default_job_id = str((active_job or latest_job or {}).get("job_id") or "")
                        job_options = {
                            f"{job.get('job_id')} | {job.get('status')} | {_iso_to_short(job.get('updated_at'))}": job
                            for job in jobs
                        }
                        option_labels = list(job_options.keys())
                        default_job_idx = 0
                        for idx, label in enumerate(option_labels):
                            if str(job_options[label].get("job_id")) == default_job_id:
                                default_job_idx = idx
                                break
                        selected_job_label = st.selectbox(
                            "Run",
                            option_labels,
                            index=default_job_idx,
                            key=f"studio_live_job_{selected_match_id}",
                        )
                        selected_job = job_options[selected_job_label]
                        selected_job_id = str(selected_job.get("job_id"))
                        st.session_state.selected_job_id = selected_job_id

                        st.markdown(
                            _build_stage_tracker(selected_job.get("stage"), selected_job.get("status")),
                            unsafe_allow_html=True,
                        )
                        st.progress(float(selected_job.get("progress", 0.0)))
                        st.caption(
                            f"Model `{selected_job.get('config', {}).get('model_version', '-')}` | "
                            f"Camera `{selected_job.get('config', {}).get('camera_mode', 'wide')}` | "
                            f"Targets `{_focus_targets_from_config(selected_job.get('config', {}))}`"
                        )

                        live_payload = list_job_bookmarks(api_base, tenant_id, token, selected_job_id, limit=5000)
                        live_items = list(live_payload.get("items", []))
                        if str(live_payload.get("source")) == "events":
                            live_rows = _build_bookmark_rows(selected_job, live_items)
                        else:
                            live_rows = _bookmark_rows_from_live_items(live_items, source=str(live_payload.get("source", "manifest")))
                        if live_rows:
                            st.dataframe(live_rows, use_container_width=True, hide_index=True)
                        else:
                            st.info("No bookmarks yet for this run.")

                        live_status = str(selected_job.get("status", "")).lower()
                        auto_refresh_live = st.checkbox(
                            "Auto-refresh while run is active",
                            value=True,
                            key=f"studio_live_autorefresh_{selected_match_id}",
                        )
                        if auto_refresh_live and live_status in {"queued", "claimed", "running", "cancel_requested"}:
                            time.sleep(1.5)
                            safe_rerun()

                with workspace_tabs[2]:
                    if not jobs:
                        st.info("No runs available yet.")
                    else:
                        run_rows = []
                        for job in jobs:
                            run_rows.append(
                                {
                                    "job_id": job.get("job_id"),
                                    "status": job.get("status"),
                                    "stage": job.get("stage"),
                                    "model": (job.get("config", {}) or {}).get("model_version", "-"),
                                    "camera": (job.get("config", {}) or {}).get("camera_mode", "wide"),
                                    "targets": _focus_targets_from_config(job.get("config", {}) or {}),
                                    "analysis_only": bool((job.get("config", {}) or {}).get("analysis_only", False)),
                                    "bookmarks": int((job.get("result", {}) or {}).get("bookmarks_count", 0) or 0),
                                    "updated_at": _iso_to_short(job.get("updated_at")),
                                }
                            )
                        st.dataframe(run_rows, use_container_width=True, hide_index=True)

                        terminal_jobs = [
                            job
                            for job in jobs
                            if str(job.get("status", "")).lower() in {"completed", "failed", "canceled"}
                        ]
                        delete_options = {
                            f"{job.get('job_id')} | {job.get('status')} | {_iso_to_short(job.get('updated_at'))}": job
                            for job in terminal_jobs
                        }
                        selected_delete = st.multiselect(
                            "Delete old runs",
                            list(delete_options.keys()),
                            key=f"studio_delete_runs_{selected_match_id}",
                        )
                        if st.button(
                            "Delete Selected Runs",
                            key=f"studio_delete_runs_btn_{selected_match_id}",
                            disabled=not selected_delete,
                        ):
                            success_count = 0
                            failed_count = 0
                            for label in selected_delete:
                                target = delete_options.get(label) or {}
                                target_job_id = str(target.get("job_id") or "")
                                if not target_job_id:
                                    continue
                                delete_result = api_request("DELETE", api_base, f"/jobs/{target_job_id}", tenant_id, token)
                                if delete_result["ok"]:
                                    success_count += 1
                                else:
                                    failed_count += 1
                            st.session_state.portal_flash_message = (
                                f"Deleted runs for match {selected_match_id}: success={success_count}, failed={failed_count}"
                            )
                            safe_rerun()

                with workspace_tabs[3]:
                    metadata = selected_match.get("metadata", {}) if isinstance(selected_match.get("metadata"), dict) else {}
                    highlight_exports = list(metadata.get("highlight_exports", []) or [])
                    generated_clips = list(metadata.get("generated_clips", []) or [])

                    ex_col1, ex_col2, ex_col3 = st.columns(3)
                    ex_col1.metric("Highlight Exports", len(highlight_exports))
                    ex_col2.metric("Bookmark Clips", len(generated_clips))
                    ex_col3.metric("Assets", len(list(metadata.get("assets", []) or [])))

                    active_non_analysis = next(
                        (
                            job
                            for job in jobs
                            if str(job.get("status", "")).lower() in {"queued", "claimed", "running", "cancel_requested"}
                            and not bool((job.get("config", {}) or {}).get("analysis_only", False))
                        ),
                        None,
                    )
                    if active_non_analysis:
                        st.info(
                            f"Export is in progress for run `{active_non_analysis.get('job_id')}`. Bookmark detections update in Live Analysis while clips are being prepared."
                        )

                    export_rows = [
                        {
                            "export_id": item.get("export_id"),
                            "title": item.get("title") or "Selected Highlights",
                            "clips": item.get("clip_count"),
                            "duration_s": round(float(item.get("duration_ms", 0) or 0) / 1000.0, 1),
                            "created_at": _iso_to_short(item.get("created_at")),
                            "path": item.get("path"),
                        }
                        for item in sorted(
                            highlight_exports,
                            key=lambda entry: str(entry.get("created_at") or ""),
                            reverse=True,
                        )
                    ]
                    if export_rows:
                        st.dataframe(export_rows, use_container_width=True, hide_index=True)
                        export_options = {
                            f"{row['title']} | {row['created_at']} | {row['clips']} clips": row["path"]
                            for row in export_rows
                            if row.get("path")
                        }
                        if export_options:
                            selected_export = st.selectbox(
                                "Preview Export",
                                list(export_options.keys()),
                                key=f"studio_export_preview_{selected_match_id}",
                            )
                            st.video(str(export_options[selected_export]))
                    else:
                        st.caption("No highlight exports yet for this match.")

                with workspace_tabs[4]:
                    _render_match_assistant(
                        api_base=api_base,
                        tenant_id=tenant_id,
                        token=token,
                        selected_match_id=selected_match_id,
                    )

elif nav_key == "New Processing Run":
    st.subheader("Launch Processing")
    mode = st.radio("Run Type", ["Upload New Game", "Rerun Existing Game"], horizontal=True, key="portal_run_mode")

    matches = list_matches(api_base, tenant_id, token, limit=200)
    models = list_training_models(api_base, tenant_id, token)
    model_versions = sorted({str(item.get("version")) for item in models if item.get("version")}, reverse=True)
    if "event-v0" not in model_versions:
        model_versions.append("event-v0")
    model_versions = list(dict.fromkeys(model_versions))

    if mode == "Upload New Game":
        left, right = st.columns([1.2, 1.8])
        with left:
            match_name = st.text_input("Game Name", value="U13 Matchday", key="portal_new_match_name")
            home_team = st.text_input("Home Team", value="Home", key="portal_new_home_team")
            away_team = st.text_input("Away Team", value="Away", key="portal_new_away_team")
            match_date = st.text_input("Match Date", value=datetime.utcnow().strftime("%Y-%m-%d"), key="portal_new_match_date")
            profile_name = st.selectbox(
                "Processing Profile",
                ["Balanced", "Offense Focus", "Set Piece Focus", "Discipline Review", "Custom"],
                key="portal_new_profile_name",
            )
            model_version = st.selectbox("Model Version", model_versions, index=0, key="portal_new_model_version")

        with right:
            upload = st.file_uploader(
                "Upload Game Video",
                type=["mp4", "mov", "mkv", "avi", "m4v"],
                key="portal_new_upload",
            )
            selected_targets = st.multiselect(
                "Event Targets",
                EVENT_TARGET_OPTIONS,
                default=[],
                key="portal_new_targets",
                help="Leave empty for broad highlight generation.",
            )
            custom_targets = st.text_input(
                "Additional Custom Targets (comma separated)",
                value="",
                key="portal_new_custom_targets",
            )

            cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
            pre_seconds = cfg_col1.slider("Pre Buffer", 0.5, 10.0, 2.0, 0.5, key="portal_new_pre")
            post_seconds = cfg_col2.slider("Post Buffer", 1.0, 20.0, 6.0, 0.5, key="portal_new_post")
            min_clip = cfg_col3.slider("Min Clip", 1.0, 15.0, 4.0, 0.5, key="portal_new_min_clip")

            cfg_col4, cfg_col5, cfg_col6 = st.columns(3)
            speed_sens = cfg_col4.slider("Speed Sensitivity", 1.0, 4.0, 2.0, 0.1, key="portal_new_speed_sens")
            audio_sens = cfg_col5.slider("Audio Sensitivity", 1.0, 4.0, 2.0, 0.1, key="portal_new_audio_sens")
            thread_count = cfg_col6.number_input("Threads (0=auto)", min_value=0, max_value=32, value=0, key="portal_new_threads")

            cam_col1, cam_col2, cam_col3 = st.columns(3)
            camera_mode = cam_col1.selectbox(
                "Camera Mode",
                ["wide", "follow_action", "follow_player"],
                index=0,
                key="portal_new_camera_mode",
                help="`follow_action` keeps the tracked player central while nudging toward nearby ball action.",
            )
            zoom_factor = cam_col2.slider("Zoom Factor", 1.0, 2.5, 1.6, 0.1, key="portal_new_zoom_factor")
            cam_col3.caption("Use `wide` for the original frame. Follow-cam modes are best for player-centric exports.")

            opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
            no_audio = opt_col1.checkbox("Disable Audio Detection", value=False, key="portal_new_no_audio")
            overlay = opt_col2.checkbox("Generate Spotlight Overlay", value=False, key="portal_new_overlay")
            require_gpu = opt_col3.checkbox("Require GPU", value=False, key="portal_new_require_gpu")
            analysis_only = opt_col4.checkbox(
                "Analysis Only (No Clips)",
                value=False,
                key="portal_new_analysis_only",
                help="Generate bookmark table/events quickly without writing highlight video clips.",
            )

            player_roi_payload = _render_player_roi_selector(
                context_key="portal_new",
                default_enabled=False,
                existing_roi=None,
                upload_bytes=(upload.getvalue() if upload is not None else None),
                upload_name=(upload.name if upload is not None else ""),
            )

        cooldown_until = float(st.session_state.portal_launch_cooldown_until or 0.0)
        launch_locked = time.time() < cooldown_until
        if launch_locked:
            wait_s = max(1, int(round(cooldown_until - time.time())))
            st.caption(f"Launch protection active. Please wait {wait_s}s before submitting again.")

        if st.button(
            "Start New Game Processing",
            type="primary",
            use_container_width=True,
            key="portal_new_run_btn",
            disabled=launch_locked,
        ):
            st.session_state.portal_launch_cooldown_until = time.time() + 8.0
            if upload is None:
                st.error("Upload a game video first.")
                st.session_state.portal_launch_cooldown_until = 0.0
            elif player_roi_payload.get("enabled") and not player_roi_payload.get("roi"):
                st.error("Player ROI lock is enabled, but the first-frame selector is not ready. Uncheck it or choose a readable video.")
                st.session_state.portal_launch_cooldown_until = 0.0
            else:
                target_list = [item.strip().lower() for item in selected_targets if str(item).strip()]
                if custom_targets.strip():
                    target_list.extend([item.strip().lower() for item in custom_targets.split(",") if item.strip()])
                target_list = list(dict.fromkeys(target_list))

                profile_defaults = _profile_defaults(profile_name, target_list)
                if profile_name != "Custom":
                    pre_seconds = float(profile_defaults.get("pre_seconds", pre_seconds))
                    post_seconds = float(profile_defaults.get("post_seconds", post_seconds))
                    min_clip = float(profile_defaults.get("min_clip_duration", min_clip))
                    speed_sens = float(profile_defaults.get("speed_sensitivity", speed_sens))
                    audio_sens = float(profile_defaults.get("audio_sensitivity", audio_sens))
                    target_list = list(dict.fromkeys(target_list + list(profile_defaults.get("focus_event_types", []))))

                match_payload = {
                    "name": match_name,
                    "home_team_name": home_team,
                    "away_team_name": away_team,
                    "match_date": match_date,
                    "source_video_path": "",
                    "metadata": {
                        "profile_name": profile_name,
                        "requested_targets": target_list,
                        "model_version_hint": model_version,
                        "created_from": "portal_new_game",
                    },
                }
                match_result = api_request("POST", api_base, "/matches", tenant_id, token, json_body=match_payload)
                if not match_result["ok"]:
                    st.error(f"Failed to create match: {match_result['payload']}")
                    st.session_state.portal_launch_cooldown_until = 0.0
                else:
                    match_id = match_result["payload"]["match_id"]
                    files = {
                        "file": (
                            upload.name,
                            upload.getvalue(),
                            upload.type or "application/octet-stream",
                        )
                    }
                    upload_result = api_request(
                        "POST",
                        api_base,
                        f"/matches/{match_id}/assets/upload",
                        tenant_id,
                        token,
                        files=files,
                        timeout=900,
                    )
                    if not upload_result["ok"]:
                        st.error(f"Match created, but upload failed: {upload_result['payload']}")
                        st.session_state.portal_launch_cooldown_until = 0.0
                    else:
                        job_config: Dict[str, Any] = {
                            "profile_name": profile_name,
                            "model_version": model_version,
                            "focus_event_types": target_list,
                            "pre_seconds": float(pre_seconds),
                            "post_seconds": float(post_seconds),
                            "min_clip_duration": float(min_clip),
                            "speed_sensitivity": float(speed_sens),
                            "audio_sensitivity": float(audio_sens),
                            "camera_mode": str(camera_mode),
                            "zoom_factor": float(zoom_factor),
                            "overlay": bool(overlay),
                            "no_audio": bool(no_audio),
                            "require_gpu": bool(require_gpu),
                            "analysis_only": bool(analysis_only),
                            "run_created_at": datetime.utcnow().isoformat(),
                            "run_created_from": "portal_ui",
                        }
                        if player_roi_payload.get("roi"):
                            job_config["player_roi"] = dict(player_roi_payload["roi"])
                        if int(thread_count) > 0:
                            job_config["threads"] = int(thread_count)

                        job_result = api_request(
                            "POST",
                            api_base,
                            f"/matches/{match_id}/jobs",
                            tenant_id,
                            token,
                            json_body={"config": job_config},
                        )
                        if not job_result["ok"]:
                            st.error(f"Video uploaded, but job creation failed: {job_result['payload']}")
                            st.session_state.portal_launch_cooldown_until = 0.0
                        else:
                            job_id = job_result["payload"]["job_id"]
                            st.session_state.selected_match_id = match_id
                            st.session_state.selected_match_label = str(match_name or match_id)
                            st.session_state.selected_job_id = job_id
                            st.session_state.portal_auto_fetch_job_id = job_id
                            st.session_state.portal_pending_nav = monitor_nav_label
                            st.session_state.portal_flash_message = (
                                f"Processing transaction created. match_id={match_id} job_id={job_id}"
                            )
                            safe_rerun()
    else:
        if not matches:
            st.warning("No games available yet. Upload a new game first.")
        else:
            match_lookup = {str(m.get("match_id")): m for m in matches}
            match_options = {f"{m.get('name') or m['match_id']} ({m['match_id']})": m["match_id"] for m in matches}
            selected_label = st.selectbox("Choose Game", list(match_options.keys()), key="portal_rerun_match_select")
            match_id = match_options[selected_label]
            selected_match_obj = match_lookup.get(str(match_id)) or {}
            jobs = list_match_jobs(api_base, tenant_id, token, match_id, limit=200)
            if not jobs:
                st.warning("Selected game has no processing jobs yet.")
            else:
                source_map = {
                    f"{j['job_id']} | {j.get('status')} | {(_iso_to_short(j.get('created_at')))}": j for j in jobs
                }
                source_label = st.selectbox("Source Job", list(source_map.keys()), key="portal_rerun_source_job")
                source_job = source_map[source_label]

                if is_technical:
                    st.write("Source Configuration")
                    st.json(source_job.get("config", {}))
                else:
                    st.caption("Reprocess from this run or override model/targets below.")

                rerun_reason = st.text_input("Rerun Reason", value="model-upgrade-rerun", key="portal_rerun_reason")
                rerun_model = st.text_input(
                    "Override Model Version (optional)",
                    value="",
                    placeholder=source_job.get("config", {}).get("model_version", "event-v0"),
                    key="portal_rerun_model",
                )
                rerun_targets = st.multiselect(
                    "Override Event Targets (optional)",
                    EVENT_TARGET_OPTIONS,
                    default=[],
                    key="portal_rerun_targets",
                )
                rerun_custom_targets = st.text_input(
                    "Additional Custom Targets (comma separated)",
                    value="",
                    key="portal_rerun_custom_targets",
                )
                rerun_analysis_only = st.checkbox(
                    "Analysis Only (No Clips)",
                    value=bool(source_job.get("config", {}).get("analysis_only", False)),
                    key="portal_rerun_analysis_only",
                )
                source_config = dict(source_job.get("config", {}) or {})
                rerun_camera_options = ["wide", "follow_action", "follow_player"]
                source_camera_mode = str(source_config.get("camera_mode") or "wide")
                if source_camera_mode not in rerun_camera_options:
                    source_camera_mode = "wide"
                rerun_cam_col1, rerun_cam_col2 = st.columns(2)
                rerun_camera_mode = rerun_cam_col1.selectbox(
                    "Camera Mode",
                    rerun_camera_options,
                    index=rerun_camera_options.index(source_camera_mode),
                    key="portal_rerun_camera_mode",
                )
                rerun_zoom_factor = rerun_cam_col2.slider(
                    "Zoom Factor",
                    1.0,
                    2.5,
                    float(source_config.get("zoom_factor", 1.6)),
                    0.1,
                    key="portal_rerun_zoom_factor",
                )

                rerun_cooldown_until = float(st.session_state.portal_launch_cooldown_until or 0.0)
                rerun_locked = time.time() < rerun_cooldown_until
                if rerun_locked:
                    wait_s = max(1, int(round(rerun_cooldown_until - time.time())))
                    st.caption(f"Launch protection active. Please wait {wait_s}s before submitting again.")

                if st.button("Start Rerun", type="primary", key="portal_rerun_btn", disabled=rerun_locked):
                    st.session_state.portal_launch_cooldown_until = time.time() + 6.0
                    overrides: Dict[str, Any] = {}
                    if rerun_model.strip():
                        overrides["model_version"] = rerun_model.strip()
                    targets = [item.strip().lower() for item in rerun_targets if str(item).strip()]
                    if rerun_custom_targets.strip():
                        targets.extend([item.strip().lower() for item in rerun_custom_targets.split(",") if item.strip()])
                    targets = list(dict.fromkeys(targets))
                    if targets:
                        overrides["focus_event_types"] = targets
                    overrides["analysis_only"] = bool(rerun_analysis_only)
                    overrides["camera_mode"] = str(rerun_camera_mode)
                    overrides["zoom_factor"] = float(rerun_zoom_factor)

                    rerun_result = api_request(
                        "POST",
                        api_base,
                        f"/jobs/{source_job['job_id']}/rerun",
                        tenant_id,
                        token,
                        json_body={"config_overrides": overrides, "reason": rerun_reason},
                    )
                    if not rerun_result["ok"]:
                        st.error(f"Rerun failed: {rerun_result['payload']}")
                        st.session_state.portal_launch_cooldown_until = 0.0
                    else:
                        rerun_job_id = rerun_result["payload"]["job_id"]
                        st.session_state.selected_match_id = match_id
                        st.session_state.selected_match_label = str(selected_match_obj.get("name") or match_id)
                        st.session_state.selected_job_id = rerun_job_id
                        st.session_state.portal_auto_fetch_job_id = rerun_job_id
                        st.session_state.portal_pending_nav = monitor_nav_label
                        st.session_state.portal_flash_message = f"Rerun queued: {rerun_job_id}"
                        safe_rerun()


elif nav_key == "Game Library":
    matches = list_matches(api_base, tenant_id, token, limit=200)
    if not matches:
        st.warning("No games found for this tenant yet.")
    else:
        models = list_training_models(api_base, tenant_id, token)
        model_versions = sorted({str(item.get("version")) for item in models if item.get("version")}, reverse=True)
        if "event-v0" not in model_versions:
            model_versions.append("event-v0")
        model_versions = list(dict.fromkeys(model_versions))

        rows: List[Dict[str, Any]] = []
        match_jobs_cache: Dict[str, List[Dict[str, Any]]] = {}
        for match in matches:
            match_id = match["match_id"]
            jobs = list_match_jobs(api_base, tenant_id, token, match_id, limit=30)
            match_jobs_cache[match_id] = jobs
            latest = _latest_job(jobs)
            rows.append(
                {
                    "match_id": match_id,
                    "game": match.get("name") or match_id,
                    "date": match.get("match_date") or "-",
                    "teams": f"{match.get('home_team_name') or '?'} vs {match.get('away_team_name') or '?'}",
                    "jobs": len(jobs),
                    "latest_status": latest.get("status") if latest else "no_jobs",
                    "latest_stage": latest.get("stage") if latest else "-",
                    "model_version": (latest or {}).get("config", {}).get("model_version", "-"),
                    "camera": (latest or {}).get("config", {}).get("camera_mode", "wide"),
                    "targets": _focus_targets_from_config((latest or {}).get("config", {})),
                    "analysis_only": bool((latest or {}).get("config", {}).get("analysis_only", False)),
                    "bookmarks": int((latest or {}).get("result", {}).get("bookmarks_count", 0) or 0),
                    "latest_update": _iso_to_short((latest or {}).get("updated_at")),
                    "latest_update_raw": str((latest or {}).get("updated_at") or ""),
                }
            )

        rows = sorted(rows, key=lambda item: str(item.get("latest_update_raw", "")), reverse=True)
        search_query = st.text_input(
            "Search Games",
            value="",
            placeholder="Search by game name, team, date, or match id",
            key="portal_library_search",
        )
        query = search_query.strip().lower()
        filtered_rows = rows
        if query:
            filtered_rows = [
                row
                for row in rows
                if query in " ".join(
                    [
                        str(row.get("game", "")),
                        str(row.get("match_id", "")),
                        str(row.get("teams", "")),
                        str(row.get("date", "")),
                        str(row.get("latest_status", "")),
                    ]
                ).lower()
            ]
            if not filtered_rows:
                st.info("No games matched your search. Showing full list for selection.")

        table_rows = [{key: value for key, value in row.items() if key != "latest_update_raw"} for row in filtered_rows or rows]
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        selector = {
            f"{row['game']} | {row['date']} | {row['latest_status']}": row["match_id"]
            for row in (filtered_rows or rows)
        }
        selector_labels = list(selector.keys())
        default_match = str(st.session_state.selected_match_id or "").strip()
        selected_index = 0
        if default_match:
            for idx, label in enumerate(selector_labels):
                if selector[label] == default_match:
                    selected_index = idx
                    break
        selected_label = st.selectbox("Game Details", selector_labels, index=selected_index, key="portal_library_select")
        selected_match_id = selector[selected_label]
        st.session_state.selected_match_id = selected_match_id
        jobs = match_jobs_cache.get(selected_match_id, [])
        selected_match = next((item for item in matches if item.get("match_id") == selected_match_id), None)
        if selected_match:
            st.session_state.selected_match_label = str(selected_match.get("name") or selected_match_id)

        st.subheader("Match Workspace")
        if not selected_match:
            st.warning("Unable to resolve selected match.")
        else:
            latest_job = _latest_job(jobs)
            latest_config = dict((latest_job or {}).get("config", {}) or {})
            source_video = _resolve_match_video_source(selected_match)
            source_assets = list(((selected_match.get("metadata") or {}).get("assets", []) or []))

            ws_col1, ws_col2, ws_col3, ws_col4 = st.columns(4)
            ws_col1.metric("Match", selected_match.get("name") or selected_match_id)
            ws_col2.metric("Jobs", len(jobs))
            ws_col3.metric("Source Video", "Available" if source_video else "Missing")
            ws_col4.metric("Uploaded Assets", len(source_assets))
            if source_video:
                st.caption(f"Source file reused for all reprocessing runs: `{source_video}`")
            else:
                st.warning("No source video path found on this match. Upload an asset once, then reprocess from here.")

            quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
            if quick_col1.button("Start New Analysis", key=f"portal_match_quick_start_{selected_match_id}", disabled=not source_video):
                next_config = dict(latest_config or _recommended_job_config())
                next_config["run_created_at"] = datetime.utcnow().isoformat()
                next_config["run_created_from"] = "portal_match_workspace_quick_start"
                create_result = api_request(
                    "POST",
                    api_base,
                    f"/matches/{selected_match_id}/jobs",
                    tenant_id,
                    token,
                    json_body={"config": next_config},
                )
                if not create_result["ok"]:
                    st.error(f"Failed to queue analysis: {create_result['payload']}")
                else:
                    new_job_id = create_result["payload"]["job_id"]
                    st.session_state.selected_job_id = new_job_id
                    st.session_state.portal_auto_fetch_job_id = new_job_id
                    st.session_state.portal_pending_nav = monitor_nav_label
                    st.session_state.portal_flash_message = (
                        f"Analysis queued for match {selected_match_id}. job_id={new_job_id}"
                    )
                    safe_rerun()

            terminal_jobs = [
                job
                for job in jobs
                if str(job.get("status", "")).lower() in {"completed", "failed", "canceled"}
            ]
            terminal_map = {
                f"{job['job_id']} | {job.get('status')} | {_iso_to_short(job.get('updated_at'))}": job["job_id"]
                for job in terminal_jobs
            }
            selected_delete_labels = quick_col2.multiselect(
                "Delete Old Runs",
                list(terminal_map.keys()),
                key=f"portal_match_delete_runs_{selected_match_id}",
            )
            if quick_col3.button(
                "Delete Selected Runs",
                key=f"portal_match_delete_runs_btn_{selected_match_id}",
                disabled=not selected_delete_labels,
            ):
                success_count = 0
                failed_count = 0
                for label in selected_delete_labels:
                    target_job_id = terminal_map.get(label, "")
                    if not target_job_id:
                        continue
                    delete_result = api_request(
                        "DELETE",
                        api_base,
                        f"/jobs/{target_job_id}",
                        tenant_id,
                        token,
                    )
                    if delete_result["ok"]:
                        success_count += 1
                    else:
                        failed_count += 1
                st.session_state.portal_flash_message = (
                    f"Deleted runs for match {selected_match_id}: success={success_count}, failed={failed_count}"
                )
                safe_rerun()
            if quick_col4.button(
                "Delete All Terminal",
                key=f"portal_match_delete_all_runs_btn_{selected_match_id}",
                disabled=not terminal_jobs,
            ):
                success_count = 0
                failed_count = 0
                for job in terminal_jobs:
                    target_job_id = str(job.get("job_id") or "").strip()
                    if not target_job_id:
                        continue
                    delete_result = api_request(
                        "DELETE",
                        api_base,
                        f"/jobs/{target_job_id}",
                        tenant_id,
                        token,
                    )
                    if delete_result["ok"]:
                        success_count += 1
                    else:
                        failed_count += 1
                st.session_state.portal_flash_message = (
                    f"Bulk delete for match {selected_match_id}: success={success_count}, failed={failed_count}"
                )
                safe_rerun()

            active_job = next(
                (
                    job
                    for job in jobs
                    if str(job.get("status", "")).lower() in {"queued", "claimed", "running", "cancel_requested"}
                ),
                None,
            )
            if active_job:
                st.subheader("Live Bookmark Feed")
                st.caption(f"Live updates for active run `{active_job.get('job_id')}`. This table grows as events are detected.")
                live_payload = list_job_bookmarks(
                    api_base=api_base,
                    tenant_id=tenant_id,
                    token=token,
                    job_id=str(active_job.get("job_id")),
                    limit=5000,
                )
                live_source = str(live_payload.get("source", "none"))
                live_items = list(live_payload.get("items", []))
                if live_source == "events":
                    live_rows = _build_bookmark_rows(active_job, live_items)
                else:
                    live_rows = _bookmark_rows_from_live_items(live_items, source=live_source)
                if live_rows:
                    st.dataframe(live_rows, use_container_width=True, hide_index=True)
                else:
                    st.info("No live bookmarks yet for the active run.")
            else:
                st.caption("No active run for this match right now.")

            mode_key = f"portal_match_reprocess_mode_{selected_match_id}"
            reprocess_mode = st.radio(
                "Reprocess Mode",
                ["Latest Config", "Custom Config"],
                horizontal=True,
                key=mode_key,
            )

            if reprocess_mode == "Latest Config":
                if not latest_job:
                    st.info("No prior jobs yet. Use Custom Config to start the first processing run for this uploaded match.")
                else:
                    st.caption(f"Using latest job config from `{latest_job.get('job_id')}`.")
                    if is_technical:
                        st.json(latest_config)
                    else:
                        summary_targets = _focus_targets_from_config(latest_config)
                        st.write(f"Model: `{latest_config.get('model_version', '-')}`")
                        st.write(f"Camera: `{latest_config.get('camera_mode', 'wide')}`")
                        st.write(f"Targets: `{summary_targets}`")
                        st.write(f"Analysis-only: `{bool(latest_config.get('analysis_only', False))}`")
                    if st.button("Reprocess This Match (Latest Config)", key=f"portal_match_reprocess_latest_{selected_match_id}"):
                        next_config = dict(latest_config)
                        next_config["run_created_at"] = datetime.utcnow().isoformat()
                        next_config["run_created_from"] = "portal_match_workspace_latest"
                        create_result = api_request(
                            "POST",
                            api_base,
                            f"/matches/{selected_match_id}/jobs",
                            tenant_id,
                            token,
                            json_body={"config": next_config},
                        )
                        if not create_result["ok"]:
                            st.error(f"Failed to queue reprocess job: {create_result['payload']}")
                        else:
                            new_job_id = create_result["payload"]["job_id"]
                            st.session_state.selected_job_id = new_job_id
                            st.session_state.portal_auto_fetch_job_id = new_job_id
                            st.session_state.portal_pending_nav = monitor_nav_label
                            st.session_state.portal_flash_message = (
                                f"Reprocess queued for match {selected_match_id}. job_id={new_job_id}"
                            )
                            safe_rerun()
            else:
                st.caption("Customize processing and run against this same uploaded source video.")
                custom_targets_default = list(latest_config.get("focus_event_types", []) or [])
                custom_model_default = str(latest_config.get("model_version") or (model_versions[0] if model_versions else "event-v0"))
                custom_profile_default = str(latest_config.get("profile_name") or "Custom")
                if custom_profile_default not in {"Balanced", "Offense Focus", "Set Piece Focus", "Discipline Review", "Custom"}:
                    custom_profile_default = "Custom"

                custom_col1, custom_col2, custom_col3 = st.columns(3)
                profile_name = custom_col1.selectbox(
                    "Profile",
                    ["Balanced", "Offense Focus", "Set Piece Focus", "Discipline Review", "Custom"],
                    index=["Balanced", "Offense Focus", "Set Piece Focus", "Discipline Review", "Custom"].index(custom_profile_default),
                    key=f"portal_match_custom_profile_{selected_match_id}",
                )
                model_version = custom_col2.selectbox(
                    "Model Version",
                    model_versions if model_versions else ["event-v0"],
                    index=(model_versions.index(custom_model_default) if custom_model_default in model_versions else 0),
                    key=f"portal_match_custom_model_{selected_match_id}",
                )
                analysis_only = custom_col3.checkbox(
                    "Analysis Only",
                    value=bool(latest_config.get("analysis_only", False)),
                    key=f"portal_match_custom_analysis_only_{selected_match_id}",
                )

                targets = st.multiselect(
                    "Event Targets",
                    EVENT_TARGET_OPTIONS,
                    default=[t for t in custom_targets_default if t in EVENT_TARGET_OPTIONS],
                    key=f"portal_match_custom_targets_{selected_match_id}",
                )
                custom_targets = st.text_input(
                    "Additional Custom Targets (comma separated)",
                    value="",
                    key=f"portal_match_custom_targets_extra_{selected_match_id}",
                )

                tune_col1, tune_col2, tune_col3 = st.columns(3)
                pre_seconds = tune_col1.slider(
                    "Pre Buffer",
                    0.5,
                    10.0,
                    float(latest_config.get("pre_seconds", 2.0)),
                    0.5,
                    key=f"portal_match_custom_pre_{selected_match_id}",
                )
                post_seconds = tune_col2.slider(
                    "Post Buffer",
                    1.0,
                    20.0,
                    float(latest_config.get("post_seconds", 6.0)),
                    0.5,
                    key=f"portal_match_custom_post_{selected_match_id}",
                )
                min_clip = tune_col3.slider(
                    "Min Clip",
                    1.0,
                    15.0,
                    float(latest_config.get("min_clip_duration", 4.0)),
                    0.5,
                    key=f"portal_match_custom_min_clip_{selected_match_id}",
                )

                tune_col4, tune_col5, tune_col6 = st.columns(3)
                speed_sens = tune_col4.slider(
                    "Speed Sensitivity",
                    1.0,
                    4.0,
                    float(latest_config.get("speed_sensitivity", 2.0)),
                    0.1,
                    key=f"portal_match_custom_speed_{selected_match_id}",
                )
                audio_sens = tune_col5.slider(
                    "Audio Sensitivity",
                    1.0,
                    4.0,
                    float(latest_config.get("audio_sensitivity", 2.0)),
                    0.1,
                    key=f"portal_match_custom_audio_{selected_match_id}",
                )
                threads = int(
                    tune_col6.number_input(
                        "Threads (0=auto)",
                        min_value=0,
                        max_value=32,
                        value=int(latest_config.get("threads", 0) or 0),
                        key=f"portal_match_custom_threads_{selected_match_id}",
                    )
                )

                cam_col1, cam_col2, cam_col3 = st.columns(3)
                camera_mode_options = ["wide", "follow_action", "follow_player"]
                current_camera_mode = str(latest_config.get("camera_mode") or "wide")
                if current_camera_mode not in camera_mode_options:
                    current_camera_mode = "wide"
                camera_mode = cam_col1.selectbox(
                    "Camera Mode",
                    camera_mode_options,
                    index=camera_mode_options.index(current_camera_mode),
                    key=f"portal_match_custom_camera_mode_{selected_match_id}",
                )
                zoom_factor = cam_col2.slider(
                    "Zoom Factor",
                    1.0,
                    2.5,
                    float(latest_config.get("zoom_factor", 1.6)),
                    0.1,
                    key=f"portal_match_custom_zoom_factor_{selected_match_id}",
                )
                cam_col3.caption("Follow-cam crops around the tracked player and recenters nearby ball action.")

                opt_col1, opt_col2, opt_col3 = st.columns(3)
                no_audio = opt_col1.checkbox(
                    "Disable Audio Detection",
                    value=bool(latest_config.get("no_audio", False)),
                    key=f"portal_match_custom_no_audio_{selected_match_id}",
                )
                overlay = opt_col2.checkbox(
                    "Generate Overlay",
                    value=bool(latest_config.get("overlay", False)),
                    key=f"portal_match_custom_overlay_{selected_match_id}",
                )
                require_gpu = opt_col3.checkbox(
                    "Require GPU",
                    value=bool(latest_config.get("require_gpu", False)),
                    key=f"portal_match_custom_require_gpu_{selected_match_id}",
                )

                player_roi_payload = _render_player_roi_selector(
                    context_key=f"portal_match_custom_{selected_match_id}",
                    default_enabled=bool(latest_config.get("player_roi")),
                    existing_roi=(latest_config.get("player_roi") if isinstance(latest_config.get("player_roi"), dict) else None),
                    source_video_path=source_video or "",
                )

                if st.button("Reprocess This Match (Custom)", key=f"portal_match_reprocess_custom_{selected_match_id}"):
                    if player_roi_payload.get("enabled") and not player_roi_payload.get("roi"):
                        st.error("Player ROI lock is enabled, but the first-frame selector is not ready. Uncheck it or choose a readable local source video.")
                    else:
                        target_list = [str(item).strip().lower() for item in targets if str(item).strip()]
                        if custom_targets.strip():
                            target_list.extend([item.strip().lower() for item in custom_targets.split(",") if item.strip()])
                        target_list = list(dict.fromkeys(target_list))

                        profile_defaults = _profile_defaults(profile_name, target_list)
                        if profile_name != "Custom":
                            pre_seconds = float(profile_defaults.get("pre_seconds", pre_seconds))
                            post_seconds = float(profile_defaults.get("post_seconds", post_seconds))
                            min_clip = float(profile_defaults.get("min_clip_duration", min_clip))
                            speed_sens = float(profile_defaults.get("speed_sensitivity", speed_sens))
                            audio_sens = float(profile_defaults.get("audio_sensitivity", audio_sens))
                            target_list = list(dict.fromkeys(target_list + list(profile_defaults.get("focus_event_types", []))))

                        next_config: Dict[str, Any] = {
                            "profile_name": profile_name,
                            "model_version": model_version,
                            "focus_event_types": target_list,
                            "pre_seconds": float(pre_seconds),
                            "post_seconds": float(post_seconds),
                            "min_clip_duration": float(min_clip),
                            "speed_sensitivity": float(speed_sens),
                            "audio_sensitivity": float(audio_sens),
                            "camera_mode": str(camera_mode),
                            "zoom_factor": float(zoom_factor),
                            "overlay": bool(overlay),
                            "no_audio": bool(no_audio),
                            "require_gpu": bool(require_gpu),
                            "analysis_only": bool(analysis_only),
                            "run_created_at": datetime.utcnow().isoformat(),
                            "run_created_from": "portal_match_workspace_custom",
                        }
                        if player_roi_payload.get("roi"):
                            next_config["player_roi"] = dict(player_roi_payload["roi"])
                        if threads > 0:
                            next_config["threads"] = int(threads)

                        create_result = api_request(
                            "POST",
                            api_base,
                            f"/matches/{selected_match_id}/jobs",
                            tenant_id,
                            token,
                            json_body={"config": next_config},
                        )
                        if not create_result["ok"]:
                            st.error(f"Failed to queue reprocess job: {create_result['payload']}")
                        else:
                            new_job_id = create_result["payload"]["job_id"]
                            st.session_state.selected_job_id = new_job_id
                            st.session_state.portal_auto_fetch_job_id = new_job_id
                            st.session_state.portal_pending_nav = monitor_nav_label
                            st.session_state.portal_flash_message = (
                                f"Custom reprocess queued for match {selected_match_id}. job_id={new_job_id}"
                            )
                            safe_rerun()

        st.subheader("Processing Transactions")
        if not jobs:
            st.info("No jobs for this game yet.")
        else:
            for idx, job in enumerate(jobs):
                exp_title = f"{job['job_id']} | {job.get('status')} | {job.get('stage')} | {_iso_to_short(job.get('created_at'))}"
                with st.expander(exp_title, expanded=(idx == 0)):
                    st.markdown(_build_stage_tracker(job.get("stage"), job.get("status")), unsafe_allow_html=True)
                    st.progress(float(job.get("progress", 0.0)))
                    st.write(f"Model: `{job.get('config', {}).get('model_version', '-')}`")
                    st.write(f"Camera: `{job.get('config', {}).get('camera_mode', 'wide')}`")
                    st.write(f"Targets: `{_focus_targets_from_config(job.get('config', {}))}`")
                    st.write(f"Analysis-only: `{bool(job.get('config', {}).get('analysis_only', False))}`")
                    st.write(f"Bookmarks detected: `{int((job.get('result', {}) or {}).get('bookmarks_count', 0) or 0)}`")
                    st.write(f"Updated: `{_iso_to_short(job.get('updated_at'))}`")

                    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
                    if action_col1.button("Set Active in Console", key=f"portal_set_active_{job['job_id']}"):
                        st.session_state.selected_job_id = job["job_id"]
                        st.success(f"Active job set: {job['job_id']}")
                    if action_col2.button("Rerun From This Config", key=f"portal_rerun_from_library_{job['job_id']}"):
                        rerun_result = api_request(
                            "POST",
                            api_base,
                            f"/jobs/{job['job_id']}/rerun",
                            tenant_id,
                            token,
                            json_body={"config_overrides": {}, "reason": "library-rerun"},
                        )
                        if rerun_result["ok"]:
                            st.success(f"Rerun queued: {rerun_result['payload']['job_id']}")
                        else:
                            st.error(f"Rerun failed: {rerun_result['payload']}")
                    if action_col3.button("Kill Session", key=f"portal_kill_from_library_{job['job_id']}"):
                        kill_result = api_request(
                            "POST",
                            api_base,
                            f"/jobs/{job['job_id']}/kill-session",
                            tenant_id,
                            token,
                            json_body={},
                        )
                        if kill_result["ok"]:
                            st.warning(f"Kill requested for {job['job_id']}")
                        else:
                            st.error(f"Kill request failed: {kill_result['payload']}")
                    can_delete_run = str(job.get("status", "")).lower() not in {"running", "claimed"}
                    if action_col4.button(
                        "Delete Run",
                        key=f"portal_delete_run_{job['job_id']}",
                        disabled=not can_delete_run,
                    ):
                        delete_result = api_request(
                            "DELETE",
                            api_base,
                            f"/jobs/{job['job_id']}",
                            tenant_id,
                            token,
                        )
                        if delete_result["ok"]:
                            st.session_state.portal_flash_message = f"Run deleted: {job['job_id']}"
                            safe_rerun()
                        else:
                            st.error(f"Delete failed: {delete_result['payload']}")

                    if is_technical:
                        st.write("Configuration")
                        st.json(job.get("config", {}))
                        st.write("Result")
                        st.json(job.get("result", {}))

        _render_match_bookmark_review(
            context_key="library_review",
            api_base=api_base,
            tenant_id=tenant_id,
            token=token,
            selected_match=selected_match,
            selected_match_id=selected_match_id,
            jobs=jobs,
            is_technical=is_technical,
            mode_code="technical" if is_technical else "user",
            view_code="library",
        )
else:
    if is_technical:
        st.subheader("Operations Console")
        st.caption("Inspect transaction state, fetch deep logs, run worker ticks, and cancel/kill sessions.")
    else:
        st.subheader("Run Monitor")
        st.caption("Track run status and watch bookmarks populate as analysis progresses.")

    auto_target_job = str(st.session_state.portal_auto_fetch_job_id or "").strip()
    default_job = auto_target_job or str(st.session_state.selected_job_id or "").strip()
    if auto_target_job and not str(st.session_state.get("portal_ops_job_id", "")).strip():
        st.session_state.portal_ops_job_id = auto_target_job
    job_id_input = st.text_input("Job ID", value=default_job, key="portal_ops_job_id")
    ops_job_id_for_url = str(job_id_input or "").strip()
    auto_refresh = st.checkbox(
        "Auto-refresh selected job while active",
        value=True,
        key="portal_ops_auto_refresh",
    )

    op_col1, op_col2, op_col3, op_col4 = st.columns(4)
    fetch_job = op_col1.button("Fetch Job", key="portal_ops_fetch_job")
    cancel_job = op_col2.button("Cancel Job", key="portal_ops_cancel_job")
    kill_job = op_col3.button("Kill Session", key="portal_ops_kill_job")
    worker_tick = op_col4.button("Worker Run Once", key="portal_ops_worker_tick")

    st.subheader("Queue Controls")
    default_scope_match = str(st.session_state.selected_match_id or "").strip()
    if default_scope_match and not str(st.session_state.get("portal_ops_match_scope", "")).strip():
        st.session_state.portal_ops_match_scope = default_scope_match
    match_scope = st.text_input("Match ID Scope (bulk queue actions)", value=default_scope_match, key="portal_ops_match_scope")
    if match_scope.strip():
        scoped_jobs = list_match_jobs(api_base, tenant_id, token, match_scope.strip(), limit=500)
        queued_jobs = [job for job in scoped_jobs if str(job.get("status", "")).lower() == "queued"]
        active_jobs = [
            job
            for job in scoped_jobs
            if str(job.get("status", "")).lower() in {"queued", "claimed", "running", "cancel_requested"}
        ]
        q_col1, q_col2, q_col3 = st.columns(3)
        q_col1.metric("Scoped Jobs", len(scoped_jobs))
        q_col2.metric("Queued Jobs", len(queued_jobs))
        q_col3.metric("Active Jobs", len(active_jobs))

        queue_col1, queue_col2 = st.columns(2)
        kill_all_queued = queue_col1.button(
            "Kill All Queued (Match Scope)",
            key="portal_ops_kill_all_queued",
            disabled=not queued_jobs,
        )
        kill_all_active = queue_col2.button(
            "Kill All Active (Match Scope)",
            key="portal_ops_kill_all_active",
            disabled=not active_jobs,
        )
        if kill_all_queued or kill_all_active:
            targets = queued_jobs if kill_all_queued else active_jobs
            successes = 0
            failures = 0
            for target in targets:
                target_job_id = str(target.get("job_id", "")).strip()
                if not target_job_id:
                    continue
                kill_result = api_request(
                    "POST",
                    api_base,
                    f"/jobs/{target_job_id}/kill-session",
                    tenant_id,
                    token,
                    json_body={},
                )
                if kill_result["ok"]:
                    successes += 1
                else:
                    failures += 1
            st.session_state.portal_flash_message = (
                f"Bulk kill completed for match {match_scope.strip()}: success={successes}, failed={failures}"
            )
            safe_rerun()
    else:
        st.caption("Provide Match ID scope to bulk-kill queued jobs.")

    if worker_tick:
        tick_result = api_request("POST", api_base, "/jobs/worker/run-once", tenant_id, token, json_body={})
        if is_technical:
            st.json(tick_result)
        else:
            if tick_result.get("ok"):
                st.info(f"Worker tick complete. worked={tick_result.get('payload', {}).get('worked')}")
            else:
                st.error(f"Worker tick failed: {tick_result.get('payload')}")

    job_id_value = job_id_input.strip()
    should_fetch = False
    if job_id_value:
        if fetch_job:
            should_fetch = True
        elif auto_target_job and auto_target_job == job_id_value:
            should_fetch = True
            st.session_state.portal_auto_fetch_job_id = ""
        elif auto_refresh:
            should_fetch = True

    if should_fetch and job_id_value:
        job_result = api_request("GET", api_base, f"/jobs/{job_id_value}", tenant_id, token)
        if not job_result["ok"]:
            st.error(job_result["payload"])
        else:
            payload = job_result["payload"]
            st.session_state.selected_job_id = payload["job_id"]
            st.markdown(_build_stage_tracker(payload.get("stage"), payload.get("status")), unsafe_allow_html=True)
            st.progress(float(payload.get("progress", 0.0)))
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            metrics_col1.metric("Status", payload.get("status", "-"))
            metrics_col2.metric("Stage", payload.get("stage", "-"))
            metrics_col3.metric("Cancel Requested", str(payload.get("cancel_requested", False)))
            if is_technical:
                st.write("Config")
                st.json(payload.get("config", {}))
                st.write("Result")
                st.json(payload.get("result", {}))
            else:
                st.write(f"Model: `{payload.get('config', {}).get('model_version', '-')}`")
                st.write(f"Camera: `{payload.get('config', {}).get('camera_mode', 'wide')}`")
                st.write(f"Targets: `{_focus_targets_from_config(payload.get('config', {}))}`")
                st.write(f"Bookmarks: `{int((payload.get('result', {}) or {}).get('bookmarks_count', 0) or 0)}`")
            status_value = str(payload.get("status", "")).lower()
            if auto_refresh and status_value in {"queued", "claimed", "running", "cancel_requested"}:
                time.sleep(1.5)
                safe_rerun()

    if job_id_value and cancel_job:
        result = api_request("POST", api_base, f"/jobs/{job_id_value}/cancel", tenant_id, token, json_body={})
        if is_technical:
            st.json(result)
        else:
            st.info(f"Cancel requested for {job_id_value}.")

    if job_id_value and kill_job:
        result = api_request(
            "POST",
            api_base,
            f"/jobs/{job_id_value}/kill-session",
            tenant_id,
            token,
            json_body={},
        )
        if is_technical:
            st.json(result)
        else:
            st.warning(f"Kill requested for {job_id_value}.")

    if job_id_value:
        st.subheader("Live Bookmark Table")
        live_payload = list_job_bookmarks(api_base, tenant_id, token, job_id_value, limit=5000)
        live_items = list(live_payload.get("items", []))
        live_rows: List[Dict[str, Any]] = []
        for item in live_items:
            if not isinstance(item, dict):
                continue
            if "occurred_at_ms" in item:
                occurred_s = float(item.get("occurred_at_ms", 0)) / 1000.0
                start_s = float(item.get("start_ms", 0)) / 1000.0
                end_s = float(item.get("end_ms", 0)) / 1000.0
                live_rows.append(
                    {
                        "id": item.get("event_id"),
                        "time": _seconds_to_clock(occurred_s),
                        "event_type": item.get("event_type"),
                        "confidence": round(float(item.get("confidence", 0.0) or 0.0), 3),
                        "window": f"{_seconds_to_clock(start_s)} - {_seconds_to_clock(end_s)}",
                        "status": item.get("status"),
                        "source": "events",
                    }
                )
            else:
                occurred_s = float(item.get("occurred_at_s", 0.0) or 0.0)
                start_s = float(item.get("start_s", occurred_s) or occurred_s)
                end_s = float(item.get("end_s", occurred_s) or occurred_s)
                live_rows.append(
                    {
                        "id": item.get("bookmark_id"),
                        "time": _seconds_to_clock(occurred_s),
                        "event_type": item.get("event_type", "candidate"),
                        "confidence": round(float(item.get("confidence", 0.0) or 0.0), 3),
                        "window": f"{_seconds_to_clock(start_s)} - {_seconds_to_clock(end_s)}",
                        "status": item.get("status", "bookmark_only"),
                        "source": str(live_payload.get("source", "manifest")),
                    }
                )
        st.caption(f"Bookmark source: `{live_payload.get('source', 'none')}`")
        if live_rows:
            st.dataframe(live_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No bookmarks yet for this run. Keep this screen open; table will update as analysis writes detections.")

    if is_technical:
        st.subheader("Log Inspector")
        log_col1, log_col2, log_col3, log_col4 = st.columns(4)
        with log_col1:
            level_filter = st.selectbox(
                "Level",
                ["all", "debug", "info", "warning", "error"],
                index=0,
                key="portal_ops_log_level",
            )
        with log_col2:
            detail_filter = st.selectbox(
                "Detail",
                ["all", "basic", "detailed", "extreme"],
                index=0,
                key="portal_ops_log_detail",
            )
        with log_col3:
            stage_filter = st.text_input("Stage Filter", value="", key="portal_ops_log_stage")
        with log_col4:
            log_limit = st.number_input("Limit", min_value=10, max_value=5000, value=300, key="portal_ops_log_limit")

        if st.button("Fetch Logs", key="portal_ops_fetch_logs"):
            if not job_id_input.strip():
                st.error("Provide a job ID first.")
            else:
                query = [f"limit={int(log_limit)}"]
                if level_filter != "all":
                    query.append(f"level={level_filter}")
                if detail_filter != "all":
                    query.append(f"detail_level={detail_filter}")
                if stage_filter.strip():
                    query.append(f"stage={stage_filter.strip()}")
                suffix = "&".join(query)
                log_result = api_request(
                    "GET",
                    api_base,
                    f"/jobs/{job_id_input.strip()}/logs?{suffix}",
                    tenant_id,
                    token,
                )
                if not log_result["ok"]:
                    st.error(log_result["payload"])
                else:
                    items = list(log_result["payload"].get("items", []))
                    st.write(f"Log rows: {len(items)}")
                    st.dataframe(items, use_container_width=True, hide_index=True)

current_mode_code = "technical" if is_technical else "user"
current_view_code = _view_code_from_nav(nav_key=nav_key, is_technical=is_technical)
selected_match_id_for_url = str(st.session_state.get("selected_match_id", "")).strip()
selected_match_label_for_url = str(st.session_state.get("selected_match_label", "")).strip()
selected_job_id_for_url = str(ops_job_id_for_url or st.session_state.get("selected_job_id", "")).strip()
seek_for_url = int(st.session_state.get("portal_video_seek_s", 0) or 0)
bookmark_for_url = str(st.session_state.get("portal_selected_bookmark_ref", "")).strip()
if current_view_code not in {"studio", "library"} or not selected_job_id_for_url:
    bookmark_for_url = ""

portal_query_params = _build_portal_query_params(
    mode_code=current_mode_code,
    view_code=current_view_code,
    tenant_id=tenant_id,
    match_id=selected_match_id_for_url,
    match_label=selected_match_label_for_url,
    job_id=selected_job_id_for_url,
    seek_s=seek_for_url,
    bookmark_ref=bookmark_for_url,
)
_write_query_params(portal_query_params)
