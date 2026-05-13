
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

LOG_PROFILE_OPTIONS = ["Standard", "Detailed", "Diagnostic"]
YOLO_MODEL_OPTIONS = [
    "yolo26s.pt",
    "yolo26n.pt",
    "yolo26m.pt",
    "yolo26l.pt",
    "yolo26x.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolov8s.pt",
    "yolov8n.pt",
    "Custom .pt path",
]
TRACKER_CONFIG_OPTIONS = ["botsort.yaml", "bytetrack.yaml"]
INFERENCE_IMAGE_SIZE_OPTIONS = [640, 768, 960, 1280]
VID_STRIDE_OPTIONS = [1, 2, 3, 4]

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
  --bg: #f8f8f8;
  --panel: #ffffff;
  --panel-soft: #f2f2f2;
  --ink: #0f0f0f;
  --muted: #606060;
  --brand: #ff0033;
  --brand-2: #065fd4;
  --warn: #a86500;
  --danger: #c00;
  --line: #e5e5e5;
  --line-strong: #d3d3d3;
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
  background: #ffffff;
  border-right: 1px solid #e5e5e5;
}

[data-testid="stSidebar"] * {
  color: #0f0f0f;
}

[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
  color: var(--ink) !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] * {
  color: var(--ink) !important;
}

[data-testid="stSidebar"] [role="radiogroup"] * {
  color: #0f0f0f !important;
}

[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
  color: #606060 !important;
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
  padding: .2rem 0 .75rem 0;
  margin-bottom: .35rem;
  border-bottom: 1px solid var(--line);
  background: #fff;
}

.portal-header h1 {
  margin: 0;
  font-size: 1.55rem;
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

.run-monitor-shell {
  border: 1px solid var(--line);
  border-radius: 14px;
  background:
    linear-gradient(135deg, rgba(255, 0, 51, .055), transparent 36%),
    linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
  padding: 1rem;
  margin: .55rem 0 1rem 0;
  box-shadow: 0 8px 22px rgba(15, 15, 15, .055);
}

.run-monitor-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: .9rem;
}

.run-monitor-kicker {
  color: var(--muted);
  font-size: .78rem;
  font-weight: 800;
  letter-spacing: .08em;
  text-transform: uppercase;
}

.run-monitor-title {
  margin: .1rem 0 0 0;
  font-family: "Space Grotesk", sans-serif;
  font-size: 1.2rem;
  font-weight: 800;
}

.run-live-badge {
  display: inline-flex;
  align-items: center;
  gap: .38rem;
  border: 1px solid #f1b8c3;
  border-radius: 999px;
  background: #fff5f7;
  color: #a10024;
  padding: .28rem .62rem;
  font-size: .78rem;
  font-weight: 800;
  white-space: nowrap;
}

.run-live-dot {
  width: .48rem;
  height: .48rem;
  border-radius: 999px;
  background: var(--brand);
  box-shadow: 0 0 0 5px rgba(255, 0, 51, .12);
}

.run-stage-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(132px, 1fr));
  gap: .55rem;
  margin: .8rem 0 1rem 0;
}

.run-stage-card {
  position: relative;
  min-height: 104px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #fff;
  padding: .72rem .72rem .68rem .72rem;
  overflow: hidden;
}

.run-stage-card::before {
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  width: 100%;
  height: 4px;
  background: #d9d9d9;
}

.run-stage-card.done::before {
  background: #178a4d;
}

.run-stage-card.active {
  border-color: #111;
  box-shadow: 0 8px 18px rgba(15, 15, 15, .08);
}

.run-stage-card.active::before {
  background: var(--brand);
}

.run-stage-card.fail::before {
  background: #c00;
}

.run-stage-label {
  margin: 0;
  font-family: "Space Grotesk", sans-serif;
  font-size: .94rem;
  font-weight: 800;
  color: var(--ink);
}

.run-stage-state {
  margin: .2rem 0 .42rem 0;
  color: var(--muted);
  font-size: .78rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .04em;
}

.run-stage-card.done .run-stage-state {
  color: #178a4d;
}

.run-stage-card.active .run-stage-state {
  color: #a10024;
}

.run-stage-desc {
  margin: 0;
  color: var(--muted);
  font-size: .82rem;
  line-height: 1.25;
}

.run-meter-row {
  display: grid;
  grid-template-columns: minmax(150px, 1fr) minmax(180px, 1.8fr) auto;
  align-items: center;
  gap: .75rem;
  padding: .55rem .68rem;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #fff;
  margin-top: .48rem;
}

.run-meter-label {
  color: var(--muted);
  font-size: .82rem;
  font-weight: 800;
}

.run-meter-track {
  height: .52rem;
  border-radius: 999px;
  background: #ececec;
  overflow: hidden;
}

.run-meter-fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--brand), #111);
}

.run-meter-value {
  color: var(--ink);
  font-size: .86rem;
  font-weight: 800;
  text-align: right;
}

.run-signal-strip {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(178px, 1fr));
  gap: .5rem;
  margin: .75rem 0 0 0;
}

.run-signal-chip {
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #fff;
  padding: .55rem .65rem;
}

.run-signal-chip b {
  display: block;
  color: var(--ink);
  font-size: .93rem;
}

.run-signal-chip span {
  color: var(--muted);
  font-size: .78rem;
}

.run-log-strip {
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #111;
  color: #f5f5f5;
  padding: .65rem .75rem;
  margin-top: .75rem;
  font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
  font-size: .78rem;
  line-height: 1.35;
  max-height: 190px;
  overflow: auto;
}

.run-log-strip div {
  white-space: nowrap;
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

.share-link-row {
  display: flex;
  flex-wrap: wrap;
  gap: .5rem;
  margin: .2rem 0 .55rem 0;
}

.share-link-row a {
  display: inline-flex;
  align-items: center;
  min-height: 2rem;
  padding: .32rem .65rem;
  border: 1px solid var(--line-strong);
  border-radius: 6px;
  background: var(--panel);
  color: var(--brand);
  font-weight: 600;
  text-decoration: none;
}

.share-link-row a:hover {
  border-color: var(--brand);
  background: #fff0f3;
  text-decoration: none;
}

.yt-studio-topbar {
  display: flex;
  align-items: center;
  gap: .8rem;
  padding: .55rem .2rem .8rem .2rem;
  border-bottom: 1px solid var(--line);
  margin-bottom: .85rem;
}

.yt-brand-mark {
  display: inline-flex;
  align-items: center;
  gap: .45rem;
  font-weight: 800;
  font-family: "Space Grotesk", sans-serif;
  font-size: 1.1rem;
}

.yt-brand-play {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 2rem;
  height: 1.35rem;
  border-radius: .42rem;
  background: var(--brand);
  color: #fff;
  font-size: .76rem;
}

.yt-search-pill {
  flex: 1;
  max-width: 560px;
  min-height: 2.35rem;
  display: inline-flex;
  align-items: center;
  padding: 0 .95rem;
  border-radius: 999px;
  background: #f1f1f1;
  color: var(--muted);
}

.yt-editor-shell {
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  background: #fff;
  margin: .35rem 0 1rem 0;
}

.yt-panel-title {
  margin: 0 0 .3rem 0;
  font-size: 1.04rem;
  font-weight: 800;
}

.yt-section-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 3.35rem;
  border-bottom: 1px solid var(--line);
  color: var(--ink);
}

.yt-section-row span {
  color: var(--muted);
  font-size: .9rem;
}

.yt-plus {
  font-size: 1.45rem;
  color: var(--ink);
}

.yt-timeline {
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  background: #fff;
  padding: .7rem 0;
  margin-top: .8rem;
}

.yt-time-ruler {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: .5rem;
  color: var(--muted);
  font-size: .8rem;
  padding: 0 .65rem .45rem .65rem;
}

.yt-waveform {
  height: 2.15rem;
  margin: 0 .65rem .4rem .65rem;
  border-left: 1px solid #cfcfcf;
  border-right: 1px solid #cfcfcf;
  background:
    linear-gradient(90deg, rgba(0,0,0,.04) 1px, transparent 1px) 0 0 / 48px 100%,
    repeating-linear-gradient(90deg, #d6d6d6 0 5px, #efefef 5px 10px);
  opacity: .95;
}

.yt-thumbstrip {
  height: 2.55rem;
  margin: 0 .65rem;
  border: 1px solid #b7d3f5;
  background:
    repeating-linear-gradient(90deg, #89a86f 0 56px, #c6bd83 56px 112px, #7fa2c4 112px 168px);
}

.yt-waveform-real {
  display: flex;
  align-items: center;
  gap: 1px;
  height: 2.35rem;
  margin: 0 .65rem .42rem .65rem;
  padding: 0 .35rem;
  border-left: 1px solid #cfcfcf;
  border-right: 1px solid #cfcfcf;
  background: #fafafa;
  overflow: hidden;
}

.yt-wavebar {
  flex: 1 1 0;
  min-width: 2px;
  background: #c7c7c7;
  border-radius: 2px;
}

.yt-thumbstrip-real {
  display: flex;
  align-items: stretch;
  gap: 2px;
  min-height: 3.25rem;
  margin: 0 .65rem;
  border: 1px solid #b7d3f5;
  background: #f7fbff;
  overflow-x: auto;
}

.yt-thumbstrip-real img {
  display: block;
  height: 3.25rem;
  width: auto;
  object-fit: cover;
}

.audio-edit-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: .75rem .85rem;
  background: #fff;
  margin-bottom: .55rem;
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
    data: Optional[Dict[str, Any]] = None,
    files: Optional[Dict[str, Any]] = None,
    timeout: int = 180,
) -> Dict[str, Any]:
    headers: Dict[str, str] = {"X-Tenant-Id": tenant_id}
    if token.strip():
        headers["Authorization"] = f"Bearer {token.strip()}"
    url = f"{api_base.rstrip('/')}{path}"
    try:
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            json=json_body,
            data=data,
            files=files,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return {
            "ok": False,
            "status_code": 0,
            "payload": {
                "error": "connection_error",
                "message": str(exc),
                "url": url,
                "hint": (
                    "Use http://127.0.0.1:8000/v1 when running the portal locally. "
                    "The http://api:8000/v1 hostname only works inside Docker Compose."
                ),
            },
        }
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


def get_match_timeline(
    api_base: str,
    tenant_id: str,
    token: str,
    match_id: str,
    thumbnail_count: int = 18,
    waveform_bins: int = 120,
) -> Dict[str, Any]:
    result = api_request(
        "GET",
        api_base,
        f"/matches/{match_id}/timeline?thumbnail_count={thumbnail_count}&waveform_bins={waveform_bins}",
        tenant_id,
        token,
        timeout=240,
    )
    if not result["ok"]:
        return {"ok": False, "error": result.get("payload"), "thumbnails": [], "waveform": {"peaks": []}, "video": {}}
    payload = result["payload"] or {}
    if isinstance(payload, dict):
        payload["ok"] = True
        return payload
    return {"ok": False, "error": str(payload), "thumbnails": [], "waveform": {"peaks": []}, "video": {}}


def list_training_models(api_base: str, tenant_id: str, token: str) -> List[Dict[str, Any]]:
    result = api_request("GET", api_base, "/training/models", tenant_id, token)
    if not result["ok"]:
        return []
    payload = result["payload"] or []
    return list(payload)


def create_training_run(api_base: str, tenant_id: str, token: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return api_request("POST", api_base, "/training/runs", tenant_id, token, json_body=payload, timeout=60)


def get_training_run(api_base: str, tenant_id: str, token: str, run_id: str) -> Dict[str, Any]:
    result = api_request("GET", api_base, f"/training/runs/{run_id}", tenant_id, token, timeout=30)
    return result["payload"] if result["ok"] and isinstance(result["payload"], dict) else {}


def get_gpu_health(api_base: str, tenant_id: str, token: str) -> Dict[str, Any]:
    result = api_request("GET", api_base, "/health/gpu", tenant_id, token, timeout=15)
    if not result["ok"]:
        return {"ready": False, "error": result.get("payload"), "torch": {}, "nvidia_smi": {}, "ffmpeg_nvenc": {}}
    payload = result["payload"] or {}
    return dict(payload) if isinstance(payload, dict) else {"ready": False, "error": payload}


def get_agent_status(api_base: str, tenant_id: str, token: str) -> Dict[str, Any]:
    result = api_request("GET", api_base, "/agent/status", tenant_id, token, timeout=15)
    if not result["ok"]:
        return {"configured": False, "reachable": False, "provider": "unknown", "message": str(result.get("payload"))}
    payload = result["payload"] or {}
    return dict(payload) if isinstance(payload, dict) else {"configured": False, "reachable": False, "message": str(payload)}


def inspect_local_video(api_base: str, tenant_id: str, token: str, path: str) -> Dict[str, Any]:
    result = api_request(
        "POST",
        api_base,
        "/matches/assets/inspect-local",
        tenant_id,
        token,
        json_body={"path": path, "set_as_source": False},
        timeout=30,
    )
    if not result["ok"]:
        payload = result.get("payload") or {}
        if isinstance(payload, dict):
            message = (
                ((payload.get("error") or {}) if isinstance(payload.get("error"), dict) else {}).get("message")
                or payload.get("detail")
                or payload.get("message")
                or str(payload)
            )
        else:
            message = str(payload)
        fallback = _local_video_probe(path)
        fallback.update({"ok": False, "code": "api_inspect_failed", "message": f"API could not inspect local path: {message}"})
        return fallback
    payload = result["payload"] or {}
    return dict(payload) if isinstance(payload, dict) else {"ok": False, "code": "bad_response", "message": str(payload)}


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


def get_job_diagnostics(api_base: str, tenant_id: str, token: str, job_id: str) -> Dict[str, Any]:
    result = api_request("GET", api_base, f"/jobs/{job_id}/diagnostics", tenant_id, token, timeout=30)
    if not result["ok"]:
        return {"ok": False, "summary": "Diagnostics unavailable.", "next_action": str(result.get("payload"))}
    payload = result["payload"] or {}
    return dict(payload) if isinstance(payload, dict) else {"ok": False, "summary": str(payload), "next_action": ""}


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


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        text = value.strip()
        if ":" in text:
            try:
                parts = [float(part) for part in text.split(":")]
            except ValueError:
                return None
            if len(parts) == 3:
                return (parts[0] * 3600.0) + (parts[1] * 60.0) + parts[2]
            if len(parts) == 2:
                return (parts[0] * 60.0) + parts[1]
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _format_trim_window(config: Dict[str, Any]) -> str:
    trim_start = _float_or_none((config or {}).get("trim_start"))
    trim_end = _float_or_none((config or {}).get("trim_end"))
    if trim_start is None and trim_end is None:
        return "full game"
    start = trim_start or 0.0
    if trim_end is None:
        return f"{_seconds_to_clock(start)} - end"
    return f"{_seconds_to_clock(start)} - {_seconds_to_clock(trim_end)}"


def _render_trim_window_controls(
    *,
    context_key: str,
    config: Optional[Dict[str, Any]] = None,
    default_enabled: bool = False,
    default_duration_minutes: float = 2.0,
) -> Dict[str, Any]:
    source_config = config or {}
    existing_start = _float_or_none(source_config.get("trim_start"))
    existing_end = _float_or_none(source_config.get("trim_end"))
    has_existing_window = existing_start is not None or existing_end is not None
    start_default = max(0.0, (existing_start or 0.0) / 60.0)
    if existing_end is not None and existing_end > (existing_start or 0.0):
        duration_default = max(0.25, (existing_end - (existing_start or 0.0)) / 60.0)
    else:
        duration_default = default_duration_minutes

    enabled = st.checkbox(
        "Limit to test window",
        value=has_existing_window or default_enabled,
        key=f"{context_key}_trim_enabled",
        help="Process only a short section first. Leave this on for smoke tests before running an entire full-match file.",
    )
    trim_col1, trim_col2, trim_col3 = st.columns([1.0, 1.0, 1.2])
    start_minutes = trim_col1.number_input(
        "Start Minute",
        min_value=0.0,
        max_value=720.0,
        value=round(float(start_default), 2),
        step=0.5,
        disabled=not enabled,
        key=f"{context_key}_trim_start_minutes",
    )
    duration_minutes = trim_col2.number_input(
        "Duration Minutes",
        min_value=0.25,
        max_value=240.0,
        value=round(float(duration_default), 2),
        step=0.25,
        disabled=not enabled,
        key=f"{context_key}_trim_duration_minutes",
    )
    trim_start = round(float(start_minutes) * 60.0, 3)
    trim_end = round(trim_start + (float(duration_minutes) * 60.0), 3)
    label = f"{_seconds_to_clock(trim_start)} - {_seconds_to_clock(trim_end)}" if enabled else "full game"
    trim_col3.metric("Processing Window", label)
    if enabled:
        st.caption("Smoke-test mode is active. This run will only analyze the selected slice of the video.")

    return {
        "enabled": bool(enabled),
        "trim_start": trim_start if enabled else None,
        "trim_end": trim_end if enabled else None,
        "label": label,
    }


def _resolve_match_video_source(match: Dict[str, Any]) -> str:
    source_path = str(match.get("source_video_path") or "").strip()
    if source_path:
        return source_path
    metadata = match.get("metadata") or {}
    assets = list(metadata.get("assets", []) or [])
    if assets:
        return str(assets[0].get("path") or "").strip()
    return ""


def _render_timeline_media(timeline: Dict[str, Any]) -> None:
    video = timeline.get("video", {}) if isinstance(timeline.get("video"), dict) else {}
    duration = float(video.get("duration_seconds") or 0.0)
    ruler_points = [0.0]
    if duration > 0:
        ruler_points = [duration * (idx / 4.0) for idx in range(5)]
    ruler_html = "".join(f"<span>{_seconds_to_clock(point)}</span>" for point in ruler_points)

    waveform = timeline.get("waveform", {}) if isinstance(timeline.get("waveform"), dict) else {}
    peaks = [float(item or 0.0) for item in list(waveform.get("peaks", []) or [])[:240]]
    if peaks:
        bars = "".join(
            f'<div class="yt-wavebar" style="height:{max(8.0, min(100.0, peak * 100.0)):.1f}%"></div>'
            for peak in peaks
        )
        waveform_html = f'<div class="yt-waveform-real">{bars}</div>'
    else:
        waveform_html = '<div class="yt-waveform"></div>'

    thumbs = [
        item
        for item in list(timeline.get("thumbnails", []) or [])
        if isinstance(item, dict) and str(item.get("data_url") or "").startswith("data:image/")
    ]
    if thumbs:
        thumb_html = "".join(
            f'<img src="{_h(item.get("data_url"))}" alt="{_seconds_to_clock(float(item.get("t", 0.0)))}" />'
            for item in thumbs
        )
        strip_html = f'<div class="yt-thumbstrip-real">{thumb_html}</div>'
    else:
        strip_html = '<div class="yt-thumbstrip"></div>'

    st.markdown(
        f"""
<div class="yt-timeline">
  <div class="yt-time-ruler">{ruler_html}</div>
  {waveform_html}
  {strip_html}
</div>
""",
        unsafe_allow_html=True,
    )
    waveform_error = str(waveform.get("error") or "").strip()
    if waveform_error and not peaks:
        st.caption(f"Waveform unavailable: {waveform_error}")


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
    upload_file: Any = None,
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
    if upload_file is not None:
        preview_payload = _extract_first_frame_from_video_bytes(upload_file.getvalue(), suffix)
    elif upload_bytes:
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


def _render_job_diagnostics_panel(diagnostics: Dict[str, Any], is_technical: bool) -> None:
    summary = str(diagnostics.get("summary") or "Diagnostics unavailable.")
    next_action = str(diagnostics.get("next_action") or "").strip()
    severity = str(diagnostics.get("severity") or "info").lower()
    status = str(diagnostics.get("status") or "-")
    stage = str(diagnostics.get("stage") or "-")
    progress = float(diagnostics.get("progress", 0.0) or 0.0)

    if severity == "error":
        st.error(summary)
    elif severity == "warning":
        st.warning(summary)
    elif severity == "success":
        st.success(summary)
    else:
        st.info(summary)

    if next_action:
        st.write(f"Next: {next_action}")

    diag_col1, diag_col2, diag_col3, diag_col4 = st.columns(4)
    diag_col1.metric("Status", status)
    diag_col2.metric("Stage", stage)
    diag_col3.metric("Progress", f"{int(round(progress * 100))}%")
    result_summary = diagnostics.get("result_summary") if isinstance(diagnostics.get("result_summary"), dict) else {}
    diag_col4.metric("Bookmarks", int(result_summary.get("bookmarks_count", 0) or 0))

    issue_logs = list(diagnostics.get("error_logs") or []) + list(diagnostics.get("warning_logs") or [])
    if issue_logs:
        rows = [
            {
                "level": item.get("level"),
                "stage": item.get("stage"),
                "process": ((item.get("data") or {}) if isinstance(item.get("data"), dict) else {}).get("process_message")
                or item.get("message"),
                "technical": ((item.get("data") or {}) if isinstance(item.get("data"), dict) else {}).get("technical_message")
                or "",
                "created_at": _iso_to_short(item.get("created_at")),
            }
            for item in issue_logs[:8]
            if isinstance(item, dict)
        ]
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)

    if is_technical:
        with st.expander("Recent Logs", expanded=False):
            recent_rows = [
                {
                    "level": item.get("level"),
                    "stage": item.get("stage"),
                    "detail": item.get("detail_level"),
                    "process": ((item.get("data") or {}) if isinstance(item.get("data"), dict) else {}).get("process_message")
                    or item.get("message"),
                    "technical": ((item.get("data") or {}) if isinstance(item.get("data"), dict) else {}).get("technical_message")
                    or "",
                    "created_at": _iso_to_short(item.get("created_at")),
                }
                for item in list(diagnostics.get("recent_logs") or [])
                if isinstance(item, dict)
            ]
            if recent_rows:
                st.dataframe(recent_rows, use_container_width=True, hide_index=True)
            else:
                st.caption("No logs yet.")


def _render_job_outcome_notice(job: Dict[str, Any]) -> None:
    status = str(job.get("status") or "").lower()
    result = job.get("result", {}) if isinstance(job.get("result"), dict) else {}
    bookmarks_count = int(result.get("bookmarks_count", 0) or 0)
    error_message = str(job.get("error_message") or "").strip()
    if status == "failed":
        st.error(error_message or "This run failed. Open Run Monitor for diagnostics and logs.")
    elif status == "canceled":
        st.warning(error_message or "This run was canceled.")
    elif status == "completed" and bookmarks_count <= 0:
        st.warning("Run completed, but no bookmarks were detected. Try a longer test window or broader targets.")
    elif status == "completed":
        st.success(f"Run completed with {bookmarks_count} bookmarks.")


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
        "yolo_model": "yolo26s.pt",
        "tracker_config": "botsort.yaml",
        "inference_imgsz": 960,
        "detection_conf": 0.18,
        "vid_stride": 1,
        "overlay": False,
        "no_audio": False,
        "require_gpu": False,
        "analysis_only": True,
        "log_profile": "detailed",
    }


def _option_index(options: List[Any], value: Any, default: int = 0) -> int:
    try:
        return options.index(value)
    except ValueError:
        return default


def _bounded_float(value: Any, fallback: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = fallback
    return min(maximum, max(minimum, parsed))


def _render_gpu_analysis_controls(
    context_key: str,
    config: Optional[Dict[str, Any]] = None,
    *,
    expanded: bool = True,
) -> Dict[str, Any]:
    config = dict(config or {})
    current_model = str(config.get("yolo_model") or "yolo26s.pt")
    current_model_is_custom = current_model not in YOLO_MODEL_OPTIONS
    current_tracker = str(config.get("tracker_config") or "botsort.yaml")
    current_imgsz = int(config.get("inference_imgsz", 960) or 960)
    if current_imgsz not in INFERENCE_IMAGE_SIZE_OPTIONS:
        current_imgsz = 960
    current_stride = int(config.get("vid_stride", 1) or 1)
    if current_stride not in VID_STRIDE_OPTIONS:
        current_stride = 1

    with st.expander("GPU Analysis", expanded=expanded):
        gpu_col1, gpu_col2 = st.columns([1.1, 0.9])
        yolo_model = gpu_col1.selectbox(
            "Detector",
            YOLO_MODEL_OPTIONS,
            index=_option_index(YOLO_MODEL_OPTIONS, current_model, default=(len(YOLO_MODEL_OPTIONS) - 1 if current_model_is_custom else 0)),
            key=f"{context_key}_yolo_model",
        )
        tracker_config = gpu_col2.selectbox(
            "Tracker",
            TRACKER_CONFIG_OPTIONS,
            index=_option_index(TRACKER_CONFIG_OPTIONS, current_tracker),
            key=f"{context_key}_tracker_config",
        )
        gpu_col3, gpu_col4, gpu_col5 = st.columns(3)
        inference_imgsz = int(
            gpu_col3.select_slider(
                "Image Size",
                options=INFERENCE_IMAGE_SIZE_OPTIONS,
                value=current_imgsz,
                key=f"{context_key}_inference_imgsz",
            )
        )
        detection_conf = float(
            gpu_col4.slider(
                "Confidence",
                0.05,
                0.5,
                _bounded_float(config.get("detection_conf", 0.18), 0.18, 0.05, 0.5),
                0.01,
                key=f"{context_key}_detection_conf",
            )
        )
        vid_stride = int(
            gpu_col5.select_slider(
                "Frame Stride",
                options=VID_STRIDE_OPTIONS,
                value=current_stride,
                key=f"{context_key}_vid_stride",
            )
        )
        st.caption(
            "Use YOLO26s/YOLO26m with image size 960 or 1280 to put more of the analysis load on the GPU. "
            "Frame stride 1 analyzes every frame."
        )
        custom_model_path = st.text_input(
            "Custom Detector Weights",
            value=current_model if current_model_is_custom else "",
            placeholder=r"C:\path\to\runs\detect\train\weights\best.pt",
            key=f"{context_key}_custom_yolo_model",
            help="Use this for a fine-tuned Ultralytics detector. Leave blank to use the selected pretrained model.",
        )

    selected_model = str(yolo_model)
    resolved_model = str(custom_model_path or ("yolo26s.pt" if selected_model == "Custom .pt path" else selected_model)).strip()
    return {
        "yolo_model": resolved_model,
        "tracker_config": str(tracker_config),
        "inference_imgsz": int(inference_imgsz),
        "detection_conf": float(detection_conf),
        "vid_stride": int(vid_stride),
    }


def _render_portal_header() -> None:
    st.markdown(
        """
<div class="yt-studio-topbar">
  <div class="yt-brand-mark"><span class="yt-brand-play">▶</span><span>Highlights Studio</span></div>
  <div class="yt-search-pill">Search games, runs, clips, and exports</div>
  <div class="portal-statusbar">
    <span class="portal-chip">Create</span>
    <span class="portal-chip">Tenant workspace</span>
  </div>
</div>
<div class="portal-header">
  <div>
    <div class="portal-kicker">Channel content</div>
    <h1>Match Operations Studio</h1>
    <p>Manage source video, zoomed follow-cam clips, audio, exports, and diagnostics.</p>
  </div>
  <div class="portal-statusbar">
    <span class="portal-chip">Zoom-first editor</span>
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


def _format_file_size(size_bytes: Any) -> str:
    try:
        size = float(size_bytes or 0)
    except Exception:
        size = 0.0
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    return f"{size:.1f} {units[unit_index]}"


def _format_media_probe_summary(probe: Dict[str, Any]) -> str:
    parts = []
    ffprobe = probe.get("ffprobe") if isinstance(probe.get("ffprobe"), dict) else {}
    duration = ffprobe.get("duration_seconds")
    try:
        if duration is not None:
            parts.append(f"duration {_seconds_to_clock(float(duration))}")
    except Exception:
        pass
    width = ffprobe.get("width")
    height = ffprobe.get("height")
    if width and height:
        parts.append(f"{width}x{height}")
    codec = str(ffprobe.get("codec_name") or "").strip()
    if codec:
        parts.append(codec)
    if ffprobe and not ffprobe.get("ok") and ffprobe.get("error"):
        parts.append(f"ffprobe: {ffprobe.get('error')}")
    return " | ".join(parts)


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone().replace(tzinfo=None)
        return parsed
    except Exception:
        return None


def _format_elapsed_since(value: Any) -> str:
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        return "-"
    elapsed = max(0, int((datetime.now() - parsed).total_seconds()))
    hours = elapsed // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _resolve_job_output_dir(job: Dict[str, Any]) -> str:
    config = job.get("config", {}) if isinstance(job.get("config"), dict) else {}
    result = job.get("result", {}) if isinstance(job.get("result"), dict) else {}
    candidates = [
        result.get("output_dir"),
        config.get("output_dir"),
        os.path.join("outputs", str(job.get("job_id") or "").strip()),
    ]
    for raw_path in candidates:
        path = str(raw_path or "").strip()
        if not path:
            continue
        path = os.path.expandvars(os.path.expanduser(path))
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        if os.path.isdir(path):
            return path
    fallback = str(candidates[-1] or "").strip()
    if not fallback:
        return ""
    fallback = os.path.expandvars(os.path.expanduser(fallback))
    return fallback if os.path.isabs(fallback) else os.path.abspath(fallback)


def _collect_job_artifacts(job: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = _resolve_job_output_dir(job)
    artifacts: List[Dict[str, Any]] = []
    file_names = set()
    total_bytes = 0
    if output_dir and os.path.isdir(output_dir):
        try:
            for name in sorted(os.listdir(output_dir)):
                path = os.path.join(output_dir, name)
                if not os.path.isfile(path):
                    continue
                try:
                    size_bytes = os.path.getsize(path)
                    modified_at = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%H:%M:%S")
                except OSError:
                    size_bytes = 0
                    modified_at = "-"
                file_names.add(name)
                total_bytes += int(size_bytes or 0)
                artifacts.append(
                    {
                        "name": name,
                        "size_bytes": int(size_bytes or 0),
                        "size": _format_file_size(size_bytes),
                        "modified_at": modified_at,
                    }
                )
        except OSError:
            pass
    return {
        "output_dir": output_dir,
        "exists": bool(output_dir and os.path.isdir(output_dir)),
        "files": artifacts,
        "file_names": file_names,
        "total_bytes": total_bytes,
    }


def _artifact_present(artifacts: Dict[str, Any], *names: str) -> bool:
    file_names = artifacts.get("file_names", set())
    return any(name in file_names for name in names)


def _latest_artifact_names(artifacts: Dict[str, Any], limit: int = 4) -> str:
    files = list(artifacts.get("files") or [])
    if not files:
        return "No files written yet"
    latest = files[-limit:]
    return ", ".join(f"{item.get('name')} ({item.get('size')})" for item in latest)


def _runtime_signal_lines(job_id: str, limit: int = 12) -> List[str]:
    log_paths = [
        os.path.join("logs", "api.out.log"),
        os.path.join("logs", "api.err.log"),
    ]
    signal_terms = (
        "[1/5]",
        "[2/5]",
        "[3/5]",
        "[4/5]",
        "[5/5]",
        "YOLO runtime",
        "Using device",
        "GPU:",
        "Analysis-only mode",
        "MoviePy",
        "frame_index:",
        "Wrote analysis",
        "tracking",
        "bookmarks",
    )
    normal_lines: List[str] = []
    progress_lines: List[str] = []
    for log_path in log_paths:
        if not os.path.isfile(log_path):
            continue
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
                lines = handle.readlines()[-500:]
        except OSError:
            continue
        for raw_line in lines:
            line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw_line).strip()
            if not line:
                continue
            if job_id and job_id in line:
                normal_lines.append(line)
                continue
            if any(term.lower() in line.lower() for term in signal_terms):
                if line.startswith("frame_index:"):
                    progress_lines.append(line)
                else:
                    normal_lines.append(line)
    combined: List[str] = []
    seen = set()
    for line in normal_lines[-(limit - 2) :] + progress_lines[-2:]:
        if line in seen:
            continue
        seen.add(line)
        combined.append(line)
    return combined[-limit:]


def _first_gpu_snapshot(gpu_health: Dict[str, Any]) -> Dict[str, Any]:
    nvidia_info = gpu_health.get("nvidia_smi", {}) if isinstance(gpu_health.get("nvidia_smi"), dict) else {}
    gpus = nvidia_info.get("gpus", []) if isinstance(nvidia_info.get("gpus"), list) else []
    if gpus and isinstance(gpus[0], dict):
        return dict(gpus[0])
    return {}


def _run_stage_cards(job: Dict[str, Any], artifacts: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    status = str(job.get("status") or "").lower()
    config = job.get("config", {}) if isinstance(job.get("config"), dict) else {}
    result = job.get("result", {}) if isinstance(job.get("result"), dict) else {}
    analysis_only = bool(config.get("analysis_only", True))
    terminal_success = status == "completed"
    terminal_problem = status in {"failed", "canceled", "cancel_requested"}
    has_trim = _artifact_present(artifacts, "trimmed_working_video.mp4")
    has_tracking = _artifact_present(artifacts, "analysis_tracking.json")
    has_bookmarks = _artifact_present(artifacts, "analysis_bookmarks.json", "analysis_bookmarks.csv")
    clip_files = [
        item
        for item in list(artifacts.get("files") or [])
        if str(item.get("name", "")).lower().endswith(".mp4")
        and str(item.get("name", "")).lower() != "trimmed_working_video.mp4"
    ]
    bookmark_count = int(result.get("bookmarks_count", 0) or 0)

    stage_defs = [
        {
            "key": "queued",
            "label": "Queued",
            "desc": "Run accepted by the worker queue.",
            "done": bool(job.get("started_at")) or status in {"claimed", "running", "completed", "failed"},
        },
        {
            "key": "source",
            "label": "Source",
            "desc": "Video path, trim window, model, and GPU requirements resolved.",
            "done": bool(job.get("started_at")) or has_trim or terminal_success,
        },
        {
            "key": "trim",
            "label": "Trim",
            "desc": "Working video generated for the selected test window.",
            "done": has_trim or has_tracking or has_bookmarks or terminal_success,
        },
        {
            "key": "track",
            "label": "YOLO26 Track",
            "desc": "CUDA detector/tracker is scanning players, ball, and motion.",
            "done": has_tracking or has_bookmarks or terminal_success,
        },
        {
            "key": "signals",
            "label": "Signal Score",
            "desc": "Motion, audio, and event signals are converted into review candidates.",
            "done": has_bookmarks or terminal_success,
        },
        {
            "key": "bookmarks",
            "label": "Bookmarks",
            "desc": "Review markers are written for the timeline and table.",
            "done": has_bookmarks or bookmark_count > 0 or terminal_success,
        },
    ]
    if analysis_only:
        stage_defs.append(
            {
                "key": "finish",
                "label": "Ready",
                "desc": "Analysis-only run is finalized for review.",
                "done": terminal_success,
            }
        )
    else:
        stage_defs.extend(
            [
                {
                    "key": "export",
                    "label": "Export",
                    "desc": "Clips, overlays, and montage files are rendered.",
                    "done": bool(clip_files) or terminal_success,
                },
                {
                    "key": "finish",
                    "label": "Ready",
                    "desc": "Rendered output is finalized.",
                    "done": terminal_success,
                },
            ]
        )

    active_index = 0
    for idx, stage in enumerate(stage_defs):
        if not stage["done"]:
            active_index = idx
            break
    else:
        active_index = len(stage_defs) - 1

    active_label = str(stage_defs[active_index]["label"])
    for idx, stage in enumerate(stage_defs):
        if stage["done"]:
            stage["state"] = "done"
            stage["state_label"] = "Confirmed"
        elif terminal_problem and idx == active_index:
            stage["state"] = "fail"
            stage["state_label"] = status.replace("_", " ").title()
        elif idx == active_index and status in {"queued", "claimed", "running", "cancel_requested"}:
            stage["state"] = "active"
            stage["state_label"] = "Running Now" if status != "queued" else "Waiting"
        else:
            stage["state"] = "pending"
            stage["state_label"] = "Waiting"
    return stage_defs, active_label


def _render_meter(label: str, value: float, detail: str) -> str:
    clamped = max(0.0, min(100.0, float(value or 0.0)))
    return f"""
<div class="run-meter-row">
  <div class="run-meter-label">{_h(label)}</div>
  <div class="run-meter-track"><div class="run-meter-fill" style="width:{clamped:.1f}%"></div></div>
  <div class="run-meter-value">{_h(detail)}</div>
</div>
"""


def _render_realtime_job_timeline(
    job: Dict[str, Any],
    diagnostics: Dict[str, Any],
    gpu_health: Dict[str, Any],
) -> None:
    config = job.get("config", {}) if isinstance(job.get("config"), dict) else {}
    result = job.get("result", {}) if isinstance(job.get("result"), dict) else {}
    status = str(job.get("status") or "-")
    job_id = str(job.get("job_id") or "").strip()
    artifacts = _collect_job_artifacts(job)
    stages, active_label = _run_stage_cards(job, artifacts)
    gpu = _first_gpu_snapshot(gpu_health)
    gpu_util = float(gpu.get("utilization_gpu_percent", 0) or 0)
    memory_used = float(gpu.get("memory_used_mb", 0) or 0)
    memory_total = float(gpu.get("memory_total_mb", 0) or 0)
    memory_pct = (memory_used / memory_total * 100.0) if memory_total > 0 else 0.0
    db_progress = max(0.0, min(100.0, float(job.get("progress", 0.0) or 0.0) * 100.0))
    confirmed_count = sum(1 for stage in stages if stage.get("state") == "done")
    milestone_pct = (confirmed_count / max(1, len(stages))) * 100.0
    detector = str(config.get("yolo_model") or config.get("model_version") or "-")
    tracker = str(config.get("tracker_config") or "-")
    imgsz = str(config.get("inference_imgsz") or "-")
    conf = str(config.get("detection_conf") or "-")
    stride = str(config.get("vid_stride") or "-")
    elapsed = _format_elapsed_since(job.get("started_at") or job.get("created_at"))
    bookmark_count = int(result.get("bookmarks_count", 0) or 0)
    source_summary = _latest_artifact_names(artifacts)
    live_text = "LIVE" if status.lower() in {"queued", "claimed", "running", "cancel_requested"} else status.upper()

    cards_html = "".join(
        f"""
<div class="run-stage-card {_h(stage.get('state'))}">
  <p class="run-stage-label">{_h(stage.get('label'))}</p>
  <p class="run-stage-state">{_h(stage.get('state_label'))}</p>
  <p class="run-stage-desc">{_h(stage.get('desc'))}</p>
</div>
"""
        for stage in stages
    )
    meters_html = "".join(
        [
            _render_meter("GPU Load", gpu_util, f"{gpu_util:.0f}%"),
            _render_meter(
                "VRAM",
                memory_pct,
                f"{memory_used / 1024.0:.1f}/{memory_total / 1024.0:.1f} GB" if memory_total else "not reported",
            ),
            _render_meter("DB Progress", db_progress, f"{db_progress:.0f}% stored"),
            _render_meter("Confirmed Milestones", milestone_pct, f"{confirmed_count}/{len(stages)} stages"),
        ]
    )
    signal_html = f"""
<div class="run-signal-strip">
  <div class="run-signal-chip"><b>{_h(active_label)}</b><span>current inferred stage</span></div>
  <div class="run-signal-chip"><b>{_h(detector)}</b><span>{_h(tracker)} | imgsz {_h(imgsz)} | conf {_h(conf)} | stride {_h(stride)}</span></div>
  <div class="run-signal-chip"><b>{_h(elapsed)}</b><span>elapsed since worker start</span></div>
  <div class="run-signal-chip"><b>{_h(bookmark_count)}</b><span>bookmarks written so far</span></div>
  <div class="run-signal-chip"><b>{_h(_format_file_size(artifacts.get('total_bytes', 0)))}</b><span>{_h(source_summary)}</span></div>
</div>
"""
    log_lines = _runtime_signal_lines(job_id)
    logs_html = ""
    if log_lines:
        logs_html = "<div class=\"run-log-strip\">" + "".join(f"<div>{_h(line)}</div>" for line in log_lines) + "</div>"

    diag_summary = str(diagnostics.get("summary") or "").strip()
    st.markdown(
        f"""
<div class="run-monitor-shell">
  <div class="run-monitor-header">
    <div>
      <div class="run-monitor-kicker">Realtime Run Timeline</div>
      <div class="run-monitor-title">{_h(active_label)} · { _h(status.replace('_', ' ').title()) }</div>
      <div class="studio-muted">{_h(diag_summary or 'Monitoring job state, artifacts, and GPU telemetry.')}</div>
    </div>
    <div class="run-live-badge"><span class="run-live-dot"></span>{_h(live_text)}</div>
  </div>
  <div class="run-stage-grid">{cards_html}</div>
  {meters_html}
  {signal_html}
  {logs_html}
</div>
""",
        unsafe_allow_html=True,
    )


def _local_video_probe(path: str) -> Dict[str, Any]:
    clean_path = str(path or "").strip().strip('"')
    if not clean_path:
        return {"ok": False, "code": "path_required", "path": "", "message": "Choose a local video file path."}
    if not os.path.isfile(clean_path):
        return {"ok": False, "code": "not_found", "path": clean_path, "message": "Local video file was not found."}
    try:
        size_bytes = os.path.getsize(clean_path)
    except OSError as exc:
        return {"ok": False, "code": "unreadable", "path": clean_path, "message": f"Could not read local file size: {exc}"}
    if size_bytes <= 0:
        return {
            "ok": False,
            "code": "zero_bytes",
            "path": clean_path,
            "size_bytes": size_bytes,
            "message": (
                "Windows reports this file is 0 bytes. If it is still copying, syncing, or a cloud placeholder, "
                "wait for the real local file to finish downloading before launching."
            ),
        }
    extension = os.path.splitext(clean_path)[1].lower()
    if extension not in {".mp4", ".mov", ".mkv", ".avi", ".m4v"}:
        return {
            "ok": False,
            "code": "bad_extension",
            "path": clean_path,
            "size_bytes": size_bytes,
            "message": "Use a video file ending in .mp4, .mov, .mkv, .avi, or .m4v.",
        }
    return {"ok": True, "code": "ready", "path": clean_path, "size_bytes": size_bytes, "message": "Local video ready."}


def _choose_local_video_file(initial_path: str = "") -> Dict[str, str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        return {"path": "", "error": f"Native file chooser is unavailable: {exc}"}

    initial_dir = ""
    clean_initial = str(initial_path or "").strip().strip('"')
    if clean_initial:
        initial_dir = clean_initial if os.path.isdir(clean_initial) else os.path.dirname(clean_initial)
    if not initial_dir or not os.path.isdir(initial_dir):
        initial_dir = os.path.expanduser("~")

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        selected = filedialog.askopenfilename(
            parent=root,
            title="Choose local match video",
            initialdir=initial_dir,
            filetypes=[
                ("Video files", "*.mp4 *.mov *.mkv *.avi *.m4v"),
                ("All files", "*.*"),
            ],
        )
        return {"path": str(selected or ""), "error": ""}
    except Exception as exc:
        return {"path": "", "error": f"Could not open native file chooser: {exc}"}
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


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


def _studio_match_selector_label(match: Dict[str, Any]) -> str:
    return (
        f"{(match.get('name') or match.get('match_id'))} | "
        f"{match.get('home_team_name') or '?'} vs {match.get('away_team_name') or '?'} | "
        f"{match.get('match_date') or '-'}"
    )


def _library_match_selector_label(row: Dict[str, Any]) -> str:
    return f"{row['game']} | {row['date']} | {row['latest_status']}"


def _option_label_for_entity_id(options_by_label: Dict[str, Any], entity_id: str) -> Tuple[str, int]:
    target_id = str(entity_id or "").strip()
    if not target_id:
        return "", 0
    for idx, (label, value) in enumerate(options_by_label.items()):
        if str(value) == target_id:
            return label, idx
        if isinstance(value, dict):
            if str(value.get("match_id") or value.get("job_id") or "") == target_id:
                return label, idx
    return "", 0


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
        "training": "Training Lab",
        "training_lab": "Training Lab",
        "training-lab": "Training Lab",
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
                "Training Lab": "Training",
                "Operations Console": "Run Monitor",
            }
            st.session_state.portal_user_nav_main = user_nav_aliases.get(target_nav, "Studio")

    match_ref = str(query_params.get("match") or query_params.get("match_id") or "").strip()
    match_id = _extract_entity_id_from_ref(match_ref, "match")
    if match_id:
        st.session_state.selected_match_id = match_id
        st.session_state.portal_pending_match_select_id = match_id
        if "--" in match_ref:
            raw_label = match_ref.split("--", 1)[0]
            if raw_label and raw_label not in {"match", "run", "job"}:
                st.session_state.selected_match_label = raw_label.replace("-", " ").strip()

    job_ref = str(query_params.get("job") or query_params.get("job_id") or "").strip()
    job_id = _extract_entity_id_from_ref(job_ref, "job")
    if job_id:
        st.session_state.selected_job_id = job_id
        st.session_state.portal_pending_job_select_id = job_id
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
        st.session_state.portal_pending_workspace_view = "Review"
        st.session_state.portal_studio_focus_review = True


def _view_code_from_nav(nav_key: str, is_technical: bool) -> str:
    if nav_key == "New Processing Run":
        return "upload"
    if nav_key == "Game Library":
        return "library"
    if nav_key == "Training Lab":
        return "training"
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
    match_clean = _extract_entity_id_from_ref(str(match_id or "").strip(), "match")
    if match_clean:
        params["match"] = _build_entity_ref(match_clean, match_label)
    job_clean = _extract_entity_id_from_ref(str(job_id or "").strip(), "job")
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
    pending_job_id = _extract_entity_id_from_ref(
        str(st.session_state.get("portal_pending_job_select_id", "") or ""),
        "job",
    )
    default_job_id = pending_job_id or _extract_entity_id_from_ref(previous_job_id, "job")
    desired_job_label, default_index = _option_label_for_entity_id(job_picker, default_job_id)
    review_job_key = f"{context_key}_review_job"
    current_job_label = str(st.session_state.get(review_job_key, "") or "")
    if desired_job_label and (pending_job_id or current_job_label not in labels):
        st.session_state[review_job_key] = desired_job_label
        if pending_job_id:
            st.session_state.portal_pending_job_select_id = ""
    elif current_job_label and current_job_label not in labels and labels:
        st.session_state[review_job_key] = labels[default_index]

    selected_job_label = st.selectbox(
        "Bookmark Source Job",
        labels,
        index=default_index,
        key=review_job_key,
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

            review_link = _build_portal_share_link(share_params)
            bookmark_link = _build_portal_share_link(bookmark_params)
            st.caption("Share Links")
            st.markdown(
                f"""
<div class="share-link-row">
  <a href="{_h(review_link)}" target="_self">Open selected run</a>
  <a href="{_h(bookmark_link)}" target="_self">Open selected bookmark</a>
</div>
""",
                unsafe_allow_html=True,
            )
            link_col1, link_col2 = st.columns(2)
            link_col1.text_input(
                "Run link",
                value=review_link,
                key=f"{context_key}_share_review_{selected_match_id}_{selected_job_id}",
            )
            link_col2.text_input(
                "Bookmark link",
                value=bookmark_link,
                key=f"{context_key}_share_bookmark_{selected_match_id}_{selected_job_id}",
            )

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


def _render_studio_editor(
    *,
    api_base: str,
    tenant_id: str,
    token: str,
    selected_match: Dict[str, Any],
    selected_match_id: str,
    jobs: List[Dict[str, Any]],
    source_video: str,
    monitor_nav_label: str,
) -> None:
    latest_job = _latest_job(jobs)
    latest_config = dict((latest_job or {}).get("config", {}) or _recommended_job_config())
    if str(latest_config.get("camera_mode", "wide")) == "wide":
        latest_config["camera_mode"] = "follow_action"
    latest_config.setdefault("zoom_factor", 1.8)
    latest_config.setdefault("analysis_only", False)

    st.markdown('<div class="yt-editor-shell">', unsafe_allow_html=True)
    editor_col, preview_col = st.columns([1.0, 1.35])
    with editor_col:
        st.markdown('<p class="yt-panel-title">Video editor</p>', unsafe_allow_html=True)
        st.markdown(
            """
<div class="yt-section-row"><div>Zoom & follow-cam<br><span>Center the selected player or action</span></div><div class="yt-plus">+</div></div>
<div class="yt-section-row"><div>Player lock<br><span>Use first frame ROI to choose the target</span></div><div class="yt-plus">+</div></div>
<div class="yt-section-row"><div>Audio<br><span>Clean, remove, replace, or mix</span></div><div class="yt-plus">+</div></div>
<div class="yt-section-row"><div>Clips<br><span>Export bookmarked moments</span></div><div class="yt-plus">+</div></div>
""",
            unsafe_allow_html=True,
        )

        cam_col1, cam_col2 = st.columns([1.0, 1.0])
        camera_options = ["follow_action", "follow_player", "wide"]
        current_camera = str(latest_config.get("camera_mode") or "follow_action")
        if current_camera not in camera_options:
            current_camera = "follow_action"
        camera_mode = cam_col1.selectbox(
            "Camera",
            camera_options,
            index=camera_options.index(current_camera),
            key=f"studio_editor_camera_{selected_match_id}",
        )
        zoom_factor = cam_col2.slider(
            "Zoom",
            1.0,
            3.0,
            min(3.0, max(1.0, float(latest_config.get("zoom_factor", 1.8) or 1.8))),
            0.1,
            key=f"studio_editor_zoom_{selected_match_id}",
        )
        st.caption("Zoom presets to try: 1.4x wide context, 1.8x balanced, 2.2x player focus, 2.6x tight clips.")

        gpu_settings = _render_gpu_analysis_controls(
            context_key=f"studio_editor_{selected_match_id}",
            config=latest_config,
            expanded=True,
        )

        trim_window = _render_trim_window_controls(
            context_key=f"studio_editor_{selected_match_id}",
            config=latest_config,
            default_enabled=False,
            default_duration_minutes=3.0,
        )
        analysis_only = st.checkbox(
            "Analysis only",
            value=bool(latest_config.get("analysis_only", False)),
            key=f"studio_editor_analysis_only_{selected_match_id}",
        )

        with st.expander("Player Lock", expanded=True):
            player_roi_payload = _render_player_roi_selector(
                context_key=f"studio_editor_{selected_match_id}",
                default_enabled=bool(latest_config.get("player_roi")),
                existing_roi=(latest_config.get("player_roi") if isinstance(latest_config.get("player_roi"), dict) else None),
                source_video_path=source_video or "",
            )

        if st.button(
            "Reprocess With Zoom Settings",
            type="primary",
            use_container_width=True,
            key=f"studio_editor_reprocess_{selected_match_id}",
            disabled=not bool(source_video),
        ):
            if player_roi_payload.get("enabled") and not player_roi_payload.get("roi"):
                st.error("Player lock is enabled, but the first-frame selector is not ready.")
            else:
                next_config = dict(latest_config)
                next_config.update(
                    {
                        "camera_mode": str(camera_mode),
                        "zoom_factor": float(zoom_factor),
                        "analysis_only": bool(analysis_only),
                        **gpu_settings,
                        "run_created_at": datetime.utcnow().isoformat(),
                        "run_created_from": "studio_youtube_editor",
                        "test_window_enabled": bool(trim_window["enabled"]),
                    }
                )
                if trim_window["enabled"]:
                    next_config["trim_start"] = float(trim_window["trim_start"])
                    next_config["trim_end"] = float(trim_window["trim_end"])
                else:
                    next_config.pop("trim_start", None)
                    next_config.pop("trim_end", None)
                if player_roi_payload.get("roi"):
                    next_config["player_roi"] = dict(player_roi_payload["roi"])
                elif "player_roi" in next_config:
                    next_config.pop("player_roi", None)

                create_result = api_request(
                    "POST",
                    api_base,
                    f"/matches/{selected_match_id}/jobs",
                    tenant_id,
                    token,
                    json_body={"config": next_config},
                )
                if not create_result["ok"]:
                    st.error(f"Failed to queue zoom reprocess: {create_result['payload']}")
                else:
                    new_job_id = str(create_result["payload"]["job_id"])
                    st.session_state.selected_job_id = new_job_id
                    st.session_state.portal_pending_job_select_id = new_job_id
                    st.session_state.portal_pending_workspace_view = "Live"
                    st.session_state.portal_pending_nav = monitor_nav_label
                    st.session_state.portal_flash_message = f"Zoom reprocess queued. job_id={new_job_id}"
                    safe_rerun()

    with preview_col:
        st.markdown('<p class="yt-panel-title">Preview</p>', unsafe_allow_html=True)
        if source_video:
            seek_seconds = st.number_input(
                "Timeline position",
                min_value=0,
                value=int(st.session_state.portal_video_seek_s or 0),
                step=1,
                key=f"studio_editor_seek_{selected_match_id}",
            )
            st.session_state.portal_video_seek_s = int(seek_seconds)
            st.video(source_video, start_time=int(seek_seconds))
        else:
            st.warning("This match has no playable source video.")

        stat_col1, stat_col2, stat_col3 = st.columns(3)
        stat_col1.metric("Camera", str(camera_mode))
        stat_col2.metric("Zoom", f"{float(zoom_factor):.1f}x")
        stat_col3.metric("Window", trim_window["label"])

    st.markdown("</div>", unsafe_allow_html=True)
    timeline_key = f"studio_timeline_{selected_match_id}"
    timeline_col1, timeline_col2 = st.columns([1.0, 3.0])
    refresh_timeline = timeline_col1.button(
        "Refresh Timeline Media",
        key=f"studio_timeline_refresh_{selected_match_id}",
        use_container_width=True,
        disabled=not bool(source_video),
    )
    if source_video and (refresh_timeline or timeline_key not in st.session_state):
        with st.spinner("Building timeline thumbnails and waveform..."):
            st.session_state[timeline_key] = get_match_timeline(
                api_base=api_base,
                tenant_id=tenant_id,
                token=token,
                match_id=selected_match_id,
                thumbnail_count=20,
                waveform_bins=140,
            )
    timeline = st.session_state.get(timeline_key, {})
    if isinstance(timeline, dict) and timeline.get("ok"):
        video_meta = timeline.get("video", {}) if isinstance(timeline.get("video"), dict) else {}
        timeline_col2.caption(
            f"Timeline media: `{_seconds_to_clock(float(video_meta.get('duration_seconds') or 0.0))}` | "
            f"`{video_meta.get('width') or '-'}x{video_meta.get('height') or '-'}` | "
            f"{len(list(timeline.get('thumbnails', []) or []))} thumbnails"
        )
        _render_timeline_media(timeline)
    else:
        if isinstance(timeline, dict) and timeline.get("error"):
            timeline_col2.warning(f"Timeline media unavailable: {timeline.get('error')}")
        _render_timeline_media({"video": {"duration_seconds": 0.0}, "thumbnails": [], "waveform": {"peaks": []}})


def _render_studio_audio(
    *,
    api_base: str,
    tenant_id: str,
    token: str,
    selected_match: Dict[str, Any],
    selected_match_id: str,
    source_video: str,
) -> None:
    st.subheader("Audio")
    metadata = selected_match.get("metadata", {}) if isinstance(selected_match.get("metadata"), dict) else {}
    audio_edits = list(metadata.get("audio_edits", []) or [])

    controls_col, preview_col = st.columns([1.0, 1.2])
    with controls_col:
        st.markdown('<div class="audio-edit-card">', unsafe_allow_html=True)
        mode_labels = {
            "Clean original audio": "keep",
            "Remove all audio": "remove",
            "Replace with MP3": "replace",
            "Mix MP3 under original": "mix",
        }
        selected_mode_label = st.selectbox(
            "Audio operation",
            list(mode_labels.keys()),
            key=f"studio_audio_mode_{selected_match_id}",
        )
        mode = mode_labels[selected_mode_label]

        cleanup_labels = {
            "None": "none",
            "Stadium clean": "stadium_clean",
            "Reduce wind noise": "wind_reduce",
            "Reduce conversations": "speech_reduce",
            "AI RNNoise model": "ai_rnnoise",
        }
        cleanup_label = st.selectbox(
            "Cleanup",
            list(cleanup_labels.keys()),
            index=0 if mode in {"replace", "remove"} else 2,
            disabled=mode in {"replace", "remove"},
            key=f"studio_audio_cleanup_{selected_match_id}",
        )
        cleanup_profile = "none" if mode in {"replace", "remove"} else cleanup_labels[cleanup_label]

        original_volume = st.slider(
            "Original volume",
            0.0,
            2.0,
            1.0,
            0.05,
            disabled=mode in {"replace", "remove"},
            key=f"studio_audio_original_volume_{selected_match_id}",
        )
        music_volume = st.slider(
            "MP3 volume",
            0.0,
            2.0,
            0.35 if mode == "mix" else 1.0,
            0.05,
            disabled=mode not in {"replace", "mix"},
            key=f"studio_audio_music_volume_{selected_match_id}",
        )
        uploaded_audio = st.file_uploader(
            "MP3 / audio file",
            type=["mp3", "wav", "m4a", "aac"],
            disabled=mode not in {"replace", "mix"},
            key=f"studio_audio_upload_{selected_match_id}",
        )
        loop_external_audio = st.checkbox(
            "Loop MP3 to match video length",
            value=True,
            disabled=mode not in {"replace", "mix"},
            key=f"studio_audio_loop_{selected_match_id}",
        )
        title = st.text_input(
            "Output title",
            value=f"{selected_match.get('name') or selected_match_id} audio edit",
            key=f"studio_audio_title_{selected_match_id}",
        )

        render_disabled = not bool(source_video) or (mode in {"replace", "mix"} and uploaded_audio is None)
        if st.button(
            "Render Audio Edit",
            type="primary",
            use_container_width=True,
            disabled=render_disabled,
            key=f"studio_audio_render_{selected_match_id}",
        ):
            files = None
            if uploaded_audio is not None:
                files = {
                    "audio_file": (
                        uploaded_audio.name,
                        uploaded_audio.getvalue(),
                        uploaded_audio.type or "audio/mpeg",
                    )
                }
            result = api_request(
                "POST",
                api_base,
                f"/matches/{selected_match_id}/audio/render",
                tenant_id,
                token,
                data={
                    "mode": mode,
                    "cleanup_profile": cleanup_profile,
                    "original_volume": str(float(original_volume)),
                    "music_volume": str(float(music_volume)),
                    "loop_external_audio": str(bool(loop_external_audio)).lower(),
                    "title": title.strip() or "Audio Edit",
                    "expires_seconds": "3600",
                },
                files=files,
                timeout=900,
            )
            if not result["ok"]:
                st.error(f"Audio render failed: {result['payload']}")
            else:
                payload = result["payload"]
                playback = _resolve_clip_playback_source(
                    path=str(payload.get("path") or ""),
                    download_url=str(payload.get("download_url") or ""),
                )
                st.session_state.portal_audio_edit_source = playback
                st.session_state.portal_audio_edit_summary = payload
                st.success("Audio edit rendered.")
        if mode in {"replace", "mix"} and uploaded_audio is None:
            st.caption("Upload an MP3 or audio file to enable this render.")
        if cleanup_profile == "ai_rnnoise":
            st.caption("AI cleanup uses ffmpeg arnndn and requires `VH_RNNOISE_MODEL_PATH` on the API server.")
        st.markdown("</div>", unsafe_allow_html=True)

    with preview_col:
        st.markdown('<p class="yt-panel-title">Preview</p>', unsafe_allow_html=True)
        preview_source = str(st.session_state.get("portal_audio_edit_source", "") or "") or source_video
        if preview_source:
            st.video(preview_source)
        else:
            st.warning("This match has no playable source video.")

        if st.session_state.get("portal_audio_edit_summary"):
            st.caption("Latest audio render")
            st.json(st.session_state.portal_audio_edit_summary)

        if audio_edits:
            rows = [
                {
                    "title": item.get("title") or "Audio Edit",
                    "mode": item.get("mode"),
                    "cleanup": item.get("cleanup_profile"),
                    "created_at": _iso_to_short(item.get("created_at")),
                    "path": item.get("path"),
                }
                for item in sorted(audio_edits, key=lambda entry: str(entry.get("created_at") or ""), reverse=True)
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)
            options = {
                f"{row['title']} | {row['mode']} | {row['created_at']}": row["path"]
                for row in rows
                if row.get("path")
            }
            if options:
                selected_audio = st.selectbox(
                    "Open Audio Edit",
                    list(options.keys()),
                    key=f"studio_audio_history_{selected_match_id}",
                )
                st.video(str(options[selected_audio]))


def _render_match_assistant(
    *,
    api_base: str,
    tenant_id: str,
    token: str,
    selected_match_id: str,
) -> None:
    st.subheader("Match Assistant")
    agent_status = get_agent_status(api_base, tenant_id, token)
    status_col1, status_col2, status_col3 = st.columns(3)
    status_col1.metric("Provider", str(agent_status.get("provider") or "fallback"))
    status_col2.metric("Model", str(agent_status.get("model") or "-"))
    status_col3.metric("Reachable", "yes" if agent_status.get("reachable") is True else "no" if agent_status.get("reachable") is False else "n/a")
    message = str(agent_status.get("message") or "").strip()
    if message:
        st.caption(message)
    with st.expander("Local AI setup", expanded=not bool(agent_status.get("configured"))):
        st.code(
            "\n".join(
                [
                    "$env:VH_LLM_PROVIDER='ollama'",
                    "$env:VH_LLM_MODEL='gemma4:e2b'",
                    "$env:VH_LLM_BASE_URL='http://127.0.0.1:11434'",
                    "python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000",
                ]
            ),
            language="powershell",
        )

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
if "portal_pending_match_select_id" not in st.session_state:
    st.session_state.portal_pending_match_select_id = ""
if "portal_pending_job_select_id" not in st.session_state:
    st.session_state.portal_pending_job_select_id = ""
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
if "portal_audio_edit_source" not in st.session_state:
    st.session_state.portal_audio_edit_source = ""
if "portal_audio_edit_summary" not in st.session_state:
    st.session_state.portal_audio_edit_summary = {}
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
if "portal_studio_workspace_view" not in st.session_state:
    st.session_state.portal_studio_workspace_view = "Overview"
if "portal_pending_workspace_view" not in st.session_state:
    st.session_state.portal_pending_workspace_view = ""

_apply_url_state_to_session()

with st.sidebar:
    st.header("Workspace")
    env_api_base = os.getenv("VH_PORTAL_API_BASE", "").strip()
    api_base_default = env_api_base or "http://127.0.0.1:8000/v1"
    if not env_api_base and str(st.session_state.get("portal_api_base", "")).rstrip("/") == "http://api:8000/v1":
        st.session_state.portal_api_base = api_base_default
    tenant_default = os.getenv("VH_PORTAL_TENANT", "sandbox")
    api_base = st.text_input("API Base URL", value=api_base_default, key="portal_api_base")
    if "://api:" in api_base:
        st.warning("`api` is the Docker Compose hostname. For local Windows runs, use `http://127.0.0.1:8000/v1`.")
    tenant_id = st.text_input("Tenant", value=tenant_default, key="portal_tenant")
    token = st.text_input("Bearer Token", value="", type="password", key="portal_token")
    gpu_health = get_gpu_health(api_base, tenant_id, token)
    st.session_state.portal_gpu_ready = bool(gpu_health.get("ready"))
    torch_info = dict(gpu_health.get("torch", {}) or {})
    nvidia_info = dict(gpu_health.get("nvidia_smi", {}) or {})
    nvenc_info = dict(gpu_health.get("ffmpeg_nvenc", {}) or {})
    gpu_names = []
    if isinstance(torch_info.get("devices"), list):
        gpu_names = [str(item) for item in torch_info.get("devices", []) if str(item).strip()]
    if not gpu_names and isinstance(nvidia_info.get("gpus"), list):
        gpu_names = [str(item.get("name")) for item in nvidia_info.get("gpus", []) if isinstance(item, dict) and item.get("name")]
    if st.session_state.portal_gpu_ready:
        st.success(f"GPU ready: {gpu_names[0] if gpu_names else 'CUDA available'}")
    else:
        st.warning("GPU not ready for PyTorch CUDA. Check `/v1/health/gpu`.")
    if st.session_state.portal_gpu_ready and nvenc_info:
        if nvenc_info.get("available"):
            st.caption("NVENC clip rendering: ready")
        else:
            st.caption("NVENC clip rendering: not detected; clips will fall back to CPU encoding.")
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
                "Training Lab": "Training",
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
            ["Portal Home", "New Processing Run", "Game Library", "Training Lab", "Operations Console"],
            index=0,
            key="portal_nav",
        )
    if experience_mode == "Technical":
        st.caption("Tip: Keep this screen open while worker runs. Use Operations Console for live logs and kill controls.")
    else:
        st.caption("Tip: Use Studio for a YouTube-style game library. Start runs there without re-uploading.")

is_technical = experience_mode == "Technical"
if experience_mode == "User Friendly":
    user_nav_options = ["Studio", "Upload", "Training", "Run Monitor"]
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
        "Training": "Training Lab",
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
            st.subheader("Selected Game")
            library_rows: List[Dict[str, Any]] = []
            for match, latest in filtered_snapshots[:90]:
                library_rows.append(
                    {
                        "game": match.get("name") or match.get("match_id"),
                        "date": match.get("match_date") or "-",
                        "teams": f"{match.get('home_team_name') or '?'} vs {match.get('away_team_name') or '?'}",
                        "status": (latest or {}).get("status", "no_runs"),
                        "bookmarks": int((latest or {}).get("result", {}).get("bookmarks_count", 0) or 0),
                        "window": _format_trim_window((latest or {}).get("config", {}) or {}),
                    }
                )
            list_height = min(300, 58 + (len(library_rows) * 34))
            with st.expander(f"Browse library ({len(library_rows)} games)", expanded=False):
                st.dataframe(
                    library_rows,
                    use_container_width=True,
                    hide_index=True,
                    height=list_height,
                )

            selector = {_studio_match_selector_label(m): str(m.get("match_id")) for m, _ in filtered_snapshots}
            selector_labels = list(selector.keys())
            default_match_id = _extract_entity_id_from_ref(str(st.session_state.selected_match_id or ""), "match")
            pending_match_id = _extract_entity_id_from_ref(
                str(st.session_state.get("portal_pending_match_select_id", "") or ""),
                "match",
            )
            desired_label, default_idx = _option_label_for_entity_id(selector, pending_match_id or default_match_id)
            current_selector_label = str(st.session_state.get("portal_studio_active_match", "") or "")
            if desired_label and (pending_match_id or current_selector_label not in selector_labels):
                st.session_state.portal_studio_active_match = desired_label
                if pending_match_id:
                    st.session_state.portal_pending_match_select_id = ""
            elif current_selector_label and current_selector_label not in selector_labels and selector_labels:
                st.session_state.portal_studio_active_match = selector_labels[default_idx]
            selected_label = st.selectbox(
                "Open Game",
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
                st.subheader(f"Workspace: {selected_match.get('name') or selected_match_id}")
                if bool(st.session_state.portal_studio_focus_review):
                    st.session_state.portal_pending_workspace_view = "Review"
                    st.session_state.portal_studio_focus_review = False
                workspace_options = ["Overview", "Editor", "Audio", "Review", "Live", "Runs", "Exports", "Assistant"]
                pending_workspace_view = str(st.session_state.get("portal_pending_workspace_view", "") or "")
                if pending_workspace_view in workspace_options:
                    st.session_state.portal_studio_workspace_view = pending_workspace_view
                    st.session_state.portal_pending_workspace_view = ""
                if str(st.session_state.get("portal_studio_workspace_view", "")) not in workspace_options:
                    st.session_state.portal_studio_workspace_view = "Overview"
                workspace_view = st.radio(
                    "Workspace View",
                    workspace_options,
                    horizontal=True,
                    key="portal_studio_workspace_view",
                )
                latest_job = _latest_job(jobs)
                source_video = _resolve_match_video_source(selected_match)

                if workspace_view == "Overview":
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

                    action_col1, action_col2, action_col3, action_col4, action_col5 = st.columns(5)
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
                            st.session_state.portal_pending_job_select_id = new_job_id
                            st.session_state.portal_studio_focus_review = False
                            st.session_state.portal_pending_workspace_view = "Live"
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
                        "Open Editor",
                        key=f"studio_workspace_editor_{selected_match_id}",
                        use_container_width=True,
                        disabled=not bool(source_video),
                    ):
                        st.session_state.portal_pending_workspace_view = "Editor"
                        safe_rerun()
                    if action_col4.button(
                        "Open Audio",
                        key=f"studio_workspace_audio_{selected_match_id}",
                        use_container_width=True,
                        disabled=not bool(source_video),
                    ):
                        st.session_state.portal_pending_workspace_view = "Audio"
                        safe_rerun()
                    if action_col5.button(
                        "Open Review & Bookmarks",
                        key=f"studio_workspace_review_{selected_match_id}",
                        use_container_width=True,
                        disabled=not bool(jobs),
                    ):
                        latest_job_id = str((latest_job or {}).get("job_id") or "").strip()
                        if latest_job_id:
                            st.session_state.selected_job_id = latest_job_id
                            st.session_state.portal_pending_job_select_id = latest_job_id
                        st.session_state.portal_studio_focus_review = True
                        st.session_state.portal_pending_workspace_view = "Review"
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

                elif workspace_view == "Editor":
                    _render_studio_editor(
                        api_base=api_base,
                        tenant_id=tenant_id,
                        token=token,
                        selected_match=selected_match,
                        selected_match_id=selected_match_id,
                        jobs=jobs,
                        source_video=source_video,
                        monitor_nav_label=monitor_nav_label,
                    )

                elif workspace_view == "Audio":
                    _render_studio_audio(
                        api_base=api_base,
                        tenant_id=tenant_id,
                        token=token,
                        selected_match=selected_match,
                        selected_match_id=selected_match_id,
                        source_video=source_video,
                    )

                elif workspace_view == "Review":
                    _render_match_bookmark_review(
                        context_key="studio_review",
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

                elif workspace_view == "Live":
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
                        pending_live_job_id = _extract_entity_id_from_ref(
                            str(st.session_state.get("portal_pending_job_select_id", "") or ""),
                            "job",
                        )
                        selected_live_job_id = _extract_entity_id_from_ref(
                            str(st.session_state.get("selected_job_id", "") or ""),
                            "job",
                        )
                        preferred_job_id = pending_live_job_id or selected_live_job_id
                        if preferred_job_id and not any(
                            str(job.get("job_id")) == preferred_job_id for job in jobs
                        ):
                            preferred_job_id = ""
                        default_job_id = preferred_job_id or default_job_id
                        desired_live_label, default_job_idx = _option_label_for_entity_id(job_options, default_job_id)
                        live_job_key = f"studio_live_job_{selected_match_id}"
                        current_live_label = str(st.session_state.get(live_job_key, "") or "")
                        if desired_live_label and (pending_live_job_id or current_live_label not in option_labels):
                            st.session_state[live_job_key] = desired_live_label
                            if pending_live_job_id:
                                st.session_state.portal_pending_job_select_id = ""
                        elif current_live_label and current_live_label not in option_labels and option_labels:
                            st.session_state[live_job_key] = option_labels[default_job_idx]
                        selected_job_label = st.selectbox(
                            "Run",
                            option_labels,
                            index=default_job_idx,
                            key=live_job_key,
                        )
                        selected_job = job_options[selected_job_label]
                        selected_job_id = str(selected_job.get("job_id"))
                        st.session_state.selected_job_id = selected_job_id

                        st.markdown(
                            _build_stage_tracker(selected_job.get("stage"), selected_job.get("status")),
                            unsafe_allow_html=True,
                        )
                        st.progress(float(selected_job.get("progress", 0.0)))
                        _render_job_outcome_notice(selected_job)
                        st.caption(
                            f"Model `{selected_job.get('config', {}).get('model_version', '-')}` | "
                            f"Camera `{selected_job.get('config', {}).get('camera_mode', 'wide')}` | "
                            f"Window `{_format_trim_window(selected_job.get('config', {}) or {})}` | "
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

                elif workspace_view == "Runs":
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
                                    "window": _format_trim_window(job.get("config", {}) or {}),
                                    "targets": _focus_targets_from_config(job.get("config", {}) or {}),
                                    "analysis_only": bool((job.get("config", {}) or {}).get("analysis_only", False)),
                                    "bookmarks": int((job.get("result", {}) or {}).get("bookmarks_count", 0) or 0),
                                    "error": job.get("error_message") or "",
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

                elif workspace_view == "Exports":
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

                elif workspace_view == "Assistant":
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
        left, right = st.columns([1.0, 2.1])
        with left:
            st.markdown("#### Game")
            match_name = st.text_input("Game Name", value="U13 Matchday", key="portal_new_match_name")
            home_team = st.text_input("Home Team", value="Home", key="portal_new_home_team")
            away_team = st.text_input("Away Team", value="Away", key="portal_new_away_team")
            match_date = st.text_input("Match Date", value=datetime.utcnow().strftime("%Y-%m-%d"), key="portal_new_match_date")
            if bool(st.session_state.get("portal_gpu_ready", False)):
                st.success("GPU ready")
            else:
                st.warning("GPU not ready")

        with right:
            upload = None
            local_video_path = ""
            local_video_probe: Dict[str, Any] = {"ok": False, "message": "Choose a local video file path."}
            ingest_mode = "Local File Path (10GB+)"
            profile_name = "Balanced"
            model_version = model_versions[0] if model_versions else "event-v0"
            selected_targets: List[str] = []
            custom_targets = ""
            pre_seconds = 2.0
            post_seconds = 6.0
            min_clip = 4.0
            speed_sens = 2.0
            audio_sens = 2.0
            thread_count = 0
            camera_mode = "wide"
            zoom_factor = 1.6
            gpu_settings = {
                "yolo_model": "yolo26s.pt",
                "tracker_config": "botsort.yaml",
                "inference_imgsz": 960,
                "detection_conf": 0.18,
                "vid_stride": 1,
            }
            no_audio = False
            overlay = False
            require_gpu = bool(st.session_state.get("portal_gpu_ready", False))
            analysis_only = True
            log_profile = "detailed"
            player_roi_payload: Dict[str, Any] = {"enabled": False, "roi": None}

            source_tab, analysis_tab, output_tab, advanced_tab = st.tabs(["Source", "Analysis", "Output", "Advanced"])

            with source_tab:
                ingest_mode = st.radio(
                    "Video Source",
                    ["Local File Path (10GB+)", "Browser Upload"],
                    index=0,
                    horizontal=True,
                    key="portal_new_ingest_mode",
                    help="Use local file path for large videos already on this machine. Browser upload is only practical for smaller files.",
                )
                if ingest_mode == "Local File Path (10GB+)":
                    path_col, browse_col = st.columns([5, 1])
                    with browse_col:
                        st.caption("")
                        if st.button(
                            "Browse",
                            key="portal_new_local_video_browse",
                            help="Open a native file picker on this machine and register the selected file path without uploading it.",
                            use_container_width=True,
                        ):
                            picked = _choose_local_video_file(str(st.session_state.get("portal_new_local_video_path", "")))
                            if picked.get("path"):
                                st.session_state.portal_new_local_video_path = picked["path"]
                                st.session_state.portal_new_local_video_pick_error = ""
                                safe_rerun()
                            else:
                                st.session_state.portal_new_local_video_pick_error = picked.get("error") or "No file selected."
                    with path_col:
                        local_video_path = st.text_input(
                            "Local Video Path",
                            value=str(st.session_state.get("portal_new_local_video_path", "")),
                            placeholder=r"C:\Videos\full-match.mp4",
                            key="portal_new_local_video_path",
                            help="The API/worker must be able to read this path. This avoids copying huge video files through the browser.",
                        )
                    picker_error = str(st.session_state.get("portal_new_local_video_pick_error", "") or "").strip()
                    if picker_error:
                        st.caption(picker_error)
                    if local_video_path.strip():
                        local_video_probe = inspect_local_video(api_base, tenant_id, token, local_video_path)
                    else:
                        local_video_probe = _local_video_probe(local_video_path)
                    if local_video_probe.get("ok"):
                        media_summary = _format_media_probe_summary(local_video_probe)
                        suffix = f" | {media_summary}" if media_summary else ""
                        st.success(
                            f"Local source ready: {_format_file_size(local_video_probe.get('size_bytes'))}{suffix}"
                        )
                        st.caption(f"Worker path: `{local_video_probe.get('path')}`")
                    elif local_video_path.strip():
                        status_message = str(local_video_probe.get("message") or "Local source is not ready.")
                        if local_video_probe.get("code") == "zero_bytes":
                            st.info(status_message)
                        else:
                            st.warning(status_message)
                        st.caption(f"Selected path: `{local_video_probe.get('path') or local_video_path}`")
                else:
                    st.warning("Browser upload can be unreliable for very large video. Prefer Local File Path for 10GB recordings.")
                    upload = st.file_uploader(
                        "Upload Game Video",
                        type=["mp4", "mov", "mkv", "avi", "m4v"],
                        key="portal_new_upload",
                    )
                    if upload is not None:
                        st.caption(f"Selected upload: `{upload.name}` | {_format_file_size(getattr(upload, 'size', 0))}")

                trim_window = _render_trim_window_controls(
                    context_key="portal_new",
                    default_enabled=True,
                    default_duration_minutes=2.0,
                )

            with analysis_tab:
                a_col1, a_col2, a_col3 = st.columns([1.1, 1.0, 1.0])
                profile_name = a_col1.selectbox(
                    "Processing Profile",
                    ["Balanced", "Offense Focus", "Set Piece Focus", "Discipline Review", "Custom"],
                    key="portal_new_profile_name",
                )
                model_version = a_col2.selectbox("Model Version", model_versions, index=0, key="portal_new_model_version")
                analysis_only = a_col3.checkbox(
                    "Analysis Only",
                    value=True,
                    key="portal_new_analysis_only",
                    help="Generate bookmark table/events quickly without writing highlight video clips.",
                )
                selected_targets = st.multiselect(
                    "Event Targets",
                    EVENT_TARGET_OPTIONS,
                    default=[],
                    key="portal_new_targets",
                    help="Leave empty for broad highlight generation.",
                )
                custom_targets = st.text_input(
                    "Additional Custom Targets",
                    value="",
                    key="portal_new_custom_targets",
                )

            with output_tab:
                out_col1, out_col2, out_col3 = st.columns(3)
                camera_mode = out_col1.selectbox(
                    "Camera Mode",
                    ["wide", "follow_action", "follow_player"],
                    index=0,
                    key="portal_new_camera_mode",
                    help="`follow_action` keeps the tracked player central while nudging toward nearby ball action.",
                )
                zoom_factor = out_col2.slider("Zoom Factor", 1.0, 3.0, 1.6, 0.1, key="portal_new_zoom_factor")
                require_gpu = out_col3.checkbox(
                    "Require GPU",
                    value=bool(st.session_state.get("portal_gpu_ready", False)),
                    key="portal_new_require_gpu",
                    help="When enabled, the run fails early unless PyTorch CUDA is available.",
                )
                output_col1, output_col2 = st.columns(2)
                no_audio = output_col1.checkbox("Disable Audio Detection", value=False, key="portal_new_no_audio")
                overlay = output_col2.checkbox("Generate Spotlight Overlay", value=False, key="portal_new_overlay")

            with advanced_tab:
                gpu_settings = _render_gpu_analysis_controls(
                    context_key="portal_new",
                    config=gpu_settings,
                    expanded=True,
                )

                with st.expander("Logging", expanded=True):
                    selected_log_profile = st.selectbox(
                        "Logging Profile",
                        LOG_PROFILE_OPTIONS,
                        index=1,
                        key="portal_new_log_profile",
                        help="Standard keeps core status. Detailed adds process notes. Diagnostic captures raw config and deeper technical checkpoints.",
                    )
                    log_profile = selected_log_profile.lower()
                    st.caption("Use Detailed while testing. Switch to Standard when routine runs are stable.")

                with st.expander("Timing and Detection", expanded=(profile_name == "Custom")):
                    cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
                    pre_seconds = cfg_col1.slider("Pre Buffer", 0.5, 10.0, 2.0, 0.5, key="portal_new_pre")
                    post_seconds = cfg_col2.slider("Post Buffer", 1.0, 20.0, 6.0, 0.5, key="portal_new_post")
                    min_clip = cfg_col3.slider("Min Clip", 1.0, 15.0, 4.0, 0.5, key="portal_new_min_clip")

                    cfg_col4, cfg_col5, cfg_col6 = st.columns(3)
                    speed_sens = cfg_col4.slider("Speed Sensitivity", 1.0, 4.0, 2.0, 0.1, key="portal_new_speed_sens")
                    audio_sens = cfg_col5.slider("Audio Sensitivity", 1.0, 4.0, 2.0, 0.1, key="portal_new_audio_sens")
                    thread_count = cfg_col6.number_input("Threads (0=auto)", min_value=0, max_value=32, value=0, key="portal_new_threads")

                with st.expander("Player Lock", expanded=False):
                    player_roi_payload = _render_player_roi_selector(
                        context_key="portal_new",
                        default_enabled=False,
                        existing_roi=None,
                        source_video_path=(str(local_video_probe.get("path") or "") if local_video_probe.get("ok") else ""),
                        upload_file=(upload if ingest_mode == "Browser Upload" and upload is not None else None),
                        upload_name=(upload.name if upload is not None else ""),
                    )

            source_status = "Ready" if (
                (ingest_mode == "Local File Path (10GB+)" and local_video_probe.get("ok"))
                or (ingest_mode == "Browser Upload" and upload is not None)
            ) else "Needed"
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            summary_col1.metric("Source", source_status)
            summary_col2.metric("Window", trim_window["label"])
            summary_col3.metric("Mode", "Analysis" if analysis_only else "Clips")
            summary_col4.metric("GPU", "Required" if require_gpu else "Optional")

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
            if ingest_mode == "Local File Path (10GB+)" and not local_video_probe.get("ok"):
                st.error(str(local_video_probe.get("message") or "Choose a readable non-empty local video file."))
                st.session_state.portal_launch_cooldown_until = 0.0
            elif ingest_mode == "Browser Upload" and upload is None:
                st.error("Upload a game video first.")
                st.session_state.portal_launch_cooldown_until = 0.0
            elif ingest_mode == "Browser Upload" and int(getattr(upload, "size", 0) or 0) <= 0:
                st.error("The selected video file is empty. Choose a real MP4/MOV/MKV/AVI file and try again.")
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
                        "ingest_mode": ingest_mode,
                        "processing_window": trim_window["label"],
                    },
                }
                match_result = api_request("POST", api_base, "/matches", tenant_id, token, json_body=match_payload)
                if not match_result["ok"]:
                    st.error(f"Failed to create match: {match_result['payload']}")
                    st.session_state.portal_launch_cooldown_until = 0.0
                else:
                    match_id = match_result["payload"]["match_id"]
                    if ingest_mode == "Local File Path (10GB+)":
                        ingest_result = api_request(
                            "POST",
                            api_base,
                            f"/matches/{match_id}/assets/register-local",
                            tenant_id,
                            token,
                            json_body={"path": str(local_video_probe.get("path") or ""), "set_as_source": True},
                            timeout=60,
                        )
                    else:
                        files = {
                            "file": (
                                upload.name,
                                upload.getvalue(),
                                upload.type or "application/octet-stream",
                            )
                        }
                        ingest_result = api_request(
                            "POST",
                            api_base,
                            f"/matches/{match_id}/assets/upload",
                            tenant_id,
                            token,
                            files=files,
                            timeout=900,
                        )
                    if not ingest_result["ok"]:
                        st.error(f"Match created, but video ingest failed: {ingest_result['payload']}")
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
                            **gpu_settings,
                            "overlay": bool(overlay),
                            "no_audio": bool(no_audio),
                            "require_gpu": bool(require_gpu),
                            "analysis_only": bool(analysis_only),
                            "run_created_at": datetime.utcnow().isoformat(),
                            "run_created_from": "portal_ui",
                            "ingest_mode": ingest_mode,
                            "test_window_enabled": bool(trim_window["enabled"]),
                            "log_profile": log_profile,
                        }
                        if trim_window["enabled"]:
                            job_config["trim_start"] = float(trim_window["trim_start"])
                            job_config["trim_end"] = float(trim_window["trim_end"])
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
                            st.session_state.portal_pending_match_select_id = match_id
                            st.session_state.portal_pending_job_select_id = job_id
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
                source_log_profile = str(source_config.get("log_profile") or "detailed").strip().lower()
                if source_log_profile not in {"standard", "detailed", "diagnostic"}:
                    source_log_profile = "detailed"
                rerun_log_profile = st.selectbox(
                    "Logging Profile",
                    LOG_PROFILE_OPTIONS,
                    index=LOG_PROFILE_OPTIONS.index(source_log_profile.title()),
                    key="portal_rerun_log_profile",
                    help="Detailed logs are useful while validating reruns. Diagnostic captures deeper raw config checkpoints.",
                ).lower()
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
                    3.0,
                    _bounded_float(source_config.get("zoom_factor", 1.6), 1.6, 1.0, 3.0),
                    0.1,
                    key="portal_rerun_zoom_factor",
                )
                rerun_gpu_settings = _render_gpu_analysis_controls(
                    context_key=f"portal_rerun_{source_job['job_id']}",
                    config=source_config,
                    expanded=True,
                )
                rerun_trim_window = _render_trim_window_controls(
                    context_key=f"portal_rerun_{source_job['job_id']}",
                    config=source_config,
                    default_enabled=False,
                    default_duration_minutes=2.0,
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
                    overrides.update(rerun_gpu_settings)
                    overrides["log_profile"] = rerun_log_profile
                    overrides["test_window_enabled"] = bool(rerun_trim_window["enabled"])
                    overrides["trim_start"] = (
                        float(rerun_trim_window["trim_start"]) if rerun_trim_window["enabled"] else None
                    )
                    overrides["trim_end"] = (
                        float(rerun_trim_window["trim_end"]) if rerun_trim_window["enabled"] else None
                    )

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
                        st.session_state.portal_pending_match_select_id = match_id
                        st.session_state.portal_pending_job_select_id = rerun_job_id
                        st.session_state.portal_auto_fetch_job_id = rerun_job_id
                        st.session_state.portal_pending_nav = monitor_nav_label
                        st.session_state.portal_flash_message = f"Rerun queued: {rerun_job_id}"
                        safe_rerun()


elif nav_key == "Training Lab":
    st.subheader("Training Lab")
    st.caption("Fine-tune a real Ultralytics detector from a YOLO dataset YAML, then use the resulting `best.pt` in GPU Analysis.")

    train_tab, models_tab = st.tabs(["YOLO26 Detector", "Promoted Models"])
    with train_tab:
        form_col, status_col = st.columns([1.1, 1.0])
        with form_col:
            st.markdown("#### Train Detector")
            dataset_yaml = st.text_input(
                "Dataset YAML",
                value=str(st.session_state.get("portal_train_dataset_yaml", "")),
                placeholder=r"C:\datasets\soccer-players\data.yaml",
                key="portal_train_dataset_yaml",
                help="Ultralytics YOLO dataset YAML with train/val image paths and class names.",
            )
            base_model = st.selectbox(
                "Base Model",
                ["yolo26s.pt", "yolo26n.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt", "yolo11s.pt"],
                index=0,
                key="portal_train_base_model",
            )
            run_name = st.text_input(
                "Run Name",
                value="soccer-detector-yolo26",
                key="portal_train_run_name",
            )
            tr_col1, tr_col2, tr_col3 = st.columns(3)
            train_epochs = int(tr_col1.number_input("Epochs", min_value=1, max_value=1000, value=50, step=1, key="portal_train_epochs"))
            train_imgsz = int(
                tr_col2.select_slider(
                    "Image Size",
                    options=INFERENCE_IMAGE_SIZE_OPTIONS + [1536],
                    value=960,
                    key="portal_train_imgsz",
                )
            )
            train_batch = int(tr_col3.number_input("Batch", min_value=-1, max_value=512, value=8, step=1, key="portal_train_batch"))
            tr_col4, tr_col5, tr_col6 = st.columns(3)
            train_device = tr_col4.text_input("Device", value="0" if st.session_state.portal_gpu_ready else "cpu", key="portal_train_device")
            train_workers = int(tr_col5.number_input("Workers", min_value=0, max_value=64, value=4, step=1, key="portal_train_workers"))
            train_patience = int(tr_col6.number_input("Patience", min_value=0, max_value=300, value=25, step=1, key="portal_train_patience"))
            train_notes = st.text_area(
                "Notes",
                value="Fine-tune detector for soccer field footage, players, ball, referees, and wide camera views.",
                key="portal_train_notes",
            )

            start_disabled = not bool(dataset_yaml.strip())
            if st.button("Start YOLO26 Training", type="primary", use_container_width=True, disabled=start_disabled):
                payload = {
                    "target_model": "yolo-detector",
                    "notes": train_notes,
                    "training_config": {
                        "kind": "ultralytics_yolo",
                        "dataset_yaml": dataset_yaml.strip(),
                        "base_model": str(base_model),
                        "run_name": run_name,
                        "epochs": train_epochs,
                        "imgsz": train_imgsz,
                        "batch": train_batch,
                        "device": train_device.strip(),
                        "workers": train_workers,
                        "patience": train_patience,
                    },
                }
                train_result = create_training_run(api_base, tenant_id, token, payload)
                if not train_result["ok"]:
                    st.error(f"Training failed to start: {train_result['payload']}")
                else:
                    run_id = str(train_result["payload"].get("run_id") or "")
                    st.session_state.portal_training_run_id = run_id
                    st.session_state.portal_flash_message = f"YOLO training queued: {run_id}"
                    safe_rerun()

        with status_col:
            st.markdown("#### Training Status")
            run_id = st.text_input(
                "Training Run ID",
                value=str(st.session_state.get("portal_training_run_id", "")),
                key="portal_training_run_lookup",
            )
            run_payload = get_training_run(api_base, tenant_id, token, run_id.strip()) if run_id.strip() else {}
            if run_payload:
                st.metric("Status", str(run_payload.get("status") or "-"))
                st.metric("Gates", "passed" if run_payload.get("gates_passed") else "not passed")
                candidate_path = str(run_payload.get("candidate_model_version") or "").strip()
                if candidate_path:
                    st.text_input("Trained Weights Path", value=candidate_path, key="portal_training_candidate_path")
                    st.caption("Use this path in GPU Analysis -> Custom Detector Weights.")
                metrics = dict(run_payload.get("metrics") or {})
                if metrics:
                    st.json(metrics)
                promote_disabled = str(run_payload.get("status")) != "completed" or not candidate_path
                if st.button("Promote Trained Detector", use_container_width=True, disabled=promote_disabled):
                    promote_result = api_request(
                        "POST",
                        api_base,
                        f"/training/runs/{run_id.strip()}/promote",
                        tenant_id,
                        token,
                        json_body={
                            "decision": "approved",
                            "reason": "approved YOLO detector for processing",
                            "notes": "Promoted from Training Lab.",
                            "force": True,
                        },
                    )
                    if promote_result["ok"]:
                        st.success("Detector promoted. The weights path remains available for GPU Analysis custom detector use.")
                    else:
                        st.error(f"Promotion failed: {promote_result['payload']}")
            else:
                st.info("Start a YOLO training run, then keep this page open or paste the run id here.")

    with models_tab:
        models = list_training_models(api_base, tenant_id, token)
        if not models:
            st.info("No promoted models yet.")
        else:
            st.dataframe(
                [
                    {
                        "target_model": item.get("target_model"),
                        "version_or_path": item.get("version"),
                        "promoted": item.get("promoted"),
                        "created_at": _iso_to_short(item.get("created_at")),
                    }
                    for item in models
                ],
                use_container_width=True,
                hide_index=True,
            )

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
                    "window": _format_trim_window((latest or {}).get("config", {}) or {}),
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

        selector = {_library_match_selector_label(row): row["match_id"] for row in (filtered_rows or rows)}
        selector_labels = list(selector.keys())
        default_match = _extract_entity_id_from_ref(str(st.session_state.selected_match_id or ""), "match")
        pending_library_match = _extract_entity_id_from_ref(
            str(st.session_state.get("portal_pending_match_select_id", "") or ""),
            "match",
        )
        desired_library_label, selected_index = _option_label_for_entity_id(
            selector,
            pending_library_match or default_match,
        )
        current_library_label = str(st.session_state.get("portal_library_select", "") or "")
        if desired_library_label and (pending_library_match or current_library_label not in selector_labels):
            st.session_state.portal_library_select = desired_library_label
            if pending_library_match:
                st.session_state.portal_pending_match_select_id = ""
        elif current_library_label and current_library_label not in selector_labels and selector_labels:
            st.session_state.portal_library_select = selector_labels[selected_index]
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
                        st.write(f"Window: `{_format_trim_window(latest_config)}`")
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

                profile_name = custom_profile_default
                model_version = custom_model_default if custom_model_default in model_versions else (model_versions[0] if model_versions else "event-v0")
                analysis_only = bool(latest_config.get("analysis_only", False))
                targets: List[str] = [t for t in custom_targets_default if t in EVENT_TARGET_OPTIONS]
                custom_targets = ""
                pre_seconds = float(latest_config.get("pre_seconds", 2.0))
                post_seconds = float(latest_config.get("post_seconds", 6.0))
                min_clip = float(latest_config.get("min_clip_duration", 4.0))
                speed_sens = float(latest_config.get("speed_sensitivity", 2.0))
                audio_sens = float(latest_config.get("audio_sensitivity", 2.0))
                threads = int(latest_config.get("threads", 0) or 0)
                trim_window = {
                    "enabled": False,
                    "trim_start": None,
                    "trim_end": None,
                    "label": _format_trim_window(latest_config),
                }
                camera_mode_options = ["wide", "follow_action", "follow_player"]
                current_camera_mode = str(latest_config.get("camera_mode") or "wide")
                if current_camera_mode not in camera_mode_options:
                    current_camera_mode = "wide"
                camera_mode = current_camera_mode
                zoom_factor = float(latest_config.get("zoom_factor", 1.6))
                gpu_settings = {
                    "yolo_model": str(latest_config.get("yolo_model") or "yolo26s.pt"),
                    "tracker_config": str(latest_config.get("tracker_config") or "botsort.yaml"),
                    "inference_imgsz": int(latest_config.get("inference_imgsz", 960) or 960),
                    "detection_conf": float(latest_config.get("detection_conf", 0.18) or 0.18),
                    "vid_stride": int(latest_config.get("vid_stride", 1) or 1),
                }
                no_audio = bool(latest_config.get("no_audio", False))
                overlay = bool(latest_config.get("overlay", False))
                require_gpu = bool(latest_config.get("require_gpu", False))
                current_log_profile = str(latest_config.get("log_profile") or "detailed").strip().lower()
                if current_log_profile not in {"standard", "detailed", "diagnostic"}:
                    current_log_profile = "detailed"
                log_profile = current_log_profile
                player_roi_payload: Dict[str, Any] = {"enabled": bool(latest_config.get("player_roi")), "roi": latest_config.get("player_roi")}

                custom_analysis_tab, custom_output_tab, custom_advanced_tab = st.tabs(["Analysis", "Output", "Advanced"])

                with custom_analysis_tab:
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
                        "Additional Custom Targets",
                        value="",
                        key=f"portal_match_custom_targets_extra_{selected_match_id}",
                    )
                    trim_window = _render_trim_window_controls(
                        context_key=f"portal_match_custom_{selected_match_id}",
                        config=latest_config,
                        default_enabled=False,
                        default_duration_minutes=2.0,
                    )

                with custom_output_tab:
                    cam_col1, cam_col2, opt_col1, opt_col2, opt_col3 = st.columns(5)
                    camera_mode = cam_col1.selectbox(
                        "Camera Mode",
                        camera_mode_options,
                        index=camera_mode_options.index(current_camera_mode),
                        key=f"portal_match_custom_camera_mode_{selected_match_id}",
                    )
                    zoom_factor = cam_col2.slider(
                        "Zoom Factor",
                        1.0,
                        3.0,
                        _bounded_float(latest_config.get("zoom_factor", 1.6), 1.6, 1.0, 3.0),
                        0.1,
                        key=f"portal_match_custom_zoom_factor_{selected_match_id}",
                    )
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

                with custom_advanced_tab:
                    gpu_settings = _render_gpu_analysis_controls(
                        context_key=f"portal_match_custom_{selected_match_id}",
                        config=gpu_settings,
                        expanded=True,
                    )

                    with st.expander("Logging", expanded=True):
                        selected_log_profile = st.selectbox(
                            "Logging Profile",
                            LOG_PROFILE_OPTIONS,
                            index=LOG_PROFILE_OPTIONS.index(current_log_profile.title()),
                            key=f"portal_match_custom_log_profile_{selected_match_id}",
                            help="Standard keeps core status. Detailed adds process notes. Diagnostic captures raw config and deeper technical checkpoints.",
                        )
                        log_profile = selected_log_profile.lower()
                        st.caption("Use Detailed or Diagnostic for short test windows; use Standard for routine full-match runs.")

                    with st.expander("Timing and Detection", expanded=(profile_name == "Custom")):
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

                    with st.expander("Player Lock", expanded=False):
                        player_roi_payload = _render_player_roi_selector(
                            context_key=f"portal_match_custom_{selected_match_id}",
                            default_enabled=bool(latest_config.get("player_roi")),
                            existing_roi=(latest_config.get("player_roi") if isinstance(latest_config.get("player_roi"), dict) else None),
                            source_video_path=source_video or "",
                        )

                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                summary_col1.metric("Window", trim_window["label"])
                summary_col2.metric("Mode", "Analysis" if analysis_only else "Clips")
                summary_col3.metric("Camera", str(camera_mode))
                summary_col4.metric("GPU", "Required" if require_gpu else "Optional")

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
                            **gpu_settings,
                            "overlay": bool(overlay),
                            "no_audio": bool(no_audio),
                            "require_gpu": bool(require_gpu),
                            "analysis_only": bool(analysis_only),
                            "run_created_at": datetime.utcnow().isoformat(),
                            "run_created_from": "portal_match_workspace_custom",
                            "test_window_enabled": bool(trim_window["enabled"]),
                            "log_profile": log_profile,
                        }
                        if trim_window["enabled"]:
                            next_config["trim_start"] = float(trim_window["trim_start"])
                            next_config["trim_end"] = float(trim_window["trim_end"])
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
                    _render_job_outcome_notice(job)
                    st.write(f"Model: `{job.get('config', {}).get('model_version', '-')}`")
                    st.write(f"Camera: `{job.get('config', {}).get('camera_mode', 'wide')}`")
                    st.write(f"Window: `{_format_trim_window(job.get('config', {}) or {})}`")
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

    with st.expander("Queue Controls", expanded=is_technical):
        default_scope_match = str(st.session_state.selected_match_id or "").strip()
        if default_scope_match and not str(st.session_state.get("portal_ops_match_scope", "")).strip():
            st.session_state.portal_ops_match_scope = default_scope_match
        match_scope = st.text_input("Match ID Scope", value=default_scope_match, key="portal_ops_match_scope")
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
                "Kill All Queued",
                key="portal_ops_kill_all_queued",
                disabled=not queued_jobs,
            )
            kill_all_active = queue_col2.button(
                "Kill All Active",
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
            st.caption("Provide a Match ID to bulk-kill queued or active jobs.")

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
            diagnostics = get_job_diagnostics(api_base, tenant_id, token, job_id_value)
            _render_realtime_job_timeline(payload, diagnostics, gpu_health)
            _render_job_diagnostics_panel(diagnostics, is_technical=is_technical)
            st.caption(f"Cancel requested: `{bool(payload.get('cancel_requested', False))}`")
            if is_technical:
                st.write("Config")
                st.json(payload.get("config", {}))
                st.write("Result")
                st.json(payload.get("result", {}))
            else:
                st.write(f"Model: `{payload.get('config', {}).get('model_version', '-')}`")
                st.write(f"Camera: `{payload.get('config', {}).get('camera_mode', 'wide')}`")
                st.write(f"Window: `{_format_trim_window(payload.get('config', {}) or {})}`")
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
        log_col1, log_col2, log_col3, log_col4, log_col5 = st.columns(5)
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
        with log_col5:
            log_view = st.selectbox(
                "View",
                ["Process Story", "Technical Table", "Raw Rows"],
                index=0,
                key="portal_ops_log_view",
            )

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
                    if log_view == "Process Story":
                        rows = []
                        for item in items:
                            if not isinstance(item, dict):
                                continue
                            data = item.get("data") if isinstance(item.get("data"), dict) else {}
                            rows.append(
                                {
                                    "time": _iso_to_short(item.get("created_at")),
                                    "level": item.get("level"),
                                    "stage": item.get("stage"),
                                    "process": data.get("process_message") or item.get("message"),
                                    "technical": data.get("technical_message") or "",
                                }
                            )
                        st.dataframe(rows, use_container_width=True, hide_index=True)
                    elif log_view == "Technical Table":
                        rows = []
                        for item in items:
                            if not isinstance(item, dict):
                                continue
                            data = item.get("data") if isinstance(item.get("data"), dict) else {}
                            rows.append(
                                {
                                    "time": _iso_to_short(item.get("created_at")),
                                    "level": item.get("level"),
                                    "detail": item.get("detail_level"),
                                    "stage": item.get("stage"),
                                    "message": item.get("message"),
                                    "technical": data.get("technical_message") or "",
                                    "log_profile": data.get("log_profile") or "",
                                }
                            )
                        st.dataframe(rows, use_container_width=True, hide_index=True)
                    else:
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
