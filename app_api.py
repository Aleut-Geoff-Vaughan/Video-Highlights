"""
Video Highlights V1 API Client (Streamlit)

Run:
    streamlit run app_api.py
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

import requests
import streamlit as st


st.set_page_config(page_title="Video Highlights API Client", page_icon="⚽", layout="wide")


def api_request(
    method: str,
    base_url: str,
    path: str,
    token: Optional[str] = None,
    tenant_id: Optional[str] = None,
    json_body: Optional[Dict[str, Any]] = None,
    files: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
) -> requests.Response:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if tenant_id:
        headers["X-Tenant-Id"] = tenant_id
    url = f"{base_url.rstrip('/')}{path}"
    return requests.request(
        method=method.upper(),
        url=url,
        headers=headers,
        json=json_body,
        files=files,
        timeout=timeout,
    )


def parse_response(response: requests.Response) -> Dict[str, Any]:
    try:
        payload = response.json()
    except Exception:
        payload = {"raw": response.text}
    return {
        "status_code": response.status_code,
        "payload": payload,
        "ok": response.ok,
    }


st.title("⚽ Video Highlights V1 API Client")

with st.sidebar:
    st.header("Connection")
    api_base = st.text_input("API Base URL", value="http://localhost:8000/v1")
    token = st.text_input("Bearer Token (optional)", key="api_token", value="", type="password")
    tenant_id = st.text_input("Tenant ID/Slug (optional)", key="tenant_id", value="sandbox")
    bootstrap_key = st.text_input("Bootstrap Key (optional)", key="bootstrap_key", value="", type="password")
    auto_refresh = st.checkbox("Auto refresh job status", value=False)

tabs = st.tabs(
    [
        "Health",
        "Auth",
        "Match",
        "Job",
        "Events",
        "Feedback",
        "Training",
        "Agent",
    ]
)

if "match_id" not in st.session_state:
    st.session_state.match_id = ""
if "job_id" not in st.session_state:
    st.session_state.job_id = ""
if "event_id" not in st.session_state:
    st.session_state.event_id = ""
if "feedback_id" not in st.session_state:
    st.session_state.feedback_id = ""
if "run_id" not in st.session_state:
    st.session_state.run_id = ""


with tabs[0]:
    if st.button("Check /health"):
        r = api_request("GET", api_base, "/health", token=token or None, tenant_id=tenant_id or None, timeout=30)
        st.json(parse_response(r))


with tabs[1]:
    st.subheader("Auth Identity")
    if st.button("Get /auth/me"):
        r = api_request("GET", api_base, "/auth/me", token=token or None, tenant_id=tenant_id or None)
        st.json(parse_response(r))

    st.subheader("Issue JWT")
    token_user_id = st.text_input("Token User ID", value="coach_1")
    token_role = st.selectbox("Token Role", ["admin", "tenant_admin", "analyst", "coach", "parent", "system"])
    token_ttl = st.number_input("Expires In Minutes", min_value=1, max_value=10080, value=120)
    if st.button("Issue Token"):
        headers_override = {}
        if bootstrap_key.strip():
            headers_override["X-Bootstrap-Key"] = bootstrap_key.strip()
        response = requests.post(
            f"{api_base.rstrip('/')}/auth/token",
            headers={
                **({"Authorization": f"Bearer {token}"} if token else {}),
                **({"X-Tenant-Id": tenant_id} if tenant_id else {}),
                **headers_override,
            },
            json={
                "user_id": token_user_id,
                "role": token_role,
                "tenant_id": tenant_id or None,
                "expires_in_minutes": int(token_ttl),
            },
            timeout=60,
        )
        result = parse_response(response)
        st.json(result)
        if result["ok"] and result["payload"].get("access_token"):
            st.session_state.api_token = result["payload"]["access_token"]


with tabs[2]:
    st.subheader("Create Match")
    col_a, col_b = st.columns(2)
    with col_a:
        match_name = st.text_input("Match Name", value="Demo Match")
        source_video_path = st.text_input("Source Video Path", value="C:/tmp/nonexistent.mp4")
        home_team = st.text_input("Home Team", value="Home")
        away_team = st.text_input("Away Team", value="Away")
    with col_b:
        match_date = st.text_input("Match Date", value="2026-02-20")
        metadata_text = st.text_area("Metadata JSON", value='{"competition":"U12"}', height=120)

    if st.button("Create Match"):
        try:
            metadata = json.loads(metadata_text) if metadata_text.strip() else {}
        except json.JSONDecodeError:
            st.error("Metadata JSON is invalid.")
            metadata = None
        if metadata is not None:
            r = api_request(
                "POST",
                api_base,
                "/matches",
                token=token or None,
                tenant_id=tenant_id or None,
                json_body={
                    "name": match_name,
                    "home_team_name": home_team,
                    "away_team_name": away_team,
                    "match_date": match_date,
                    "source_video_path": source_video_path,
                    "metadata": metadata,
                },
            )
            result = parse_response(r)
            st.json(result)
            if result["ok"]:
                st.session_state.match_id = result["payload"]["match_id"]

    st.subheader("Upload Asset to Match")
    st.text_input("Match ID", key="match_id")
    uploaded_file = st.file_uploader("Upload Video Asset", type=["mp4", "mov", "mkv", "avi", "m4v"])
    if st.button("Upload Asset", disabled=not st.session_state.match_id or uploaded_file is None):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")}
        r = api_request(
            "POST",
            api_base,
            f"/matches/{st.session_state.match_id}/assets/upload",
            token=token or None,
            tenant_id=tenant_id or None,
            files=files,
            timeout=600,
        )
        st.json(parse_response(r))


with tabs[3]:
    st.subheader("Create Job")
    st.text_input("Match ID", key="match_id_2", value=st.session_state.get("match_id", ""))
    job_config_text = st.text_area(
        "Job Config JSON",
        value='{"pre_seconds":2.0,"post_seconds":6.0,"min_clip_duration":4.0,"camera_mode":"wide","zoom_factor":1.6,"overlay":false}',
        height=140,
    )
    if st.button("Create Processing Job", disabled=not st.session_state.match_id):
        try:
            job_config = json.loads(job_config_text) if job_config_text.strip() else {}
        except json.JSONDecodeError:
            st.error("Job config JSON is invalid.")
            job_config = None
        if job_config is not None:
            r = api_request(
                "POST",
                api_base,
                f"/matches/{st.session_state.match_id}/jobs",
                token=token or None,
                tenant_id=tenant_id or None,
                json_body={"config": job_config},
            )
            result = parse_response(r)
            st.json(result)
            if result["ok"]:
                st.session_state.job_id = result["payload"]["job_id"]

    st.subheader("Get Job")
    st.text_input("Job ID", key="job_id")
    if st.button("Fetch Job", disabled=not st.session_state.job_id):
        r = api_request(
            "GET",
            api_base,
            f"/jobs/{st.session_state.job_id}",
            token=token or None,
            tenant_id=tenant_id or None,
        )
        result = parse_response(r)
        st.json(result)
        if auto_refresh and result["ok"] and result["payload"].get("status") in {"queued", "claimed", "running"}:
            time.sleep(2)
            rerun_fn = getattr(st, "rerun", None)
            if callable(rerun_fn):
                rerun_fn()
            else:
                legacy_rerun = getattr(st, "experimental_rerun", None)
                if callable(legacy_rerun):
                    legacy_rerun()

    if st.button("Retry Job", disabled=not st.session_state.job_id):
        r = api_request(
            "POST",
            api_base,
            f"/jobs/{st.session_state.job_id}/retry",
            token=token or None,
            tenant_id=tenant_id or None,
            json_body={},
        )
        result = parse_response(r)
        st.json(result)
        if result["ok"]:
            st.session_state.job_id = result["payload"]["job_id"]

    if st.button("Cancel Job", disabled=not st.session_state.job_id):
        r = api_request(
            "POST",
            api_base,
            f"/jobs/{st.session_state.job_id}/cancel",
            token=token or None,
            tenant_id=tenant_id or None,
            json_body={},
        )
        st.json(parse_response(r))

    if st.button("Kill Job Session", disabled=not st.session_state.job_id):
        r = api_request(
            "POST",
            api_base,
            f"/jobs/{st.session_state.job_id}/kill-session",
            token=token or None,
            tenant_id=tenant_id or None,
            json_body={},
        )
        st.json(parse_response(r))

    if st.button("Worker Run Once"):
        r = api_request(
            "POST",
            api_base,
            "/jobs/worker/run-once",
            token=token or None,
            tenant_id=tenant_id or None,
            json_body={},
        )
        st.json(parse_response(r))

    st.subheader("Job Logs")
    log_col1, log_col2, log_col3, log_col4 = st.columns(4)
    with log_col1:
        log_level_filter = st.selectbox(
            "Level Filter",
            ["all", "debug", "info", "warning", "error"],
            index=0,
            key="job_log_level_filter",
        )
    with log_col2:
        log_detail_filter = st.selectbox(
            "Detail Filter",
            ["all", "basic", "detailed", "extreme"],
            index=0,
            key="job_log_detail_filter",
        )
    with log_col3:
        log_stage_filter = st.text_input("Stage Filter", value="", key="job_log_stage_filter")
    with log_col4:
        log_limit = st.number_input("Log Limit", min_value=10, max_value=5000, value=200, key="job_log_limit")

    if st.button("Fetch Job Logs", disabled=not st.session_state.job_id, key="fetch_job_logs_btn"):
        params = [f"limit={int(log_limit)}"]
        if log_level_filter != "all":
            params.append(f"level={log_level_filter}")
        if log_detail_filter != "all":
            params.append(f"detail_level={log_detail_filter}")
        if log_stage_filter.strip():
            params.append(f"stage={log_stage_filter.strip()}")
        suffix = f"?{'&'.join(params)}" if params else ""
        r = api_request(
            "GET",
            api_base,
            f"/jobs/{st.session_state.job_id}/logs{suffix}",
            token=token or None,
            tenant_id=tenant_id or None,
        )
        result = parse_response(r)
        st.json(result)


with tabs[4]:
    st.subheader("List Events")
    st.text_input("Match ID", key="match_id_3", value=st.session_state.get("match_id", ""))
    col1, col2 = st.columns(2)
    with col1:
        filter_event_type = st.text_input("Event Type Filter", value="")
        min_conf = st.text_input("Min Confidence", value="")
    with col2:
        from_ms = st.text_input("From ms", value="")
        to_ms = st.text_input("To ms", value="")

    if st.button("Fetch Events", disabled=not st.session_state.match_id):
        params = []
        if filter_event_type.strip():
            params.append(f"event_type={filter_event_type.strip()}")
        if min_conf.strip():
            params.append(f"min_confidence={min_conf.strip()}")
        if from_ms.strip():
            params.append(f"from_ms={from_ms.strip()}")
        if to_ms.strip():
            params.append(f"to_ms={to_ms.strip()}")
        suffix = f"?{'&'.join(params)}" if params else ""
        r = api_request(
            "GET",
            api_base,
            f"/matches/{st.session_state.match_id}/events{suffix}",
            token=token or None,
            tenant_id=tenant_id or None,
        )
        result = parse_response(r)
        st.json(result)
        if result["ok"] and result["payload"].get("items"):
            first = result["payload"]["items"][0]
            st.session_state.event_id = first.get("event_id", "")

    st.subheader("Event Clip On Demand")
    st.text_input("Event ID  ", key="event_id")
    clip_col1, clip_col2 = st.columns(2)
    with clip_col1:
        clip_pre = st.number_input("Pre Seconds", min_value=0.0, max_value=120.0, value=1.5, step=0.5)
        clip_anchor = st.selectbox("Anchor", ["event_window", "occurred_at"], index=0)
    with clip_col2:
        clip_post = st.number_input("Post Seconds", min_value=0.0, max_value=300.0, value=5.0, step=0.5)
        clip_audio = st.checkbox("Include Audio", value=True)
    if st.button("Render Event Clip", disabled=not st.session_state.match_id or not st.session_state.event_id):
        r = api_request(
            "POST",
            api_base,
            f"/matches/{st.session_state.match_id}/events/{st.session_state.event_id}/clip-on-demand",
            token=token or None,
            tenant_id=tenant_id or None,
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
        result = parse_response(r)
        st.json(result)


with tabs[5]:
    st.subheader("Submit Feedback")
    st.text_input("Match ID", key="match_id_4", value=st.session_state.get("match_id", ""))
    st.text_input("Event ID", key="event_id_2", value=st.session_state.get("event_id", ""))
    feedback_type = st.selectbox(
        "Feedback Type",
        [
            "wrong_timestamp",
            "false_positive",
            "missed_event",
            "wrong_event_type",
            "wrong_player",
            "wrong_team",
            "duplicate_event",
            "confidence_miscalibrated",
        ],
    )
    feedback_comment = st.text_area("Feedback Comment", value="Needs correction")
    correction_json = st.text_area(
        "Correction JSON",
        value='{"corrected_occurred_at_ms":1000,"corrected_start_ms":900,"corrected_end_ms":1200}',
        height=100,
    )
    if st.button("Submit Feedback", disabled=not st.session_state.match_id):
        try:
            correction = json.loads(correction_json) if correction_json.strip() else {}
        except json.JSONDecodeError:
            st.error("Correction JSON is invalid.")
            correction = None
        if correction is not None:
            body = {
                "feedback_type": feedback_type,
                "comment": feedback_comment,
                "submitted_by": {"user_id": "streamlit_user", "role": "coach"},
                "correction": correction,
                "evidence": [],
            }
            if feedback_type == "missed_event":
                path = f"/matches/{st.session_state.match_id}/feedback"
            else:
                if not st.session_state.event_id:
                    st.error("Event ID is required for non-missed feedback.")
                    st.stop()
                path = f"/matches/{st.session_state.match_id}/events/{st.session_state.event_id}/feedback"
            r = api_request("POST", api_base, path, token=token or None, tenant_id=tenant_id or None, json_body=body)
            result = parse_response(r)
            st.json(result)
            if result["ok"]:
                st.session_state.feedback_id = result["payload"]["feedback_id"]

    st.subheader("Review Feedback")
    st.text_input("Feedback ID", key="feedback_id")
    review_decision = st.selectbox("Review Decision", ["approved", "rejected", "needs_more_info", "merged"])
    if st.button("Review Feedback", disabled=not st.session_state.match_id or not st.session_state.feedback_id):
        r = api_request(
            "POST",
            api_base,
            f"/matches/{st.session_state.match_id}/feedback/{st.session_state.feedback_id}/review",
            token=token or None,
            tenant_id=tenant_id or None,
            json_body={"review_decision": review_decision, "review_note": "Reviewed in app_api"},
        )
        st.json(parse_response(r))


with tabs[6]:
    st.subheader("Training Batch + Run")
    st.text_input("Match ID", key="match_id_5", value=st.session_state.get("match_id", ""))
    if st.button("Create Feedback Batch", disabled=not st.session_state.match_id):
        r = api_request(
            "POST",
            api_base,
            "/training/feedback-batches",
            token=token or None,
            tenant_id=tenant_id or None,
            json_body={
                "match_ids": [st.session_state.match_id],
                "feedback_status": "approved",
                "feedback_types": ["wrong_timestamp", "false_positive", "missed_event"],
                "from_date": "2026-01-01",
                "to_date": "2026-12-31",
            },
        )
        result = parse_response(r)
        st.json(result)
        if result["ok"]:
            st.session_state.batch_id = result["payload"]["batch_id"]

    batch_id = st.text_input("Batch ID", value=st.session_state.get("batch_id", ""))
    if st.button("Create Training Run", disabled=not batch_id):
        r = api_request(
            "POST",
            api_base,
            "/training/runs",
            token=token or None,
            tenant_id=tenant_id or None,
            json_body={"batch_id": batch_id, "target_model": "event-v0", "notes": "Streamlit run"},
        )
        result = parse_response(r)
        st.json(result)
        if result["ok"]:
            st.session_state.run_id = result["payload"]["run_id"]

    st.text_input("Run ID", key="run_id")
    if st.button("Get Training Run", disabled=not st.session_state.run_id):
        r = api_request(
            "GET",
            api_base,
            f"/training/runs/{st.session_state.run_id}",
            token=token or None,
            tenant_id=tenant_id or None,
        )
        st.json(parse_response(r))


with tabs[7]:
    st.subheader("Agent Query")
    st.text_input("Match ID", key="match_id_6", value=st.session_state.get("match_id", ""))
    agent_query = st.text_area("Query", value="Summarize key events and likely missing high-impact moments.")
    if st.button("Run Agent Query", disabled=not st.session_state.match_id):
        r = api_request(
            "POST",
            api_base,
            f"/matches/{st.session_state.match_id}/agent/query",
            token=token or None,
            tenant_id=tenant_id or None,
            json_body={"query": agent_query, "include_event_limit": 100},
        )
        st.json(parse_response(r))

    st.subheader("Agent Explain Event")
    st.text_input("Event ID", key="event_id_3", value=st.session_state.get("event_id", ""))
    explain_q = st.text_input("Explain Question", value="Why was this event detected?")
    if st.button("Explain Event", disabled=not st.session_state.match_id or not st.session_state.event_id):
        r = api_request(
            "POST",
            api_base,
            f"/matches/{st.session_state.match_id}/agent/explain/{st.session_state.event_id}",
            token=token or None,
            tenant_id=tenant_id or None,
            json_body={"question": explain_q},
        )
        st.json(parse_response(r))
