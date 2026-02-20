# Video Highlights

Video Highlights is a soccer video analysis platform that converts full-match recordings into events, player spotlights, team analytics, and shareable highlight packages.

## Project Status

This repository now provides a local Python pipeline plus a web/API foundation with multitenant controls:

1. CLI pipeline: `VideoHighlights.py`
2. Desktop UI: `VideoHighlightsGUI.py`
3. Streamlit UI: `app.py`
4. FastAPI backend: `backend/main.py`
5. API operator UI: `app_api.py`
6. Global admin portal: `app_admin_global.py`
7. Tenant admin portal: `app_admin_tenant.py`

The planned direction is a web/cloud architecture with Dockerized workers and optional GPU acceleration.

## Target Architecture (Planned)

1. Web app for upload, review, annotation, and export
2. API for users, teams, projects, jobs, and outputs
3. Queue and scheduler for asynchronous processing
4. GPU analysis workers for tracking, event detection, and player analytics
5. Rendering workers for clips, overlays, and montages
6. Object storage, relational database, and observability stack

## Product Capability Targets

1. Follow-cam generation from panoramic match recordings
2. Automated soccer event timeline detection
3. Player spotlight reels and jersey-assisted identity workflows
4. Team momentum graph, heatmaps, and position summaries
5. Timeline editor with annotation and sharing tools
6. Live streaming and instant replay markers (phase-gated)
7. Data trust workflows (confidence calibration and human review queues)
8. Season intelligence and opponent scouting automation
9. Coaching action plans, recruiting workflows, and distribution tooling
10. Open APIs, integrations, and edge/fleet operational visibility
11. AI copilot workflows via LLM API integration for query, explainability, and review assistance
12. Feedback-driven continuous learning loop for improving event quality over time

## Quick Start (Current Local Pipeline)

### Prerequisites

- Python 3.10+
- FFmpeg in PATH
- Optional: NVIDIA GPU + CUDA drivers

### Install

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Run CLI

```bash
python VideoHighlights.py --video path/to/match.mp4 --out ./highlights_output
```

### Run Web UI (Streamlit)

```bash
streamlit run app.py
```

### Run API Client UI (Streamlit)

```bash
streamlit run app_api.py
```

### Run Global Admin Portal (Streamlit)

```bash
streamlit run app_admin_global.py
```

### Run Tenant Admin Portal (Streamlit)

```bash
streamlit run app_admin_tenant.py
```

### Run V1 API

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Or use launcher scripts:

```bash
run_api.bat
```

```bash
./run_api.sh
```

### Run Queue Worker (Optional Queue Mode)

Set API execution mode to queue:

```bash
set VH_JOB_EXECUTION_MODE=queue
```

Start API and worker separately:

```bash
run_api.bat
```

```bash
run_worker.bat
```

Validation:

```bash
python test_api_smoke.py
python test_api_auth_queue.py
python -m pytest --cov=backend --cov-report=term-missing
```

or:

```bash
run_tests.bat
```

### Auth and Roles (Configurable)

Default mode is open dev access (`VH_AUTH_REQUIRED=false`).

To require bearer tokens:

```bash
set VH_AUTH_REQUIRED=true
set VH_API_TOKENS=admin-token:admin,coach-token:coach,analyst-token:analyst,tenant-admin-token:tenant_admin
```

Then call APIs with:

```bash
Authorization: Bearer admin-token
```

JWT support is also available:

1. Set `VH_JWT_SECRET` (required for JWT issue/verify)
2. Optional: `VH_AUTH_BOOTSTRAP_KEY` for initial token bootstrap
3. Issue JWT via `POST /v1/auth/token`
4. Inspect current identity via `GET /v1/auth/me`
5. Send `X-Tenant-Id` header (tenant id or slug) for tenant-scoped endpoints

### Multi-tenant and Admin API

Tenant-scoped APIs require tenant context. Use request header:

```bash
X-Tenant-Id: <tenant_id_or_slug>
```

Admin surfaces:

1. Global admin API: `/v1/admin/global/*` (tenant, user, membership, inventory controls)
2. Tenant admin API: `/v1/admin/tenant/*` (tenant-scoped users, memberships, summary)

Portal launchers:

```bash
run_admin_global.bat
run_admin_tenant.bat
```

### Storage Backend

Default storage is local filesystem (`VH_STORAGE_BACKEND=local`).

S3-compatible mode:

```bash
set VH_STORAGE_BACKEND=s3
set VH_S3_BUCKET=video-highlights
set VH_S3_ENDPOINT_URL=https://<s3-compatible-endpoint>
set VH_S3_ACCESS_KEY_ID=<key>
set VH_S3_SECRET_ACCESS_KEY=<secret>
set VH_S3_REGION=<region>
set VH_S3_KEY_PREFIX=video-highlights
```

Use `GET /v1/matches/{match_id}/assets/{asset_id}/download-url` to resolve local paths or signed URLs.

### Run Desktop UI

```bash
python VideoHighlightsGUI.py
```

## Docker and GPU Notes

Containerized GPU execution supports local PC hosts, NVIDIA-capable edge systems, and cloud GPU nodes when configured correctly.

Required host setup:

1. NVIDIA drivers installed
2. NVIDIA Container Toolkit installed
3. CUDA-compatible runtime/image stack
4. GPU worker run with `--gpus all`

Example:

```bash
docker run --rm -it --gpus all your-image:gpu
```

## Docker Desktop Launch (This Repo)

1. Open Docker Desktop and confirm the engine is running.
2. From repo root, launch core services:

```bash
docker compose up --build api worker api-client admin-global admin-tenant
```

3. Open apps:
- API: `http://localhost:8000/docs`
- API Client: `http://localhost:8501`
- Global Admin Portal: `http://localhost:8502`
- Tenant Admin Portal: `http://localhost:8503`

Windows convenience:

```bash
run_docker.bat
```

GPU worker profile (requires NVIDIA toolkit and Docker GPU support):

```bash
docker compose --profile gpu up --build api worker-gpu api-client admin-global admin-tenant
```

Windows convenience:

```bash
run_docker_gpu.bat
```

## Documentation

1. `PRD.md`: Product requirements and acceptance criteria
2. `ROADMAP.md`: Phase plan and milestone scope
3. `REQUIREMENTS_TRACEABILITY.md`: Mapping of context-window requirements to PRD and roadmap
4. `LOCAL_SETUP.md`: Local and Docker setup guidance
5. `PERFORMANCE_IMPROVEMENTS.md`: Implemented optimizations in current code
6. `PERFORMANCE_RECOMMENDATIONS.md`: Next optimization opportunities
7. `FEEDBACK_EVENT_API_SCHEMA.md`: Concrete event/feedback payload schema and API endpoints
8. `IMPLEMENTATION_STATUS.md`: Current build status and next autonomous implementation queue
9. `TESTING.md`: Automated testing framework, local commands, and CI
