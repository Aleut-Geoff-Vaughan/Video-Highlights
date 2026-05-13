# Video Highlights

Video Highlights is a soccer video analysis platform that converts full-match recordings into events, player spotlights, team analytics, and shareable highlight packages.

## Project Status

This repository now provides a local Python pipeline plus a web/API foundation with multitenant controls:

1. CLI pipeline: `VideoHighlights.py`
2. Desktop UI: `VideoHighlightsGUI.py`
3. Streamlit processing portal UI: `app.py`
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

Follow-cam clip export is available for player-centric runs:

```bash
python VideoHighlights.py --video path/to/match.mp4 --out ./highlights_output --camera-mode follow_action --zoom-factor 1.6
```

Use `--camera-mode wide` to preserve the original frame. `follow_player` keeps the selected or auto-selected player centered; `follow_action` blends the player track with nearby ball detections.

### Run Processing Portal (Streamlit)

```bash
set VH_PORTAL_API_BASE=http://127.0.0.1:8000/v1
python -m streamlit run app.py
```

For large match files, including 10GB+ recordings, use the portal's **Local File Path (10GB+)** video source. Use **Browse** or paste a path to register a file that already exists on the API/worker machine and process it in place. The portal preflights the path through the API worker and reports file size, basic media metadata when `ffprobe` is available, and clear messages for missing, zero-byte, cloud-placeholder, or still-copying files. Browser upload is intended only for smaller files.

Leave **Limit to test window** enabled for the first smoke test. The default window processes only the first 2 minutes, which is much faster and safer than starting with a full 10GB match.

### GPU Acceleration

The API exposes GPU readiness at:

```bash
curl http://127.0.0.1:8000/v1/health/gpu
```

On NVIDIA systems, install CUDA-enabled PyTorch instead of the CPU wheel. For this machine's CUDA 13 driver path:

```bash
python -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install "pillow<12,>=9.2.0"
```

The portal sidebar shows whether PyTorch CUDA is ready and whether FFmpeg can see `h264_nvenc` for GPU clip rendering. When **Require GPU** is enabled, jobs fail early if CUDA is not available.

The main analysis controls include a **GPU Analysis** section. The current default is `yolo26s.pt`, `botsort.yaml`, image size `960`, confidence `0.18`, and frame stride `1`. For heavier GPU use and better small-player detection, try `yolo26m.pt` with image size `1280`; for quick smoke tests, lower image size or increase frame stride. You can also provide a custom `.pt` path from a fine-tuned detector.

### YOLO Detector Training

The Training Lab can launch real Ultralytics detector training from a YOLO dataset YAML. The training run writes a normal Ultralytics `best.pt`; paste that path into **GPU Analysis > Custom Detector Weights** for future processing runs.

API example:

```bash
curl -X POST http://127.0.0.1:8000/v1/training/runs ^
  -H "Content-Type: application/json" ^
  -H "X-Tenant-Id: sandbox" ^
  -d "{\"target_model\":\"yolo-detector\",\"training_config\":{\"kind\":\"ultralytics_yolo\",\"dataset_yaml\":\"C:\\\\datasets\\\\soccer\\\\data.yaml\",\"base_model\":\"yolo26s.pt\",\"epochs\":50,\"imgsz\":960,\"batch\":8,\"device\":\"0\"}}"
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

### LLM Analysis Provider

The match assistant endpoints (`/v1/matches/{match_id}/agent/*`) can run with fallback summaries, OpenAI, or a local LLM.

OpenAI cloud:

```bash
set VH_LLM_PROVIDER=openai
set VH_LLM_MODEL=gpt-4o-mini
set OPENAI_API_KEY=<your-key>
```

Ollama local:

```bash
set VH_LLM_PROVIDER=ollama
set VH_LLM_MODEL=gemma4:e2b
set VH_LLM_BASE_URL=http://127.0.0.1:11434
```

Check the active AI configuration:

```bash
curl -H "X-Tenant-Id: sandbox" http://127.0.0.1:8000/v1/agent/status
```

The Streamlit match assistant also shows provider, model, reachability, and local setup hints.

OpenAI-compatible local servers (LM Studio, vLLM, llama.cpp server, Ollama `/v1` mode):

```bash
set VH_LLM_PROVIDER=openai_compatible
set VH_LLM_MODEL=local-model-name
set VH_LLM_BASE_URL=http://127.0.0.1:1234/v1
set VH_LLM_API_KEY=local-dev-key
```

Optional timeout override:

```bash
set VH_LLM_TIMEOUT_SECONDS=20
```

By default, Ollama models are unloaded immediately after each assistant response so the GPU stays available for video analysis. Override only if you want faster back-to-back assistant chats:

```bash
set VH_LLM_KEEP_ALIVE=5m
```

### Multi-tenant and Admin API

Tenant-scoped APIs require tenant context. Use request header:

```bash
X-Tenant-Id: <tenant_id_or_slug>
```

Admin surfaces:

1. Global admin API: `/v1/admin/global/*` (tenant, user, membership, inventory controls)
2. Tenant admin API: `/v1/admin/tenant/*` (tenant-scoped users, memberships, summary)

Quick dev mode to test core flows without managing users:

```bash
set VH_SKIP_USER_MANAGEMENT=true
set VH_BASE_TENANT_SLUG=sandbox
set VH_BASE_TENANT_NAME=Sandbox Tenant
```

With this enabled, tenant membership is auto-provisioned on first use for the selected tenant.

Test/Debug logging mode:

```bash
set VH_TEST_MODE=true
set VH_LOG_LEVEL=DEBUG
set VH_JOB_LOG_DETAIL=extreme
```

Per-run logging profiles are also available from the Processing Portal:

1. `Standard`: core status and failure logs.
2. `Detailed`: process-language checkpoints plus technical context for normal testing.
3. `Diagnostic`: detailed logs plus raw config/pipeline invocation checkpoints for deep debugging.

Run Monitor's Log Inspector can show the same logs as a **Process Story**, a **Technical Table**, or raw rows.

Job-level debug endpoints:

1. `GET /v1/jobs/{job_id}/logs` (supports `level`, `stage`, `detail_level`, `limit`)
2. `GET /v1/jobs/{job_id}/diagnostics` (human-readable status summary, likely issue, and next action)
3. `GET /v1/jobs/{job_id}/bookmarks` (live bookmark table from events/job result/manifest)
4. `POST /v1/jobs/{job_id}/kill-session` (fast cancel path for testing)
5. `POST /v1/jobs/{job_id}/rerun` (rerun with optional config/model/event-target overrides)
6. `GET /v1/matches/{match_id}/events?job_id=<job_id>` (bookmark/event table for a specific processing run)
7. `POST /v1/matches/{match_id}/events/{event_id}/clip-on-demand` (frame-accurate bookmark clip rendering with cache reuse)
8. `DELETE /v1/jobs/{job_id}` (delete old run, including job logs and job-linked events)
9. `POST /v1/matches/{match_id}/exports/highlights` (export one highlight reel from selected bookmarks/events)

Codec fallback behavior:

1. Clip export attempts `h264_nvenc` first when available.
2. If NVENC fails at runtime, export auto-falls back to `libx264`, then `mpeg4`.

Analysis-only mode and bookmark outputs:

1. Set job config `"analysis_only": true` to skip clip rendering and generate fast event/bookmark analysis only.
2. Set job config `"trim_start"` and `"trim_end"` in seconds, or use the portal **Limit to test window** control, to process only a short slice of a match.
3. Every run writes `analysis_bookmarks.json` and `analysis_bookmarks.csv` to the job output directory.
4. Completed jobs persist bookmark data in job result payload and auto-create `Event` rows linked to the job.
5. Processing Portal Game Library includes full-match playback with bookmark jump controls.
6. Bookmark rows can render frame-accurate on-demand clips without reprocessing the full match.
7. Operations Console includes bulk queue controls to kill queued/active jobs for a selected match.
8. Game Library includes a Match Workspace for one-click reprocess (latest config) and custom reprocess (same uploaded source video).
9. Processing Portal supports an Experience toggle (`User Friendly` / `Technical`) for non-technical vs advanced workflows.
10. Match Studio supports deleting old runs per match and exporting selected bookmarks into a final highlight reel.

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
docker compose up --build api worker api-client processing-ui admin-global admin-tenant
```

3. Open apps:
- API: `http://localhost:8000/docs`
- API Client: `http://localhost:8501`
- Processing Portal (dashboard + runs): `http://localhost:8504`
- Global Admin Portal: `http://localhost:8502`
- Tenant Admin Portal: `http://localhost:8503`

Windows convenience:

```bash
run_docker.bat
```

Stop all containers quickly:

```bash
stop_docker.bat
```

GPU worker profile (requires NVIDIA toolkit and Docker GPU support):

```bash
docker compose --profile gpu up --build api worker-gpu api-client processing-ui admin-global admin-tenant
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
