# Local Setup

This guide covers local development plus Docker execution on GPU-capable hosts.

Supported deployment styles:

1. Local PC host (Windows + WSL2, or Linux)
2. NVIDIA-capable edge/workstation hosts (for example, NVIDIA-enabled appliance or server)
3. Cloud VM/container hosts with NVIDIA GPU support

## 1. Prerequisites

- Docker Desktop (Windows) or Docker Engine (Linux)
- Python 3.10+ for non-container runs
- FFmpeg available for local non-container runs
- Optional but recommended: NVIDIA GPU

For GPU containers:

- Latest NVIDIA driver installed on host
- NVIDIA Container Toolkit configured on host runtime

## 2. Python Environment (Non-Docker)

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Run locally:

```bash
python VideoHighlights.py --video path/to/match.mp4 --out ./highlights_output
```

Run API locally:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

If you are upgrading from an older local DB schema, reset the SQLite file first:

```bash
del video_highlights_v1.db
rm -f video_highlights_v1.db
```

Optional API client UI:

```bash
streamlit run app_api.py
```

Processing portal UI:

```bash
streamlit run app.py
```

Global admin portal UI:

```bash
streamlit run app_admin_global.py
```

Tenant admin portal UI:

```bash
streamlit run app_admin_tenant.py
```

Queue mode (API + worker split):

```bash
set VH_JOB_EXECUTION_MODE=queue
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

In a second terminal:

```bash
python -m backend.worker
```

Run API smoke test:

```bash
python test_api_smoke.py
```

Run auth + queue smoke test:

```bash
python test_api_auth_queue.py
```

Run full automated test suite:

```bash
python -m pytest --cov=backend --cov-report=term-missing
```

or:

```bash
run_tests.bat
```

Optional auth lock-down:

```bash
set VH_AUTH_REQUIRED=true
set VH_API_TOKENS=admin-token:admin,coach-token:coach,analyst-token:analyst,tenant-admin-token:tenant_admin
```

JWT auth mode:

```bash
set VH_JWT_SECRET=replace-with-strong-secret
set VH_JWT_ISSUER=video-highlights
set VH_JWT_DEFAULT_EXP_MINUTES=120
```

Optional bootstrap token issuing:

```bash
set VH_AUTH_BOOTSTRAP_KEY=bootstrap-secret
```

Issue JWT token (once API is running):

```bash
curl -X POST http://localhost:8000/v1/auth/token ^
  -H "Authorization: Bearer admin-token" ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\":\"coach_1\",\"role\":\"coach\",\"tenant_id\":\"default\",\"expires_in_minutes\":120}"
```

Tenant-scoped API requests:

```bash
curl -X GET http://localhost:8000/v1/matches ^
  -H "Authorization: Bearer coach-token" ^
  -H "X-Tenant-Id: default"
```

Skip user-management for core testing (auto-provision memberships):

```bash
set VH_SKIP_USER_MANAGEMENT=true
set VH_BASE_TENANT_SLUG=sandbox
set VH_BASE_TENANT_NAME=Sandbox Tenant
```

Enable deep test/debug logging (including extreme job logs):

```bash
set VH_TEST_MODE=true
set VH_LOG_LEVEL=DEBUG
set VH_JOB_LOG_DETAIL=extreme
```

Global admin API examples:

```bash
curl -X GET http://localhost:8000/v1/admin/global/summary ^
  -H "Authorization: Bearer admin-token"
```

```bash
curl -X POST http://localhost:8000/v1/admin/global/tenants ^
  -H "Authorization: Bearer admin-token" ^
  -H "Content-Type: application/json" ^
  -d "{\"slug\":\"club-a\",\"name\":\"Club A\",\"status\":\"active\",\"metadata\":{}}"
```

Tenant admin API example:

```bash
curl -X GET http://localhost:8000/v1/admin/tenant/summary ^
  -H "Authorization: Bearer tenant-admin-token" ^
  -H "X-Tenant-Id: club-a"
```

Job log and fast-kill examples:

```bash
curl -X GET http://localhost:8000/v1/jobs/{job_id}/logs?detail_level=extreme&limit=500 ^
  -H "X-Tenant-Id: sandbox"
```

```bash
curl -X POST http://localhost:8000/v1/jobs/{job_id}/kill-session ^
  -H "X-Tenant-Id: sandbox" ^
  -H "Content-Type: application/json" ^
  -d "{}"
```

Rerun job with updated model/version targets:

```bash
curl -X POST http://localhost:8000/v1/jobs/{job_id}/rerun ^
  -H "X-Tenant-Id: sandbox" ^
  -H "Content-Type: application/json" ^
  -d "{\"config_overrides\":{\"model_version\":\"event-v1\",\"focus_event_types\":[\"goal\",\"corner_kick\"]},\"reason\":\"model-upgrade\"}"
```

Create analysis-only job (bookmark table only, no clip rendering):

```bash
curl -X POST http://localhost:8000/v1/matches/{match_id}/jobs ^
  -H "X-Tenant-Id: sandbox" ^
  -H "Content-Type: application/json" ^
  -d "{\"config\":{\"analysis_only\":true,\"model_version\":\"event-v1\",\"focus_event_types\":[\"goal\",\"corner_kick\"]}}"
```

Fetch bookmark/event table for a specific run:

```bash
curl -X GET "http://localhost:8000/v1/matches/{match_id}/events?job_id={job_id}&limit=1000" ^
  -H "X-Tenant-Id: sandbox"
```

Fetch live bookmark table for a running/completed job:

```bash
curl -X GET "http://localhost:8000/v1/jobs/{job_id}/bookmarks?limit=5000" ^
  -H "X-Tenant-Id: sandbox"
```

Delete an old run (removes run + logs + run-linked events):

```bash
curl -X DELETE "http://localhost:8000/v1/jobs/{job_id}" ^
  -H "X-Tenant-Id: sandbox"
```

Render frame-accurate clip for an event bookmark:

```bash
curl -X POST http://localhost:8000/v1/matches/{match_id}/events/{event_id}/clip-on-demand ^
  -H "X-Tenant-Id: sandbox" ^
  -H "Content-Type: application/json" ^
  -d "{\"pre_seconds\":1.5,\"post_seconds\":5.0,\"anchor\":\"event_window\",\"include_audio\":true,\"prefer_gpu\":true,\"force_rebuild\":false}"
```

Export a final highlight reel from selected bookmark events:

```bash
curl -X POST "http://localhost:8000/v1/matches/{match_id}/exports/highlights" ^
  -H "X-Tenant-Id: sandbox" ^
  -H "Content-Type: application/json" ^
  -d "{\"event_ids\":[\"evt_1\",\"evt_2\"],\"pre_seconds\":1.0,\"post_seconds\":3.0,\"anchor\":\"event_window\",\"include_audio\":true,\"prefer_gpu\":true,\"title\":\"Selected Highlights\"}"
```

S3-compatible storage mode:

```bash
set VH_STORAGE_BACKEND=s3
set VH_S3_BUCKET=video-highlights
set VH_S3_ENDPOINT_URL=https://<s3-endpoint>
set VH_S3_ACCESS_KEY_ID=<key>
set VH_S3_SECRET_ACCESS_KEY=<secret>
set VH_S3_REGION=<region>
set VH_S3_KEY_PREFIX=video-highlights
```

## 3. GPU Verification

Host checks:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

Expected result: CUDA availability is `True` when GPU-enabled torch/runtime is installed.

## 4. Docker Execution

Docker assets are now included in repo:

- `Dockerfile`
- `docker-compose.yml`

### CPU Container

```bash
docker run --rm -it \
  -v ${PWD}:/workspace \
  your-image:cpu \
  python VideoHighlights.py --video /workspace/input.mp4 --out /workspace/output
```

### GPU Container

```bash
docker run --rm -it --gpus all \
  -v ${PWD}:/workspace \
  your-image:gpu \
  python VideoHighlights.py --video /workspace/input.mp4 --out /workspace/output
```

### Docker Desktop (Recommended for This Project)

Start API, queue worker, API client portal, global admin portal, and tenant admin portal:

```bash
docker compose up --build api worker api-client processing-ui admin-global admin-tenant
```

Endpoints:

1. API docs: `http://localhost:8000/docs`
2. API client: `http://localhost:8501`
3. Processing portal (dashboard + runs): `http://localhost:8504`
4. Global admin portal: `http://localhost:8502`
5. Tenant admin portal: `http://localhost:8503`

Windows helper script:

```bash
run_docker.bat
```

Stop everything quickly:

```bash
stop_docker.bat
```

GPU worker profile:

```bash
docker compose --profile gpu up --build api worker-gpu api-client processing-ui admin-global admin-tenant
```

Windows helper script:

```bash
run_docker_gpu.bat
```

## 5. Recommended Container Topology

1. `web` (CPU)
2. `api` (CPU)
3. `worker-analysis-gpu` (GPU required)
4. `worker-render` (CPU or GPU optional)

Only schedule `worker-analysis-gpu` to GPU nodes.

## 6. Common Issues

### GPU not visible inside container

1. Confirm `nvidia-smi` works on host
2. Confirm NVIDIA Container Toolkit is installed
3. Run container with `--gpus all`
4. Verify image includes CUDA-compatible runtime and torch build

### FFmpeg codec errors

1. Ensure ffmpeg is installed in the image
2. Confirm required encoders are available (`libx264`, optional `h264_nvenc`)

### Slow output generation

1. Prefer SSD/NVMe output storage
2. Lower thread count on low-memory systems (`--threads 2`)
3. Disable overlay when collecting baseline performance data
