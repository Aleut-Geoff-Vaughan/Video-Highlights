#!/usr/bin/env bash
# All-in-one launcher: API + processing worker + all web UIs in one container.
#
# This is the image's default command so that `docker run` (or Docker
# Desktop's "Run a new container" dialog, which cannot override the command)
# delivers the complete product from a plain `docker pull`:
#
#   8000  FastAPI backend (docs at /docs)
#   8504  Processing portal (main UI)
#   8501  API client portal
#   8502  Global admin portal
#   8503  Tenant admin portal
#
# docker-compose deployments are unaffected: every compose service sets its
# own `command:`, overriding this script.
set -u

# Sensible single-container defaults; every value can be overridden with -e.
export VH_DB_URL="${VH_DB_URL:-sqlite:////app/data/video_highlights.db}"
export VH_OUTPUT_ROOT="${VH_OUTPUT_ROOT:-/app/data/outputs}"
export VH_LOCAL_STORAGE_ROOT="${VH_LOCAL_STORAGE_ROOT:-/app/data/storage}"
export VH_JOB_EXECUTION_MODE="${VH_JOB_EXECUTION_MODE:-queue}"
export VH_AUTH_REQUIRED="${VH_AUTH_REQUIRED:-false}"
export VH_SKIP_USER_MANAGEMENT="${VH_SKIP_USER_MANAGEMENT:-true}"
export VH_BASE_TENANT_SLUG="${VH_BASE_TENANT_SLUG:-sandbox}"
export VH_BASE_TENANT_NAME="${VH_BASE_TENANT_NAME:-Sandbox Tenant}"
# The portals talk to the API over localhost inside this same container.
export VH_PORTAL_API_BASE="${VH_PORTAL_API_BASE:-http://localhost:8000/v1}"
export VH_PORTAL_TENANT="${VH_PORTAL_TENANT:-sandbox}"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS="${STREAMLIT_BROWSER_GATHER_USAGE_STATS:-false}"

mkdir -p "$VH_OUTPUT_ROOT" "$VH_LOCAL_STORAGE_ROOT" /app/data

pids=()

start() {
    echo "[start-all] launching: $*"
    "$@" &
    pids+=($!)
}

shutdown() {
    echo "[start-all] shutting down..."
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait
    exit 0
}
trap shutdown SIGTERM SIGINT

start python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
# Give the API a moment to create the database before dependents start.
sleep 3
start python -m backend.worker
start streamlit run app.py --server.address=0.0.0.0 --server.port=8504 --server.headless=true
start streamlit run app_api.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true
start streamlit run app_admin_global.py --server.address=0.0.0.0 --server.port=8502 --server.headless=true
start streamlit run app_admin_tenant.py --server.address=0.0.0.0 --server.port=8503 --server.headless=true
start streamlit run app_review.py --server.address=0.0.0.0 --server.port=8505 --server.headless=true

echo "[start-all] all services launched:"
echo "  API:                http://localhost:8000/docs"
echo "  Processing portal:  http://localhost:8504"
echo "  API client portal:  http://localhost:8501"
echo "  Global admin:       http://localhost:8502"
echo "  Tenant admin:       http://localhost:8503"
echo "  Review portal:      http://localhost:8505"

# If any service dies, stop the container so the failure is visible
# (restart policies can then recover it).
wait -n
echo "[start-all] a service exited; stopping container"
shutdown
