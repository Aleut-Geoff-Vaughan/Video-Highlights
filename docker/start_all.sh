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

echo "[start-all] all services launched:"
echo "  Studio (web UI + API):  http://localhost:8000"
echo "  API docs:               http://localhost:8000/docs"

# If any service dies, stop the container so the failure is visible
# (restart policies can then recover it).
wait -n
echo "[start-all] a service exited; stopping container"
shutdown
