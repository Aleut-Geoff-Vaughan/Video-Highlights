#!/usr/bin/env bash
set -euo pipefail
docker compose --profile gpu up --build api worker-gpu api-client admin-global admin-tenant
