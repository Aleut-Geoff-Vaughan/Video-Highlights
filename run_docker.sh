#!/usr/bin/env bash
set -euo pipefail
docker compose up --build api worker api-client processing-ui admin-global admin-tenant
