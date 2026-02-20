#!/usr/bin/env bash
set -euo pipefail

python -m pytest --cov=backend --cov-report=term-missing
