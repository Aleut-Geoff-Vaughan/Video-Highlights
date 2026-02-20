@echo off
setlocal

python -m pytest --cov=backend --cov-report=term-missing
