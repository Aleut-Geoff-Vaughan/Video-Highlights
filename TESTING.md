# Testing Framework

This project now includes an automated testing framework designed to evolve with implementation.

## 1. Goals

1. Catch regressions on every backend change.
2. Keep tests fast enough for frequent local execution.
3. Validate critical workflows end-to-end (events, feedback, training, agent, queue mode, auth).

## 2. Test Stack

1. `pytest` for test execution
2. `fastapi.testclient` for API integration testing
3. `pytest-cov` for backend coverage reporting
4. Isolated SQLite test DB per test function via fixture-driven engine reset

## 3. Test Structure

- `tests/conftest.py`: shared fixtures (isolated DB, client/auth_client)
- `tests/test_health.py`: health and basic middleware checks
- `tests/test_matches_events.py`: match/event API behavior
- `tests/test_feedback_training_agent.py`: feedback/review/training/agent flows
- `tests/test_auth_and_queue.py`: auth-required mode and queue-worker execution
- `tests/test_validation_errors.py`: validation envelope behavior
- `tests/test_jwt_auth.py`: JWT issue/verify and role restrictions
- `tests/test_storage_backends.py`: local and S3-compatible storage backend behavior
- `tests/test_contract_api.py`: OpenAPI path and schema-doc enum contract checks
- `tests/test_multitenancy_admin.py`: tenant isolation and global/tenant admin API workflows
- `tests/test_dev_skip_user_management.py`: seeded tenant + auto-provisioned membership test mode behavior
- `tests/test_job_logging_and_kill.py`: job log persistence/query and kill-session behavior
- `tests/test_job_bookmarks_analysis.py`: analysis-only bookmark manifest ingestion and job-linked event persistence
- `tests/test_event_clip_on_demand.py`: frame-accurate bookmark clip-on-demand creation and cache reuse behavior
- `tests/test_job_delete_and_live_bookmarks.py`: job-level bookmark feed and run deletion cleanup behavior
- `tests/test_highlight_export_selected.py`: selected-bookmark highlight export generation and metadata persistence

## 4. Running Tests

Run full suite:

```bash
python -m pytest
```

Install lightweight backend test dependencies:

```bash
pip install -r requirements-backend-test.txt
```

Run with coverage:

```bash
python -m pytest --cov=backend --cov-report=term-missing
```

Convenience scripts:

```bash
run_tests.bat
```

```bash
./run_tests.sh
```

## 5. CI

GitHub Actions workflow:

- File: `.github/workflows/ci.yml`
- Installs `requirements-backend-test.txt`
- Runs pytest with coverage output

## 6. Test Design Notes

1. Each test runs against an isolated SQLite DB configured by fixtures.
2. Queue mode is used in tests to avoid non-deterministic inline async background behavior.
3. Auth-required flows are tested with token-role mappings.
4. Tenant-scoped isolation is validated with explicit cross-tenant access checks.
5. Tests avoid requiring heavy CV runtime dependencies for fast CI.

## 7. Next Testing Enhancements

1. Add negative/edge tests for large payload and pagination boundaries.
2. Add performance benchmarks for queue-worker throughput.
3. Add integration tests for real processing runs in a GPU-capable pipeline environment.
4. Add API contract snapshot tests across versions.
