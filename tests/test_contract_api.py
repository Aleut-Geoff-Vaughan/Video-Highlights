from __future__ import annotations

from pathlib import Path
from typing import get_args

from fastapi.testclient import TestClient

from backend.schemas import EventType, FeedbackType


REQUIRED_PATHS = {
    "/v1/matches/{match_id}/events",
    "/v1/matches/{match_id}/events/{event_id}",
    "/v1/matches/{match_id}/events/{event_id}/feedback",
    "/v1/matches/{match_id}/feedback",
    "/v1/training/feedback-batches",
    "/v1/training/runs",
    "/v1/training/runs/{run_id}/promote",
    "/v1/training/models",
    "/v1/auth/token",
    "/v1/auth/me",
    "/v1/matches/{match_id}/assets/{asset_id}/download-url",
    "/v1/admin/global/summary",
    "/v1/admin/global/tenants",
    "/v1/admin/global/users",
    "/v1/admin/global/tenants/{tenant_id}/memberships",
    "/v1/admin/tenant/summary",
    "/v1/admin/tenant/users",
}


def test_openapi_contains_required_paths(client: TestClient) -> None:
    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200, openapi.text
    paths = set(openapi.json().get("paths", {}).keys())
    missing = REQUIRED_PATHS - paths
    assert not missing, f"Missing API paths: {sorted(missing)}"


def test_schema_doc_contains_event_and_feedback_enums() -> None:
    schema_doc = Path("FEEDBACK_EVENT_API_SCHEMA.md").read_text(encoding="utf-8")
    for item in get_args(EventType):
        assert f"`{item}`" in schema_doc
    for item in get_args(FeedbackType):
        assert f"`{item}`" in schema_doc
