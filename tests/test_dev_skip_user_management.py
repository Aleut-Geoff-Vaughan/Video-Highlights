from __future__ import annotations

from fastapi.testclient import TestClient

from backend.config import settings


def test_skip_user_management_auto_provisions_membership(client: TestClient) -> None:
    settings.skip_user_management = True
    settings.base_tenant_slug = "autoseed"
    settings.base_tenant_name = "Auto Seed Tenant"
    try:
        create = client.post(
            "/v1/matches",
            headers={"X-User-Id": "coach_no_membership", "X-User-Role": "coach", "X-Tenant-Id": "autoseed"},
            json={"name": "Core Test Match", "source_video_path": "C:/tmp/nonexistent.mp4", "metadata": {}},
        )
        assert create.status_code == 201, create.text
        assert create.json()["tenant_id"] is not None

        summary = client.get(
            "/v1/admin/tenant/summary",
            headers={"X-User-Id": "tenant_admin_no_setup", "X-User-Role": "tenant_admin", "X-Tenant-Id": "autoseed"},
        )
        assert summary.status_code == 200, summary.text
        assert summary.json()["tenant_slug"] == "autoseed"
    finally:
        settings.skip_user_management = False
        settings.base_tenant_slug = "sandbox"
        settings.base_tenant_name = "Sandbox Tenant"
