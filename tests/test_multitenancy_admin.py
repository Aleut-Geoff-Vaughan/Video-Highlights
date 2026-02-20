from __future__ import annotations

from fastapi.testclient import TestClient


def _admin_headers() -> dict[str, str]:
    return {"X-User-Id": "global_admin_1", "X-User-Role": "admin"}


def _create_tenant(client: TestClient, slug: str, name: str) -> str:
    response = client.post(
        "/v1/admin/global/tenants",
        headers=_admin_headers(),
        json={"slug": slug, "name": name, "status": "active", "metadata": {}},
    )
    assert response.status_code == 201, response.text
    return response.json()["tenant_id"]


def _create_user(client: TestClient, user_id: str) -> None:
    response = client.post(
        "/v1/admin/global/users",
        headers=_admin_headers(),
        json={
            "user_id": user_id,
            "email": f"{user_id}@example.com",
            "display_name": user_id,
            "status": "active",
            "is_global_admin": False,
            "metadata": {},
        },
    )
    assert response.status_code == 201, response.text


def _create_membership(client: TestClient, tenant_id: str, user_id: str, role: str) -> None:
    response = client.post(
        f"/v1/admin/global/tenants/{tenant_id}/memberships",
        headers=_admin_headers(),
        json={"user_id": user_id, "role": role, "status": "active", "metadata": {}},
    )
    assert response.status_code in {200, 201}, response.text


def test_cross_tenant_match_isolation(client: TestClient) -> None:
    tenant_a = _create_tenant(client, "club-a", "Club A")
    tenant_b = _create_tenant(client, "club-b", "Club B")

    _create_user(client, "coach_a")
    _create_user(client, "coach_b")
    _create_membership(client, tenant_a, "coach_a", "coach")
    _create_membership(client, tenant_b, "coach_b", "coach")

    create_match = client.post(
        "/v1/matches",
        headers={"X-User-Id": "coach_a", "X-User-Role": "coach", "X-Tenant-Id": tenant_a},
        json={"name": "A Match", "source_video_path": "C:/tmp/a.mp4", "metadata": {}},
    )
    assert create_match.status_code == 201, create_match.text
    match_id = create_match.json()["match_id"]

    own_tenant_get = client.get(
        f"/v1/matches/{match_id}",
        headers={"X-User-Id": "coach_a", "X-User-Role": "coach", "X-Tenant-Id": tenant_a},
    )
    assert own_tenant_get.status_code == 200

    other_tenant_get = client.get(
        f"/v1/matches/{match_id}",
        headers={"X-User-Id": "coach_b", "X-User-Role": "coach", "X-Tenant-Id": tenant_b},
    )
    assert other_tenant_get.status_code == 404

    other_tenant_list = client.get(
        "/v1/matches",
        headers={"X-User-Id": "coach_b", "X-User-Role": "coach", "X-Tenant-Id": tenant_b},
    )
    assert other_tenant_list.status_code == 200
    assert all(item["match_id"] != match_id for item in other_tenant_list.json()["items"])


def test_global_and_tenant_admin_management_paths(client: TestClient) -> None:
    tenant_id = _create_tenant(client, "club-admin", "Club Admin")
    _create_user(client, "tenant_admin_1")
    _create_membership(client, tenant_id, "tenant_admin_1", "tenant_admin")

    summary = client.get(
        "/v1/admin/tenant/summary",
        headers={"X-User-Id": "tenant_admin_1", "X-User-Role": "coach", "X-Tenant-Id": tenant_id},
    )
    assert summary.status_code == 200, summary.text
    assert summary.json()["tenant_id"] == tenant_id

    create_team_user = client.post(
        "/v1/admin/tenant/users",
        headers={"X-User-Id": "tenant_admin_1", "X-User-Role": "coach", "X-Tenant-Id": tenant_id},
        json={
            "user_id": "analyst_1",
            "email": "analyst_1@example.com",
            "display_name": "Analyst One",
            "user_status": "active",
            "role": "analyst",
            "membership_status": "active",
            "user_metadata": {},
            "membership_metadata": {},
        },
    )
    assert create_team_user.status_code == 201, create_team_user.text
    assert create_team_user.json()["role"] == "analyst"

    list_users = client.get(
        "/v1/admin/tenant/users",
        headers={"X-User-Id": "tenant_admin_1", "X-User-Role": "coach", "X-Tenant-Id": tenant_id},
    )
    assert list_users.status_code == 200
    assert any(item["user_id"] == "analyst_1" for item in list_users.json())

    tenant_b = _create_tenant(client, "club-other", "Club Other")
    forbidden = client.get(
        "/v1/admin/tenant/summary",
        headers={"X-User-Id": "tenant_admin_1", "X-User-Role": "coach", "X-Tenant-Id": tenant_b},
    )
    assert forbidden.status_code == 403
