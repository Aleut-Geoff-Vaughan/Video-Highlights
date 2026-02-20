from __future__ import annotations

from fastapi.testclient import TestClient

from backend.config import settings


def test_issue_jwt_with_admin_token_and_auth_me(auth_client: TestClient) -> None:
    issue = auth_client.post(
        "/v1/auth/token",
        headers={"Authorization": "Bearer admin-token"},
        json={"user_id": "jwt_user_1", "role": "coach", "expires_in_minutes": 30},
    )
    assert issue.status_code == 200, issue.text
    payload = issue.json()
    assert payload["token_type"] == "bearer"
    access_token = payload["access_token"]
    assert access_token

    me = auth_client.get("/v1/auth/me", headers={"Authorization": f"Bearer {access_token}"})
    assert me.status_code == 200, me.text
    me_payload = me.json()
    assert me_payload["user_id"] == "jwt_user_1"
    assert me_payload["role"] == "coach"
    assert me_payload["auth_source"] == "jwt"


def test_issue_jwt_with_bootstrap_key(auth_client: TestClient) -> None:
    settings.auth_bootstrap_key = "bootstrap-secret"
    try:
        issue = auth_client.post(
            "/v1/auth/token",
            headers={"X-Bootstrap-Key": "bootstrap-secret"},
            json={"user_id": "bootstrap_user", "role": "analyst"},
        )
        assert issue.status_code == 200, issue.text
        token = issue.json()["access_token"]
        assert token
    finally:
        settings.auth_bootstrap_key = None


def test_issue_jwt_forbidden_for_non_admin(auth_client: TestClient) -> None:
    issue = auth_client.post(
        "/v1/auth/token",
        headers={"Authorization": "Bearer coach-token"},
        json={"user_id": "x", "role": "coach"},
    )
    assert issue.status_code == 403
