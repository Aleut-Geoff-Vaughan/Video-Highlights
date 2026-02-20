"""
Tenant Admin Portal (Streamlit)

Run:
    streamlit run app_admin_tenant.py
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests
import streamlit as st


st.set_page_config(page_title="Video Highlights - Tenant Admin", page_icon="VH", layout="wide")


def api_request(
    method: str,
    base_url: str,
    path: str,
    token: Optional[str],
    tenant_id: Optional[str],
    x_user_id: Optional[str],
    x_user_role: Optional[str],
    json_body: Optional[Dict[str, Any]] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    headers: Dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if tenant_id:
        headers["X-Tenant-Id"] = tenant_id
    if x_user_id:
        headers["X-User-Id"] = x_user_id
    if x_user_role:
        headers["X-User-Role"] = x_user_role
    url = f"{base_url.rstrip('/')}{path}"
    response = requests.request(method.upper(), url, headers=headers, json=json_body, timeout=timeout)
    try:
        payload = response.json()
    except Exception:
        payload = {"raw": response.text}
    return {"ok": response.ok, "status_code": response.status_code, "payload": payload}


st.title("Tenant Admin Portal")

with st.sidebar:
    st.header("Connection")
    api_base = st.text_input("API Base URL", value="http://localhost:8000/v1")
    token = st.text_input("Bearer Token", value="", type="password")
    tenant_id = st.text_input("Tenant ID/Slug", value="default")
    st.caption("Optional dev fallback headers (only when auth is disabled)")
    x_user_id = st.text_input("X-User-Id", value="tenant_admin_1")
    x_user_role = st.selectbox("X-User-Role", ["tenant_admin", "coach", "analyst", "admin", "parent", "system"])

tabs = st.tabs(["Summary", "Users", "Memberships", "Matches"])


with tabs[0]:
    if st.button("Refresh Tenant Summary"):
        result = api_request(
            "GET",
            api_base,
            "/admin/tenant/summary",
            token,
            tenant_id,
            x_user_id,
            x_user_role,
        )
        st.json(result)

    if st.button("Refresh Tenant Inventory"):
        result = api_request(
            "GET",
            api_base,
            "/admin/tenant/inventory",
            token,
            tenant_id,
            x_user_id,
            x_user_role,
        )
        st.json(result)


with tabs[1]:
    st.subheader("Create Tenant User")
    create_user_id = st.text_input("User ID", value="team_user_1")
    create_email = st.text_input("Email", value="team_user_1@example.com")
    create_name = st.text_input("Display Name", value="Team User 1")
    create_role = st.selectbox("Role", ["coach", "analyst", "parent", "player", "tenant_admin"])
    create_user_status = st.selectbox("User Status", ["active", "disabled", "invited"], index=0)
    create_membership_status = st.selectbox("Membership Status", ["active", "invited", "disabled"], index=0)
    create_user_meta = st.text_area("User Metadata JSON", value="{}", height=100)
    create_membership_meta = st.text_area("Membership Metadata JSON", value="{}", height=100)
    if st.button("Create Tenant User"):
        try:
            user_meta = json.loads(create_user_meta) if create_user_meta.strip() else {}
            membership_meta = json.loads(create_membership_meta) if create_membership_meta.strip() else {}
        except json.JSONDecodeError:
            st.error("Invalid JSON in metadata fields.")
            user_meta = None
            membership_meta = None
        if user_meta is not None and membership_meta is not None:
            result = api_request(
                "POST",
                api_base,
                "/admin/tenant/users",
                token,
                tenant_id,
                x_user_id,
                x_user_role,
                json_body={
                    "user_id": create_user_id,
                    "email": create_email or None,
                    "display_name": create_name or None,
                    "user_status": create_user_status,
                    "role": create_role,
                    "membership_status": create_membership_status,
                    "user_metadata": user_meta,
                    "membership_metadata": membership_meta,
                },
            )
            st.json(result)

    st.subheader("List Tenant Users")
    if st.button("List Tenant Users"):
        result = api_request(
            "GET",
            api_base,
            "/admin/tenant/users",
            token,
            tenant_id,
            x_user_id,
            x_user_role,
        )
        st.json(result)

    st.subheader("Patch Tenant User")
    patch_user_id = st.text_input("Patch User ID", value="")
    patch_role = st.selectbox("Patch Role", ["coach", "analyst", "parent", "player", "tenant_admin"], index=0)
    patch_user_status = st.selectbox("Patch User Status", ["active", "disabled", "invited"], index=0)
    patch_membership_status = st.selectbox("Patch Membership Status", ["active", "invited", "disabled"], index=0)
    if st.button("Patch Tenant User", disabled=not patch_user_id.strip()):
        result = api_request(
            "PATCH",
            api_base,
            f"/admin/tenant/users/{patch_user_id.strip()}",
            token,
            tenant_id,
            x_user_id,
            x_user_role,
            json_body={
                "role": patch_role,
                "user_status": patch_user_status,
                "membership_status": patch_membership_status,
            },
        )
        st.json(result)


with tabs[2]:
    st.subheader("Patch Membership")
    membership_id = st.text_input("Membership ID", value="")
    membership_role = st.selectbox("Membership Role", ["tenant_admin", "coach", "analyst", "parent", "player", "system"])
    membership_status = st.selectbox("Membership Status", ["active", "invited", "disabled"])
    membership_meta = st.text_area("Membership Metadata JSON ", value="{}", height=100)
    if st.button("Patch Membership", disabled=not membership_id.strip()):
        try:
            metadata = json.loads(membership_meta) if membership_meta.strip() else {}
        except json.JSONDecodeError:
            st.error("Invalid JSON in membership metadata.")
            metadata = None
        if metadata is not None:
            result = api_request(
                "PATCH",
                api_base,
                f"/admin/tenant/memberships/{membership_id.strip()}",
                token,
                tenant_id,
                x_user_id,
                x_user_role,
                json_body={"role": membership_role, "status": membership_status, "metadata": metadata},
            )
            st.json(result)


with tabs[3]:
    if st.button("List Tenant Matches"):
        result = api_request(
            "GET",
            api_base,
            "/admin/tenant/matches",
            token,
            tenant_id,
            x_user_id,
            x_user_role,
        )
        st.json(result)
