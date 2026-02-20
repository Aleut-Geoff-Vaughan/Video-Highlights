"""
Global Admin Portal (Streamlit)

Run:
    streamlit run app_admin_global.py
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests
import streamlit as st


st.set_page_config(page_title="Video Highlights - Global Admin", page_icon="VH", layout="wide")


def api_request(
    method: str,
    base_url: str,
    path: str,
    token: Optional[str],
    x_user_id: Optional[str],
    x_user_role: Optional[str],
    json_body: Optional[Dict[str, Any]] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    headers: Dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
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


st.title("Global Admin Portal")

with st.sidebar:
    st.header("Connection")
    api_base = st.text_input("API Base URL", value="http://localhost:8000/v1")
    token = st.text_input("Bearer Token", value="", type="password")
    st.caption("Optional dev fallback headers (only when auth is disabled)")
    x_user_id = st.text_input("X-User-Id", value="global_admin_1")
    x_user_role = st.selectbox("X-User-Role", ["admin", "system", "analyst", "coach", "parent", "tenant_admin"])

tabs = st.tabs(["Summary", "Tenants", "Users", "Memberships"])


with tabs[0]:
    if st.button("Refresh Global Summary"):
        result = api_request("GET", api_base, "/admin/global/summary", token, x_user_id, x_user_role)
        st.json(result)

    if st.button("Refresh Global Inventory"):
        result = api_request("GET", api_base, "/admin/global/inventory", token, x_user_id, x_user_role)
        st.json(result)


with tabs[1]:
    st.subheader("Create Tenant")
    tenant_slug = st.text_input("Tenant Slug", value="club-new")
    tenant_name = st.text_input("Tenant Name", value="New Club")
    tenant_status = st.selectbox("Tenant Status", ["active", "suspended", "archived"])
    tenant_meta = st.text_area("Tenant Metadata JSON", value="{}", height=100)

    if st.button("Create Tenant"):
        try:
            metadata = json.loads(tenant_meta) if tenant_meta.strip() else {}
        except json.JSONDecodeError:
            st.error("Invalid JSON in tenant metadata.")
            metadata = None
        if metadata is not None:
            result = api_request(
                "POST",
                api_base,
                "/admin/global/tenants",
                token,
                x_user_id,
                x_user_role,
                json_body={"slug": tenant_slug, "name": tenant_name, "status": tenant_status, "metadata": metadata},
            )
            st.json(result)

    st.subheader("List Tenants")
    if st.button("List Tenants"):
        result = api_request("GET", api_base, "/admin/global/tenants", token, x_user_id, x_user_role)
        st.json(result)

    st.subheader("Patch Tenant")
    patch_tenant_id = st.text_input("Tenant ID to Patch", value="")
    patch_tenant_name = st.text_input("New Tenant Name", value="")
    patch_tenant_status = st.selectbox("New Tenant Status", ["active", "suspended", "archived"], index=0)
    if st.button("Patch Tenant", disabled=not patch_tenant_id.strip()):
        body: Dict[str, Any] = {}
        if patch_tenant_name.strip():
            body["name"] = patch_tenant_name.strip()
        if patch_tenant_status:
            body["status"] = patch_tenant_status
        result = api_request(
            "PATCH",
            api_base,
            f"/admin/global/tenants/{patch_tenant_id.strip()}",
            token,
            x_user_id,
            x_user_role,
            json_body=body,
        )
        st.json(result)


with tabs[2]:
    st.subheader("Create User")
    create_user_id = st.text_input("User ID", value="user_new")
    create_user_email = st.text_input("Email", value="user_new@example.com")
    create_user_name = st.text_input("Display Name", value="User New")
    create_user_status = st.selectbox("User Status", ["active", "disabled", "invited"])
    create_user_global_admin = st.checkbox("Is Global Admin", value=False)
    create_user_meta = st.text_area("User Metadata JSON", value="{}", height=100)
    if st.button("Create User"):
        try:
            metadata = json.loads(create_user_meta) if create_user_meta.strip() else {}
        except json.JSONDecodeError:
            st.error("Invalid JSON in user metadata.")
            metadata = None
        if metadata is not None:
            result = api_request(
                "POST",
                api_base,
                "/admin/global/users",
                token,
                x_user_id,
                x_user_role,
                json_body={
                    "user_id": create_user_id,
                    "email": create_user_email or None,
                    "display_name": create_user_name or None,
                    "status": create_user_status,
                    "is_global_admin": create_user_global_admin,
                    "metadata": metadata,
                },
            )
            st.json(result)

    st.subheader("List Users")
    if st.button("List Users"):
        result = api_request("GET", api_base, "/admin/global/users", token, x_user_id, x_user_role)
        st.json(result)

    st.subheader("Patch User")
    patch_user_id = st.text_input("Patch User ID", value="")
    patch_user_status = st.selectbox("Patch User Status", ["active", "disabled", "invited"], index=0)
    patch_user_global_admin = st.checkbox("Patch Is Global Admin", value=False)
    if st.button("Patch User", disabled=not patch_user_id.strip()):
        result = api_request(
            "PATCH",
            api_base,
            f"/admin/global/users/{patch_user_id.strip()}",
            token,
            x_user_id,
            x_user_role,
            json_body={"status": patch_user_status, "is_global_admin": patch_user_global_admin},
        )
        st.json(result)


with tabs[3]:
    st.subheader("Create/Update Membership")
    membership_tenant_id = st.text_input("Tenant ID", value="")
    membership_user_id = st.text_input("User ID ", value="")
    membership_role = st.selectbox("Membership Role", ["tenant_admin", "coach", "analyst", "parent", "player", "system"])
    membership_status = st.selectbox("Membership Status", ["active", "invited", "disabled"])
    membership_meta = st.text_area("Membership Metadata JSON", value="{}", height=100)
    if st.button("Create/Update Membership", disabled=not membership_tenant_id.strip() or not membership_user_id.strip()):
        try:
            metadata = json.loads(membership_meta) if membership_meta.strip() else {}
        except json.JSONDecodeError:
            st.error("Invalid JSON in membership metadata.")
            metadata = None
        if metadata is not None:
            result = api_request(
                "POST",
                api_base,
                f"/admin/global/tenants/{membership_tenant_id.strip()}/memberships",
                token,
                x_user_id,
                x_user_role,
                json_body={
                    "user_id": membership_user_id.strip(),
                    "role": membership_role,
                    "status": membership_status,
                    "metadata": metadata,
                },
            )
            st.json(result)

    st.subheader("List Memberships by Tenant")
    list_memberships_tenant_id = st.text_input("Tenant ID for Listing", value="")
    if st.button("List Memberships", disabled=not list_memberships_tenant_id.strip()):
        result = api_request(
            "GET",
            api_base,
            f"/admin/global/tenants/{list_memberships_tenant_id.strip()}/memberships",
            token,
            x_user_id,
            x_user_role,
        )
        st.json(result)
