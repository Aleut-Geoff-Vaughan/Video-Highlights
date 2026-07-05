"""Video Highlights Studio - the unified portal.

ONE application, one port, studio-style navigation. The former standalone
portals (processing, review, API console, global/tenant admin) are pages of
this app; run this file and everything is in the left navigation:

    streamlit run app_studio.py --server.port=8504

Design notes: dark professional theme (set in .streamlit/config.toml), a
persistent left rail with grouped sections (Studio / Operations / Admin),
and page chrome injected here so every page shares the same look.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Video Highlights Studio",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# The page scripts below were once standalone apps and each still calls
# st.set_page_config at import time; that is only allowed once per run, so
# neutralize it after our own call.
st.set_page_config = lambda *args, **kwargs: None  # type: ignore[assignment]

# Shared studio chrome: tighten spacing, card look, nav styling.
st.markdown(
    """
    <style>
      #MainMenu, footer {visibility: hidden;}
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      [data-testid="stSidebarNav"] {padding-top: 0.4rem;}
      [data-testid="stSidebar"] {min-width: 270px;}
      [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 10px 14px;
      }
      div[data-testid="stDataFrame"] {border-radius: 10px; overflow: hidden;}
      .stRadio [role="radiogroup"] {gap: 0.35rem;}
      h1 {font-size: 1.6rem;}
      h2 {font-size: 1.25rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🎥 Video Highlights\n**Studio**")
    st.caption("Record → Analyze → Review → Share")
    st.divider()

pages = {
    "Studio": [
        st.Page("app_review.py", title="Library & Review", icon="🎬", default=True),
        st.Page("app.py", title="Create & Process", icon="⚙️"),
    ],
    "Operations": [
        st.Page("app_api.py", title="API Console", icon="🔌"),
    ],
    "Admin": [
        st.Page("app_admin_tenant.py", title="Tenant Admin", icon="👥"),
        st.Page("app_admin_global.py", title="Global Admin", icon="🌐"),
    ],
}

st.navigation(pages, position="sidebar").run()
