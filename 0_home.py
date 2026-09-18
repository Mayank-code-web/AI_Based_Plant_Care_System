"""
App entry point.

This file is intentionally thin: it owns `st.set_page_config` (which can
only be called once per app) and decides WHICH pages are visible in the
sidebar, based on login state, using Streamlit's `st.navigation` API.

Why this exists instead of the older "put files in pages/ and Streamlit
auto-lists them" approach: with plain folder-based auto-discovery, every
file in pages/ always shows in the sidebar for every user, logged in or
not — including the login page itself once you're already logged in, and
protected pages before you've logged in at all. `st.navigation` lets us
build a *different* nav list per rerun, so a logged-out visitor only ever
sees "Login", and a logged-in user never sees "Login" again.
"""

import streamlit as st
from utils.theme import apply_base_theme
from utils.env import load_env

# Load project-root .env before any page is executed. This prevents the
# first-run "missing GROQ_API_KEY" problem caused by Streamlit page navigation.
load_env()

st.set_page_config(
    page_title="AI-Powered Plant Care System",
    page_icon="🌿",
    layout="wide",
)

apply_base_theme()

logged_in = st.session_state.get("logged_in", False)

if logged_in:
    pages = [
        st.Page("pages/dashboard.py", title="Home", icon="🏠", default=True),
        st.Page("pages/1_Plant_Symptom_Diagnosis.py", title="Plant Symptom Diagnosis", icon="🔍"),
        st.Page("pages/2_Fertilizer_Recommendation.py", title="Fertilizer Recommendation", icon="🧪"),
        st.Page("pages/3_Plant_Stress_Risk_Assessment.py", title="Plant Stress Risk Assessment", icon="📊"),
        st.Page("pages/4_PlantBot.py", title="PlantBot", icon="🤖"),
        st.Page("pages/5_Profile_Dashboard.py", title="Profile Dashboard", icon="🧑‍🌾"),
    ]
else:
    pages = [
        st.Page("pages/_login.py", title="Login", icon="🔐", default=True),
    ]

pg = st.navigation(pages, position="sidebar")
pg.run()
