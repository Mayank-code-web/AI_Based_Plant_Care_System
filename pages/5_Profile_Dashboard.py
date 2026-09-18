import streamlit as st
from utils.theme import apply_base_theme
from garden_db import get_garden
import pandas as pd

# ---------- LOGIN PROTECTION ---------- #

if not st.session_state.get("logged_in", False):
    st.warning("Please login first")
    st.rerun()

# st.set_page_config is called once, in the router (0_home.py).
apply_base_theme()

st.sidebar.markdown("<h2 style='color: #ffffff;'>📌 Description</h2>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<p class='sidebar-text'>Your Profile Dashboard: garden details and a full history of every "
    "diagnosis, photo analysis, and risk assessment you've run.</p>",
    unsafe_allow_html=True
)

username = st.session_state.get("username")
garden = get_garden(username)
profile = garden.get("profile", {})
history = garden.get("history", [])

st.title("🧑‍🌾 Profile Dashboard")

# ---------- PROFILE SUMMARY ---------- #

col1, col2, col3, col4 = st.columns(4)
col1.metric("Username", username or "—")
col2.metric("Garden Name", profile.get("garden_name") or "—")
col3.metric("Number of Plants", profile.get("num_plants", "—"))
col4.metric("Setting", profile.get("setting") or "—")

st.caption("Update these details anytime from the Home page.")

st.markdown("---")

# ---------- SEARCH / DIAGNOSIS HISTORY ---------- #

st.header("📜 Past Search History")

if not history:
    st.info("No activity yet — try Plant Symptom Diagnosis, a photo upload, or the Stress Risk Assessment page.")
else:
    # Newest first
    ordered = list(reversed(history))

    type_options = sorted({entry.get("type", "Other") for entry in ordered})
    selected_types = st.multiselect("Filter by type", type_options, default=type_options)

    rows = []
    for entry in ordered:
        entry_type = entry.get("type", "Other")
        if entry_type not in selected_types:
            continue
        rows.append({
            "Date": entry.get("timestamp", "—"),
            "Type": entry_type,
            "Result": entry.get("issue", entry.get("summary", "—")),
            "Details": ", ".join(entry.get("symptoms", [])) if entry.get("symptoms") else entry.get("details", ""),
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(rows)} of {len(history)} total entries.")
    else:
        st.warning("No entries match the selected filter.")
