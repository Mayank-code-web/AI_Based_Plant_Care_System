from garden_db import save_garden_info, get_garden
import streamlit as st
from utils.theme import apply_base_theme

# Belt-and-suspenders guard: the router (0_home.py) only lists this page for
# logged-in users, but just in case someone lands here directly.
if not st.session_state.get("logged_in", False):
    st.warning("Please login first")
    st.rerun()

apply_base_theme()

# -------- SIDEBAR -------- #

st.sidebar.markdown("<h2>📌 Navigation</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p>Use the sidebar to explore different features of the AI Plant Care System.</p>", unsafe_allow_html=True)

# Logout button
if st.sidebar.button("Logout", use_container_width=True):
    st.session_state.logged_in = False
    st.rerun()


# -------- HERO -------- #

st.markdown("<div class='app-title'>🌿 AI-Powered Plant Care System</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>Helping your plants thrive with AI-driven diagnosis & care guidance</div>", unsafe_allow_html=True)


# -------- GARDEN PROFILE -------- #

st.header("🪴 My Garden Profile")

username = st.session_state.get("username")
garden = get_garden(username)

garden_name = st.text_input("Garden / Collection Name", value=garden["profile"].get("garden_name", ""))
num_plants = st.number_input("Number of Plants", min_value=0, max_value=1000, value=garden["profile"].get("num_plants", 1))
setting = st.selectbox(
    "Where do you mostly grow your plants?",
    ["Indoor", "Outdoor", "Balcony/Patio", "Greenhouse"],
    index=0 if garden["profile"].get("setting") is None else
    ["Indoor", "Outdoor", "Balcony/Patio", "Greenhouse"].index(garden["profile"].get("setting"))
)

if st.button("Save Profile"):
    save_garden_info(username, {
        "garden_name": garden_name,
        "num_plants": num_plants,
        "setting": setting
    })
    st.success("Profile saved successfully!")

st.markdown("---")


# -------- FEATURES -------- #

st.markdown("<h2 style='text-align:center;'>✨ Features</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class='app-card'>
    <h3>🔍 Plant Symptom Diagnosis</h3>
    <p>Select the symptoms you're seeing and get an AI-suggested plant health issue plus a care plan.</p>
    </div>

    <div class='app-card'>
    <h3>🧪 Fertilizer Recommendation</h3>
    <p>Find similar fertilizers and plant-care products based on what you're already using.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='app-card'>
    <h3>📊 Plant Stress Risk Assessment</h3>
    <p>Estimate how at-risk a plant is based on its light, water, soil, and care conditions.</p>
    </div>

    <div class='app-card'>
    <h3>🤖 PlantBot</h3>
    <p>Chat with an AI-powered assistant to get general plant care insights and tips.</p>
    </div>
    """, unsafe_allow_html=True)


st.markdown("---")


# -------- TECHNOLOGIES -------- #

st.markdown("<h2 style='text-align:center;'>⚙️ Technologies Used</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='app-card'>
    <h3>🤖 Machine Learning</h3>
    <p>Utilizing RandomForest classification and content-based similarity models.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='app-card'>
    <h3>🗂 NLP & AI</h3>
    <p>Leveraging Natural Language Processing for chatbot interactions.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='app-card'>
    <h3>☁️ Cloud Computing</h3>
    <p>Deployed using Streamlit Cloud.</p>
    </div>
    """, unsafe_allow_html=True)


st.markdown("---")


# -------- WHY USE THIS APP -------- #

st.markdown("""
<div class='app-card'>
<h2 style='text-align:center;'>🔍 Why Use This App?</h2>
<p>
✅ <b>Helpful Guidance:</b> AI-suggested care plans based on common plant symptoms.<br>
✅ <b>Real-Time Assistance:</b> Get quick insights and recommendations.<br>
✅ <b>User-Friendly Interface:</b> Designed for hobbyists and plant enthusiasts alike.<br>
✅ <b>Low-Stakes by Design:</b> General plant-care guidance only — see the note below.<br>
</p>
</div>
""", unsafe_allow_html=True)

st.info(
    "🌱 **A note on scope:** this app gives general, educational plant-care suggestions. "
    "It isn't a substitute for expert horticultural or agronomic advice, especially for "
    "valuable, rare, or commercial plants — when in doubt, consult a local nursery or "
    "extension service."
)

st.markdown("---")

st.header("📜 Recent Activity")

garden = get_garden(st.session_state.username)

if not garden["history"]:
    st.info("No activity yet — try the Plant Symptom Diagnosis, a photo upload, or the Stress Risk Assessment page.")
else:
    recent = list(reversed(garden["history"]))[:3]
    for item in recent:
        label = item.get("type", "Activity")
        result = item.get("issue", item.get("summary", ""))
        st.write(f"**{label}** — {result}")

if st.button("View full profile & search history →"):
    st.switch_page("pages/5_Profile_Dashboard.py")
