"""
Shared visual theme for the AI-Powered Plant Care System.

"Fern & Terracotta" - a calm, natural palette (warm cream + deep forest green
+ terracotta accent) used consistently across every page. Import and call
`apply_base_theme()` near the top of each page, after `st.set_page_config(...)`.
"""

import streamlit as st

PRIMARY = "#2F7A4D"       # forest green
PRIMARY_DARK = "#1F5233"
ACCENT = "#D97A4D"        # terracotta
BG = "#F8F6EF"            # warm cream
SURFACE = "#FFFFFF"
BORDER = "#E4DCC8"
TEXT = "#1F2937"
MUTED = "#6B7280"

BASE_CSS = f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {TEXT};
}}

h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: {PRIMARY_DARK};
}}

[data-testid="stAppViewContainer"] {{
    background-color: {BG};
}}

[data-testid="stSidebar"] {{
    background-color: {PRIMARY_DARK};
}}

[data-testid="stSidebar"] * {{
    color: #F3F1EC !important;
}}

[data-testid="stSidebar"] .stButton>button {{
    background-color: {ACCENT};
    color: white;
    border: none;
}}

[data-testid="stSidebar"] .sidebar-text {{
    font-size: 16px;
    text-align: center;
    margin-bottom: 20px;
}}

/* Cards */

.app-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 2px 10px rgba(31, 41, 55, 0.05);
    transition: all 0.2s ease-in-out;
}}

.app-card:hover {{
    box-shadow: 0 8px 24px rgba(47, 122, 77, 0.12);
    transform: translateY(-2px);
}}

.app-card h3 {{
    margin-top: 0;
    color: {PRIMARY_DARK};
}}

.app-title {{
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    color: {PRIMARY_DARK};
    margin-bottom: 6px;
}}

.app-subtitle {{
    font-size: 18px;
    text-align: center;
    color: {MUTED};
    font-weight: 400;
    margin-bottom: 30px;
}}

/* Buttons */

.stButton > button {{
    background: linear-gradient(90deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
    transition: all 0.2s ease;
}}

.stButton > button:hover {{
    background: linear-gradient(90deg, {ACCENT} 0%, #B85F35 100%);
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(217, 122, 77, 0.35);
}}

/* Inputs */

.stTextInput input, .stNumberInput input, .stSelectbox > div, textarea {{
    border-radius: 10px !important;
    border: 1px solid {BORDER} !important;
}}

.stTextInput input:focus {{
    border-color: {PRIMARY} !important;
    box-shadow: 0 0 0 2px rgba(47, 122, 77, 0.15) !important;
}}

hr {{
    border-color: {BORDER};
}}

</style>
"""


def apply_base_theme():
    """Inject the shared theme CSS. Call once near the top of every page."""
    st.markdown(BASE_CSS, unsafe_allow_html=True)


def card(title: str, body_html: str) -> str:
    """Return HTML for a themed card. Pass the result to st.markdown(..., unsafe_allow_html=True)."""
    return f"""
    <div class="app-card">
        <h3>{title}</h3>
        <p>{body_html}</p>
    </div>
    """