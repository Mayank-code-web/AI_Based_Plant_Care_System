import streamlit as st
from auth import login_user, register_user
from utils.theme import apply_base_theme, PRIMARY_DARK, BORDER

# NOTE: st.set_page_config is called once, in the router (0_home.py) — a
# second call from this page would raise an error since it's now reached
# via st.navigation rather than run directly.

apply_base_theme()

# ---------- PAGE-SPECIFIC STYLE ---------- #
# Card styling is applied to a REAL st.container(key=...) below (via the
# .st-key-login_card selector), not to a raw unsafe_allow_html <div> — a
# <div> opened in one st.markdown() call can't visually wrap widgets that
# come from later, separate Streamlit calls, which is what caused the
# stray empty white box that used to sit above the Login/Register toggle.
st.markdown(f"""
<style>

[data-testid="stSidebar"] {{ display: none; }}
[data-testid="stSidebarCollapsedControl"] {{ display: none; }}

.block-container {{
    padding-top: 6vh;
    max-width: 480px;
}}

.st-key-login_card {{
    background: #FFFFFF;
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 36px 36px 24px 36px;
    box-shadow: 0 10px 40px rgba(47, 122, 77, 0.12);
}}

div[role="radiogroup"] {{
    gap: 24px;
    margin-bottom: 6px;
}}

</style>
""", unsafe_allow_html=True)

# Already logged in -> go home
if "logged_in" in st.session_state and st.session_state.logged_in:
    st.rerun()

st.markdown(f"""
<h1 style='text-align:center; color:{PRIMARY_DARK}; font-family:"Plus Jakarta Sans", sans-serif; font-size:38px;'>
🌿 AI-Powered Plant Care System
</h1>
<p style='text-align:center; color:#6B7280; margin-bottom: 24px;'>Sign in to continue, or create a new account below</p>
""", unsafe_allow_html=True)

with st.container(key="login_card"):

    option = st.radio("Login or register", ["Login", "Register"], horizontal=True, label_visibility="collapsed")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # ---------- LOGIN ---------- #
    if option == "Login":
        if st.button("Login", use_container_width=True):
            if not username or not password:
                st.warning("Please enter both a username and a password.")
            else:
                try:
                    if login_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success("Login successful! Redirecting…")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                except Exception as e:
                    st.error(f"Something went wrong while logging in: {e}")

    # ---------- REGISTER ---------- #
    if option == "Register":
        if st.button("Create account", use_container_width=True):
            try:
                if register_user(username, password):
                    # Auto-login right after registering so it's obvious it worked
                    st.session_state.logged_in = True
                    st.session_state.username = username.strip()
                    st.success("Account created! Logging you in…")
                    st.rerun()
                else:
                    st.error("That username is already taken. Please choose another, or log in instead.")
            except ValueError as e:
                # raised by register_user for empty username/short password
                st.warning(str(e))
            except Exception as e:
                st.error(f"Something went wrong while registering: {e}")
