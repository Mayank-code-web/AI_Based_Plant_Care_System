"""
Robust .env loader.

python-dotenv's find_dotenv() searches upward from the CURRENT WORKING
DIRECTORY, which changes depending on how/where you launch `streamlit run`
from (terminal in the project folder vs. an IDE "run" button vs. a
shortcut) — that's the most common reason GROQ_API_KEY / HF_TOKEN "isn't
being picked up" even though the .env file exists and is spelled correctly.

This loads .env relative to the PROJECT ROOT instead (one level above this
file, i.e. one level above /utils), so it works the same way regardless of
where the app was launched from.
"""

import os
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")


def load_env():
    """Load environment variables from the project's .env file. Safe to call multiple times."""
    load_dotenv(ENV_PATH, override=False)
