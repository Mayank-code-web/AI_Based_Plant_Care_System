"""Manually rebuild PlantBot's FAISS knowledge base.

This script is still useful after adding/editing files in data/plantbot data/,
but the Streamlit PlantBot page will also build the index automatically when
it is missing.
"""

import sys
from pathlib import Path

# Make project-root imports work when this file is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from plantbot.knowledge_base import build_vectorstore, DB_FAISS_PATH


if __name__ == "__main__":
    db = build_vectorstore()
    print(f"Vector store saved to {DB_FAISS_PATH}")
