"""Build and load the PlantBot FAISS knowledge base.

The app can use this module during deployment, so a missing local FAISS index
is built automatically from the text files in data/plantbot data/.
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "plantbot data"
DB_FAISS_PATH = PROJECT_ROOT / "vectorstore" / "db_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _index_exists() -> bool:
    """Return True only when both FAISS files needed for loading exist."""
    return (
        (DB_FAISS_PATH / "index.faiss").is_file()
        and (DB_FAISS_PATH / "index.pkl").is_file()
    )


def build_vectorstore() -> FAISS:
    """Build the FAISS store from the bundled text knowledge base."""
    if not DATA_PATH.is_dir():
        raise FileNotFoundError(f"PlantBot data folder not found: {DATA_PATH}")

    txt_files = sorted(DATA_PATH.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt knowledge files found in: {DATA_PATH}")

    loader = DirectoryLoader(
        str(DATA_PATH),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    if not documents:
        raise RuntimeError("PlantBot knowledge files were found, but no documents could be loaded.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise RuntimeError("PlantBot knowledge files produced no text chunks.")

    # Explicit cache location makes Hugging Face downloads survive Streamlit
    # reruns and keeps the app independent of the directory it was launched from.
    cache_dir = PROJECT_ROOT / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    DB_FAISS_PATH.mkdir(parents=True, exist_ok=True)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(DB_FAISS_PATH))
    return db


def load_or_build_vectorstore() -> FAISS:
    """Load the existing store, or build it automatically when missing."""
    cache_dir = PROJECT_ROOT / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    if not _index_exists():
        DB_FAISS_PATH.mkdir(parents=True, exist_ok=True)
        # Build with the same embedding model instance to avoid downloading it twice.
        loader = DirectoryLoader(
            str(DATA_PATH),
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        documents = loader.load()
        if not documents:
            raise RuntimeError(f"No PlantBot knowledge files found in {DATA_PATH}")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        if not chunks:
            raise RuntimeError("PlantBot knowledge files produced no text chunks.")
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(str(DB_FAISS_PATH))
        return db

    return FAISS.load_local(
        str(DB_FAISS_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )
