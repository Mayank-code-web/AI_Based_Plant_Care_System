# PlantBot (RAG chatbot)

PlantBot answers general plant-care questions using retrieval-augmented
generation: your question is matched against a small local knowledge base
(plain-text plant-care notes in `data/plantbot data/`), and the most relevant
passages are given to a Groq-hosted Llama 3.3 model as context.

## Building the knowledge base

The Streamlit PlantBot page now builds the FAISS vector store **automatically on first use** if `vectorstore/db_faiss/` is missing. It downloads the `sentence-transformers/all-MiniLM-L6-v2` embedding model once, reads the `.txt` files in `data/plantbot data/`, and saves the generated index under `vectorstore/db_faiss/`.

You can still rebuild it manually after adding or editing knowledge files:

```bash
pip install -r requirements.txt
python plantbot/create_memory_for_llm.py
```

## Adding more knowledge

Drop additional `.txt` files into `data/plantbot data/` — plant-specific care
sheets, seasonal notes, your own observations — and re-run
`create_memory_for_llm.py`. No code changes needed.

## Testing outside Streamlit

`connect_memory_with_llm.py` is a small CLI script for testing the retrieval
chain directly from the terminal (uses a Hugging Face-hosted model rather
than Groq — set `HF_TOKEN` in `.env` to use it).
