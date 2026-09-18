# PlantBot / Diagnosis Stability Fix

## PlantBot
- Removed the incompatible `reasoning_effort="low"` argument from `ChatGroq`.
  The installed `langchain-groq` Pydantic model rejected that value before any API call.
- Primary model: `openai/gpt-oss-120b`.
- Fallbacks: `openai/gpt-oss-20b`, then `llama-3.1-8b-instant`.
- Keeps response cleanup so accidental `<think>...</think>` markup is not shown.

## Photo Diagnosis
- Uses Groq `qwen/qwen3.6-27b`, a current vision model.
- Uses hidden/non-thinking mode so internal reasoning is not displayed.
- Adds automatic retries for transient rate-limit, timeout, connection, and 5xx failures.
  This is specifically to prevent the "first click fails, second click works" behaviour.
- Permanent 4xx errors are not retried endlessly.

## Important
- Keep your own `.env` file in the project root with `GROQ_API_KEY=...`.
- Restart Streamlit after replacing the project so the new Python files and environment are loaded.

## Multi-Photo Diagnosis
- Plant Symptom Diagnosis page's photo uploader now accepts multiple files at once
  (`accept_multiple_files=True`).
- Each photo is analyzed independently (same `diagnose_image()` call/retries as before).
- Results are combined into one "Combined Diagnosis": majority-vote issue across photos,
  vote breakdown shown when photos disagree, per-photo notes available in an expander,
  and failed photos are reported separately without blocking the rest.
- One consolidated history entry is logged per analysis batch (not one per photo).
