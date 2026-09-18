# 🌿 AI-Powered Plant Care System

A Streamlit multi-page app combining four AI-powered plant-care tools:

1. **Plant Symptom Diagnosis & Care Recommendation** — suggests a likely plant health issue from selected symptoms (RandomForest) and returns a description, recommended actions, treatments, soil/nutrition notes, and an ongoing care routine.
2. **Fertilizer Recommendation** — recommends similar fertilizers/plant-care products using content-based (TF-IDF + cosine) similarity.
3. **Plant Stress Risk Assessment** — estimates a plant's risk of stress/decline from its environment and care routine (RandomForest) and explains which factors drove the score.
4. **PlantBot** — a RAG chatbot (LangChain + FAISS + Groq-hosted Llama 3.3) that answers general plant-care questions from a small local knowledge base.

Login/registration is handled with bcrypt-hashed passwords stored in `users.json`; per-user garden profile and diagnosis history are stored in `gardens.json`.

## Why this project changed domains

This project started as a healthcare-prediction app (symptom-based disease prediction, drug recommendation, heart-disease risk scoring). It was rebuilt as a **plant-care** app instead, for one core reason: **accountability**. A tool that mispredicts a human disease, drug interaction, or cardiac risk score can contribute to real harm if anyone relied on it — and a student project realistically can't carry that responsibility (no clinical validation, no liability coverage, no regulatory review). A wrong guess about *why a houseplant's leaves are yellowing* carries no comparable risk to a person's health.

The architecture, the four-module structure, the ML techniques (classification from selected symptoms, content-based recommendation, risk scoring with an explainability step, a RAG chatbot), the auth system, and the UI are all preserved — only the subject matter and the datasets/models changed. That was intentional: it demonstrates the same technical skills (classification, NLP-based similarity, explainable ML, retrieval-augmented generation, full-stack app structure) without the accountability problem that got the original rejected.

**This app still isn't professional advice** — for valuable, rare, or commercial plants, a local nursery, horticulturist, or agricultural extension service is the right resource. But the stakes of an imperfect suggestion are appropriately low for a student project.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The datasets and trained models (RandomForest classifier, TF-IDF similarity matrix, risk pipeline) are already included under `data/` and `models/` — they were generated from scratch by `scripts/build_models.py` (no scraped or copyrighted data). To regenerate them at any time:

```bash
python scripts/build_models.py
```

### PlantBot's knowledge base

PlantBot now builds `vectorstore/db_faiss/` **automatically the first time the PlantBot page is opened** if the index is missing. It reads the bundled text files from `data/plantbot data/` and downloads `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face on the first run.

You can still rebuild the index manually after editing the knowledge files:

```bash
python plantbot/create_memory_for_llm.py
```

See `plantbot/readme.md` for details.

Create a `.env` file in the project root (see `.env.example`) with:

```
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

- `GROQ_API_KEY` — required for the PlantBot page (get one free at console.groq.com).
- `HF_TOKEN` — only needed if you use a Hugging Face-hosted model instead of Groq.

## Run

```bash
streamlit run 0_home.py
```

This opens the login page first; register a new user, then log in to reach the dashboard and the four feature pages in the sidebar.

## Project structure

```
0_home.py                  # entry point / router — st.set_page_config + login-aware st.navigation
auth.py                    # bcrypt register/login helpers
garden_db.py                # profile + history persistence (gardens.json)
pages/
  _login.py                 # login / register UI (only shown when logged out)
  dashboard.py               # garden profile + recent activity (only shown when logged in)
  1_Plant_Symptom_Diagnosis.py
  2_Fertilizer_Recommendation.py
  3_Plant_Stress_Risk_Assessment.py
  4_PlantBot.py
  5_Profile_Dashboard.py    # full profile + searchable/filterable past-activity history
scripts/
  build_models.py            # generates every dataset + trains every model from scratch
  generate_issue_images.py   # draws the 12 original issue illustrations with Pillow
issue_images/               # generated illustrations shown on the diagnosis page
models/                    # trained .pkl / .joblib model files (output of build_models.py)
data/                      # datasets used by the diagnosis/fertilizer pages + PlantBot's text corpus
plantbot/                  # scripts to (re)build the FAISS knowledge base for PlantBot
vectorstore/db_faiss/      # FAISS vector store for PlantBot (build locally, see plantbot/readme.md)
utils/                     # shared CSS theme, env loader, and Groq vision helper
```

Run `python scripts/generate_issue_images.py` once after `build_models.py` to (re)generate the illustrations (already included in this zip, so this is only needed if you regenerate the issues list).

## What changed from the original healthcare version, module by module

| Original (Healthcare) | This version (Plant Care) |
|---|---|
| Disease Prediction & Medical Recommendation (132 symptoms → 41 diseases, RandomForest) | Plant Symptom Diagnosis (25 symptoms → 12 plant issues, RandomForest) |
| Drug Recommendation (content similarity over a real medicine dataset) | Fertilizer Recommendation (content similarity over a synthetic fertilizer/product dataset) |
| Heart Disease Risk Assessment (BRFSS survey data, LightGBM + EasyEnsemble + SHAP) | Plant Stress Risk Assessment (synthetic care-condition data, RandomForest + built-in feature importances) |
| Medibot (RAG over two medical reference PDFs) | PlantBot (RAG over plant-care text notes) |
| `patient_db.py` / `patients.json` | `garden_db.py` / `gardens.json` |
| Patient Profile (name, age, gender) | Garden Profile (garden name, plant count, setting) |

## Recent additions

- **Photo-based diagnosis (multi-photo)**: the Plant Symptom Diagnosis page accepts one or more uploaded photos at once. Each is sent independently to a Groq vision-capable model (see `utils/vision.py`), which tries to match it to one of the app's known issue categories or otherwise gives a free-text description. With multiple photos, results are combined into one **Combined Diagnosis**: the most-voted matching issue is shown with full structured care data (reusing the same data as symptom-based diagnosis), the vote breakdown is shown if opinions differed, and each individual photo's notes are available in an expander. Requires `GROQ_API_KEY`.
- **Generated issue illustrations**: each of the 12 known plant issues now shows a small illustration on the diagnosis page (symptom-based, search, and photo-based results). These are simple, original graphics drawn entirely in code with Pillow (`scripts/generate_issue_images.py`) — not photos or scraped images — so there's nothing to attribute or license.
- **Profile Dashboard** (`pages/5_Profile_Dashboard.py`): a dedicated page showing the garden profile plus a searchable, filterable table of every past symptom diagnosis, photo diagnosis, fertilizer lookup, and stress-risk assessment, each timestamped.
- **Login page fixed**: the previous version had a stray empty white box above the Login/Register toggle, caused by trying to wrap Streamlit widgets in a raw HTML `<div>` (widgets from separate `st.*` calls aren't actually nested inside markdown HTML, so the div rendered as its own empty box). It now uses a real `st.container(key="login_card")`, styled via CSS, which does contain its widgets correctly.
## Known-issue notes

- **Login hidden from the sidebar nav**: an underscore-prefixed filename (`pages/_login.py`) does *not* actually hide a page from Streamlit's automatic folder-based navigation — that only works with the newer `st.navigation()` API. `0_home.py` is now a thin router: it decides which pages exist in the sidebar based on `st.session_state.logged_in`, so a logged-out visitor only ever sees "Login", and a logged-in user never sees it again. See `0_home.py` for details.
- **Pathname collisions with `st.navigation`**: Streamlit infers each page's URL pathname from its filename (dropping any leading number/underscore), and that includes the *entry script itself* — not just the pages passed to `st.navigation`. The entry script here is `0_home.py`, which infers to pathname `home`; the dashboard page therefore can't also be named `pages/home.py` (`StreamlitAPIException: Multiple Pages specified with URL pathname home`), which is why it's named `pages/dashboard.py` instead. If you rename the entry script or any page file, keep an eye out for this.
- **Fertilizer page button alignment**: the "Recommend Product" button now has a small spacer above it so it lines up with the product dropdown itself, not its label.

## Notes on the synthetic data

The original project used real clinical/survey datasets (a symptom-disease dataset, a medicine dataset, and the CDC's BRFSS survey). None of those are appropriate to reuse for a plant-care app, so every dataset here (`data/plant_care_dataset/`, `data/fertilizer_recommendation/`, and the model training data in `scripts/build_models.py`) was authored/generated from scratch specifically for this project. The plant issues, symptoms, and care guidance reflect commonly known, general horticultural patterns (e.g. overwatering causing root rot, low light causing leggy growth) rather than any single copied source — but as with any general guidance, treat it as a starting point, not a definitive diagnosis.
