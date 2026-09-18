from garden_db import add_diagnosis
import streamlit as st
from utils.theme import apply_base_theme
from utils.vision import diagnose_image
from utils.env import load_env
from collections import Counter

# Ensure the project-root .env is loaded before the first image-analysis click.
# Streamlit can execute this page independently during navigation.
load_env()
import pickle
import json
import pandas as pd
import numpy as np
import ast
from urllib.parse import quote_plus
import os
from datetime import datetime


# ---------- LOGIN PROTECTION ---------- #
# (belt-and-suspenders: the router in 0_home.py only lists this page for
# logged-in users, so this should rarely trigger)

if not st.session_state.get("logged_in", False):
    st.warning("Please login first")
    st.rerun()


# st.set_page_config is called once, in the router (0_home.py).

apply_base_theme()


# ---------- SIDEBAR ---------- #

st.sidebar.markdown("<h2 style='color: #ffffff;'>📌 Description</h2>", unsafe_allow_html=True)

st.sidebar.markdown(
"""
<p class='sidebar-text'>
The Plant Symptom Diagnosis tool uses AI to analyze the symptoms you're seeing,
suggest a likely plant health issue, and recommend a care plan to help your plant recover.
</p>
""",
unsafe_allow_html=True
)


# ---------- LOAD DATA ---------- #

@st.cache_resource
def load_data():

    try:

        sym_des = pd.read_csv("data/plant_care_dataset/symptoms_df.csv")
        care_actions = pd.read_csv("data/plant_care_dataset/care_actions.csv")
        care_routine = pd.read_csv("data/plant_care_dataset/care_routine.csv")
        description = pd.read_csv("data/plant_care_dataset/description.csv")
        treatments = pd.read_csv("data/plant_care_dataset/treatments.csv")
        soil_nutrition = pd.read_csv("data/plant_care_dataset/soil_nutrition.csv")

        model = pickle.load(open("models/symptom_model/RandomForest.pkl", "rb"))

        with open("models/symptom_model/symptoms_list.json") as f:
            symptoms_list = json.load(f)

        with open("models/symptom_model/issues_list.json") as f:
            issues_list = {int(k): v for k, v in json.load(f).items()}

        return sym_des, care_actions, care_routine, description, treatments, soil_nutrition, model, symptoms_list, issues_list

    except Exception as e:

        st.error(f"Error loading data: {e}")
        return None, None, None, None, None, None, None, None, None


(sym_des, care_actions, care_routine, description, treatments,
 soil_nutrition, model, symptoms_list, issues_list) = load_data()

issue_names = list(description['Issue'].unique())

symptoms_list_processed = {
    symptom.replace("_", " ").lower(): value
    for symptom, value in symptoms_list.items()
}


# ---------- FUNCTIONS ---------- #

def information(predicted_issue):

    try:

        issue_description = description.loc[
            description['Issue'] == predicted_issue, 'Description'].values[0]

        issue_care_actions = care_actions.loc[
            care_actions['Issue'] == predicted_issue,
            ['Care_Action_1', 'Care_Action_2', 'Care_Action_3', 'Care_Action_4']
        ].values.flatten().tolist()

        issue_treatments = ast.literal_eval(
            treatments.loc[treatments['Issue'] == predicted_issue, 'Treatment'].values[0])

        issue_soil = ast.literal_eval(
            soil_nutrition.loc[soil_nutrition['Issue'] == predicted_issue, 'Soil_Nutrition'].values[0])

        issue_routine = care_routine.loc[
            care_routine['issue'] == predicted_issue, 'routine'].values.tolist()

        return issue_description, issue_care_actions, issue_treatments, issue_soil, issue_routine

    except Exception:

        return "Description not available", [], [], [], []


def predicted_value(observed_symptoms):

    try:

        input_vector = np.zeros(len(symptoms_list_processed))

        for symptom in observed_symptoms:

            if symptom in symptoms_list_processed:
                input_vector[symptoms_list_processed[symptom]] = 1

        prediction = model.predict([input_vector])[0]

        return issues_list.get(prediction, "Unknown Issue")

    except Exception as e:

        import traceback
        traceback.print_exc()
        st.error(f"Prediction error: {e}")
        return "Prediction Error"


def issue_image_path(issue_name):
    """Path to the small code-drawn illustration for this issue, if one exists."""
    slug = issue_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join("issue_images", f"{slug}.png")
    return path if os.path.exists(path) else None


def show_issue_illustration(issue_name):
    img_path = issue_image_path(issue_name)
    if img_path:
        st.image(img_path, caption=issue_name, width=320)


def render_issue_card(issue_name):
    """Prints the full description/actions/treatments/soil/routine block for one issue."""
    issue_desc, care_acts, care_treatments, soil_tips, routine = information(issue_name)

    show_issue_illustration(issue_name)
    st.write(f"Description: {issue_desc}")
    st.write("Recommended Actions:", ", ".join(str(x) for x in care_acts if x))

    st.write("Possible Treatments:")
    if care_treatments:
        for treatment in care_treatments:
            treatment = str(treatment).strip()
            if not treatment:
                continue
            search_url = "https://www.google.com/search?q=" + quote_plus(
                f"{treatment} plant treatment product"
            )
            st.markdown(f"- **{treatment}**  [🔗 Find treatment/product]({search_url})")
    else:
        st.write("No specific treatment listed.")

    st.write("Soil & Nutrition:", ", ".join(str(x) for x in soil_tips if x))
    st.write("Ongoing Care Routine:", ", ".join(str(x) for x in routine if x))


# ---------- UI ---------- #

st.title("🌿 Plant Symptom Diagnosis & Care Recommendation")

st.markdown("### Diagnosis Based on Observed Symptoms")

st.markdown("_Select the symptoms your plant is showing._")

user_symptoms = st.multiselect(
    "Select Symptoms",
    list(symptoms_list_processed.keys())
)


# ---------- PREDICT ---------- #

if st.button("Diagnose Plant"):

    if user_symptoms:

        predicted_issue = predicted_value(user_symptoms)

        add_diagnosis(st.session_state.username, {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "type": "Symptom Diagnosis",
            "issue": predicted_issue,
            "symptoms": user_symptoms
        })

        issue_desc, care_acts, care_treatments, soil_tips, routine = information(predicted_issue)

        st.success(f"Suggested Issue: {predicted_issue}")

        show_issue_illustration(predicted_issue)

        st.write(f"Description: {issue_desc}")

        st.write("Recommended Actions:", ", ".join(str(x) for x in care_acts if x))

        st.write("Possible Treatments:")
        if care_treatments:
            for treatment in care_treatments:
                treatment = str(treatment).strip()
                if not treatment:
                    continue
                search_url = "https://www.google.com/search?q=" + quote_plus(
                    f"{treatment} plant treatment product"
                )
                st.markdown(
                    f"- **{treatment}**  [🔗 Find treatment/product]("
                    f"{search_url})"
                )
        else:
            st.write("No specific treatment listed.")

        st.write("Soil & Nutrition:", ", ".join(str(x) for x in soil_tips if x))

        st.write("Ongoing Care Routine:", ", ".join(str(x) for x in routine if x))

        st.caption(
            "This is a general, educational suggestion based on the symptoms you selected — "
            "not a substitute for advice from a local nursery or plant specialist, especially "
            "for valuable or rare plants."
        )

    else:

        st.warning("Please select at least one symptom.")


st.markdown("---")


# ---------- SEARCH ISSUE ---------- #

st.markdown("### Search for an Issue Description")

issue_query = st.text_input(
    "Type an issue name to get recommendations:",
    placeholder="Start typing..."
)


if issue_query:

    matches = [
        d for d in issue_names
        if d.lower().startswith(issue_query.lower())
    ]

    if matches:

        selected_issue = matches[0]

        issue_desc, care_acts, care_treatments, soil_tips, routine = information(selected_issue)

        st.subheader(f"Recommendations for {selected_issue}")

        show_issue_illustration(selected_issue)

        st.write(f"Description: {issue_desc}")

        st.write("Recommended Actions:", ", ".join(str(x) for x in care_acts if x))

        st.write("Possible Treatments:")
        if care_treatments:
            for treatment in care_treatments:
                treatment = str(treatment).strip()
                if not treatment:
                    continue
                search_url = "https://www.google.com/search?q=" + quote_plus(
                    f"{treatment} plant treatment product"
                )
                st.markdown(
                    f"- **{treatment}**  [🔗 Find treatment/product]("
                    f"{search_url})"
                )
        else:
            st.write("No specific treatment listed.")

        st.write("Soil & Nutrition:", ", ".join(str(x) for x in soil_tips if x))

        st.write("Ongoing Care Routine:", ", ".join(str(x) for x in routine if x))

    else:

        st.warning("No matching issue found.")


st.markdown("---")


# ---------- IMAGE UPLOAD DIAGNOSIS ---------- #

st.markdown("### 📸 Diagnose from Photo(s)")
st.markdown(
    "_Upload one or more clear photos of the affected leaves/stem — of the same plant, from "
    "different angles or spots — and get a combined AI-suggested diagnosis and cure._"
)

uploaded_images = st.file_uploader(
    "Upload photo(s) of the problem",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

if uploaded_images:

    st.write(f"**{len(uploaded_images)} photo(s) selected**")
    thumb_cols = st.columns(min(len(uploaded_images), 4))
    for i, img_file in enumerate(uploaded_images):
        with thumb_cols[i % len(thumb_cols)]:
            st.image(img_file, caption=img_file.name, use_container_width=True)

    if st.button(f"Analyze {'Photo' if len(uploaded_images) == 1 else f'{len(uploaded_images)} Photos'}"):

        api_key = os.environ.get("GROQ_API_KEY") or st.session_state.get("groq_api_key_runtime")

        if not api_key:
            st.error(
                "This feature needs a Groq API key. Add `GROQ_API_KEY` to a `.env` file in the "
                "project root (see `.env.example`), or set it once on the PlantBot page, then try again."
            )
        else:
            per_photo_results = []

            progress = st.progress(0.0, text="Analyzing photos…")
            for i, img_file in enumerate(uploaded_images):
                result = diagnose_image(
                    api_key=api_key,
                    image_bytes=img_file.getvalue(),
                    filename=img_file.name,
                    known_issues=issue_names,
                )
                per_photo_results.append({"filename": img_file.name, **result})
                progress.progress((i + 1) / len(uploaded_images), text=f"Analyzed {i + 1}/{len(uploaded_images)} photos…")
            progress.empty()

            # ---- combine results across all photos ---- #
            successful = [r for r in per_photo_results if not r["error"]]
            failed = [r for r in per_photo_results if r["error"]]
            matched_counts = Counter(r["matched_issue"] for r in successful if r["matched_issue"])

            consensus_issue = None
            consensus_count = 0
            if matched_counts:
                consensus_issue, consensus_count = matched_counts.most_common(1)[0]

            add_diagnosis(st.session_state.username, {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "type": "Image Diagnosis",
                "issue": consensus_issue or "Unmatched / see notes",
                "summary": (
                    f"{consensus_issue} ({consensus_count}/{len(successful)} photos)"
                    if consensus_issue else f"No clear match across {len(successful)} photo(s)"
                ),
                "symptoms": [],
                "details": f"{len(uploaded_images)} photo(s) analyzed, {len(failed)} failed",
            })

            st.markdown("## 🧾 Combined Diagnosis")

            if len(uploaded_images) > 1 and matched_counts:
                # show the full vote breakdown when there's more than one distinct opinion
                if len(matched_counts) > 1:
                    st.write("**Votes across photos:** " + ", ".join(
                        f"{issue} ({count})" for issue, count in matched_counts.most_common()
                    ))

            if failed:
                with st.expander(f"⚠️ {len(failed)} photo(s) couldn't be analyzed"):
                    for r in failed:
                        st.write(f"- **{r['filename']}**: {r['error']}")

            if consensus_issue:
                agreement_note = (
                    f"{consensus_count} of {len(successful)} analyzed photo(s) point to this issue."
                    if len(successful) > 1 else "Based on the uploaded photo."
                )
                st.success(f"Most likely issue: {consensus_issue}")
                st.caption(agreement_note)
                render_issue_card(consensus_issue)
            elif successful:
                st.info("No confident match to a known issue category across the uploaded photo(s).")
            else:
                st.error("None of the uploaded photos could be analyzed — see the error details above.")

            if successful:
                with st.expander("See the AI's notes for each individual photo"):
                    for r in successful:
                        st.markdown(f"**{r['filename']}** — {r['matched_issue'] or 'no confident match'}")
                        if r["raw_response"]:
                            st.write(r["raw_response"])
                        st.markdown("---")

            st.caption(
                "AI photo analysis can be wrong, especially with unclear or poorly lit photos — "
                "this is general, educational guidance, not a substitute for a local nursery or "
                "plant specialist, especially for valuable or rare plants."
            )
