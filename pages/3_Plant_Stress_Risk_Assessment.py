import streamlit as st
from utils.theme import apply_base_theme
from garden_db import add_diagnosis

# ---------- LOGIN PROTECTION ---------- #

if not st.session_state.get("logged_in", False):
    st.warning("Please login first")
    st.rerun()

import pandas as pd
import numpy as np
import pickle as pkl
import json
import plotly.express as px
import warnings
from datetime import datetime
warnings.simplefilter(action='ignore', category=UserWarning)


# Load the pickled pipeline (preprocessing + RandomForest, trained in scripts/build_models.py)
with open('models/risk_model/risk_pipeline.pkl', 'rb') as model_file:
    pipeline = pkl.load(model_file)

with open('models/risk_model/feature_options.json') as f:
    OPTIONS = json.load(f)

# st.set_page_config is called once, in the router (0_home.py).
apply_base_theme()
st.sidebar.markdown("<h2>📌  Description</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p class='sidebar-text'>This tool analyzes a plant's environment and care routine using machine learning to estimate its risk of stress or decline, and gives personalized recommendations.</p>", unsafe_allow_html=True)


row0_0, row0_1, row0_2, row0_3 = st.columns((0.08, 6, 3, 0.17))
with row0_1:
    st.title("AI-Powered Plant Stress Risk Assessment")
    st.write("Estimate how at-risk a plant is, based on its light, water, soil, and care conditions")
st.write('---')

st.markdown("""
<div class="app-card">
    <h2>Introduction</h2>
    <p>The AI-Powered Plant Stress Risk Assessment tool provides a risk score and actionable
    recommendations to help you catch care problems before they seriously harm a plant.</p>
</div>
<div class="app-card">
    <h2>How it works</h2>
    <ul>
        <li><strong>Your Input:</strong> Enter the plant's type, light exposure, watering routine, soil, and general conditions.</li>
        <li><strong>Model:</strong> A RandomForest model trained on typical plant-care patterns estimates a risk score.</li>
        <li><strong>Explanation:</strong> See which factors contributed most to the score.</li>
        <li><strong>Recommendations:</strong> Get practical suggestions to reduce the risk.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.write('---')

# ---------- USER INPUT ---------- #

row1_0, row1_1, row1_2, row1_3, row1_5 = st.columns((0.08, 3, 3, 3, 0.17))
with row1_1:
    st.write("#### Plant & Environment")
row2_0, row2_1, row2_2, row2_3, row2_5 = st.columns((0.08, 3, 3, 3, 0.17))

plant_type = row2_1.selectbox("What type of plant is it?", OPTIONS["plant_type"], index=0)
indoor_or_outdoor = row2_2.selectbox("Where does it grow?", OPTIONS["indoor_or_outdoor"], index=0)
light_exposure = row2_3.selectbox("What light does it get?", OPTIONS["light_exposure"], index=1)

row3_0, row3_1, row3_2, row3_3, row3_5 = st.columns((0.08, 3, 3, 3, 0.17))
with row3_1:
    st.write("#### Water & Soil")
row4_0, row4_1, row4_2, row4_3, row4_5 = st.columns((0.08, 3, 3, 3, 0.17))

watering_frequency = row4_1.selectbox("How often is it watered?", OPTIONS["watering_frequency"], index=2)
humidity_level = row4_1.selectbox("What's the typical humidity?", OPTIONS["humidity_level"], index=1)
soil_type = row4_2.selectbox("What soil/potting mix is it in?", OPTIONS["soil_type"], index=1)
pot_drainage = row4_2.selectbox("Does the pot have drainage?", OPTIONS["pot_drainage"], index=0)
temperature_range = row4_3.selectbox("Typical temperature range?", OPTIONS["temperature_range"], index=2)

row5_0, row5_1, row5_2, row5_3, row5_5 = st.columns((0.08, 3, 3, 3, 0.17))
with row5_1:
    st.write("#### Care History & Symptoms")
row6_0, row6_1, row6_2, row6_3, row6_5 = st.columns((0.08, 3, 3, 3, 0.17))

fertilizing_frequency = row6_1.selectbox("How often is it fertilized?", OPTIONS["fertilizing_frequency"], index=2)
time_since_last_repot = row6_1.selectbox("When was it last repotted?", OPTIONS["time_since_last_repot"], index=1)
pest_presence = row6_2.selectbox("Any pests currently visible?", OPTIONS["pest_presence"], index=1)
leaf_symptom_count = row6_2.selectbox("How many leaf symptoms are visible?", OPTIONS["leaf_symptom_count"], index=0)

# Collect input data
FEATURE_COLUMNS = [
    "plant_type", "light_exposure", "watering_frequency", "humidity_level",
    "soil_type", "pot_drainage", "fertilizing_frequency", "temperature_range",
    "time_since_last_repot", "pest_presence", "leaf_symptom_count", "indoor_or_outdoor",
]

input_data = {
    "plant_type": plant_type,
    "light_exposure": light_exposure,
    "watering_frequency": watering_frequency,
    "humidity_level": humidity_level,
    "soil_type": soil_type,
    "pot_drainage": pot_drainage,
    "fertilizing_frequency": fertilizing_frequency,
    "temperature_range": temperature_range,
    "time_since_last_repot": time_since_last_repot,
    "pest_presence": pest_presence,
    "leaf_symptom_count": leaf_symptom_count,
    "indoor_or_outdoor": indoor_or_outdoor,
}

FRIENDLY_NAMES = {
    "plant_type": "Plant Type", "light_exposure": "Light Exposure",
    "watering_frequency": "Watering Frequency", "humidity_level": "Humidity",
    "soil_type": "Soil Type", "pot_drainage": "Pot Drainage",
    "fertilizing_frequency": "Fertilizing", "temperature_range": "Temperature",
    "time_since_last_repot": "Time Since Repot", "pest_presence": "Pest Presence",
    "leaf_symptom_count": "Leaf Symptoms", "indoor_or_outdoor": "Indoor/Outdoor",
}


def predict_risk(input_data, pipeline):
    input_df = pd.DataFrame([input_data])[FEATURE_COLUMNS]
    risk = pipeline.predict_proba(input_df)[:, 1][0] * 100
    return risk, input_df


st.write('---')
row8_0, row8_1, row8_2, row8_5 = st.columns((0.08, 7, 5, 0.27))

with row8_1:
    st.write("#### AI Plant Stress Risk Assessment")

btn1 = row8_1.button('Get Risk Assessment')

if btn1:
    try:
        risk, input_df = predict_risk(input_data, pipeline)

        add_diagnosis(st.session_state.username, {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "type": "Stress Risk Assessment",
            "summary": f"{risk:.1f}% risk ({plant_type.replace('_', ' ')})",
            "details": f"light={light_exposure}, water={watering_frequency}, soil={soil_type}",
        })

        with row8_1:
            st.write(f"Predicted Plant Stress Risk: {risk:.2f}%")

            # ---- feature importance, aggregated back to the original 12 questions ---- #
            preprocessor = pipeline.named_steps["preprocess"]
            rf_model = pipeline.named_steps["model"]
            encoded_names = preprocessor.get_feature_names_out()
            importances = rf_model.feature_importances_

            importance_df = pd.DataFrame({"encoded_feature": encoded_names, "importance": importances})
            # encoded names look like "cat__plant_type_succulent_or_cactus" -> recover original column
            importance_df["original_feature"] = importance_df["encoded_feature"].apply(
                lambda x: next((c for c in FEATURE_COLUMNS if x.startswith(f"cat__{c}_")), "other")
            )
            agg = importance_df.groupby("original_feature")["importance"].sum().sort_values(ascending=False)
            agg = (agg / agg.sum() * 100).round(2)

            recommendations = []
            if risk > 70:
                recommendations.append("This plant's risk of stress/decline is very high. Here are some suggestions:")
            elif risk > 40:
                recommendations.append("This plant's risk of stress/decline is elevated. Here are some suggestions:")
            elif risk > 25:
                recommendations.append("This plant's risk of stress/decline is moderate. Here are some suggestions:")
            else:
                recommendations.append("This plant's risk of stress/decline is low. Keep up the current care routine.")

            st.write(recommendations[0])

            tip_map = {
                ("soil_type", "waterlogged_soil"): "Repot into a fast-draining mix (add perlite or bark) to prevent root rot.",
                ("pot_drainage", "poor_or_no_drainage"): "Move to a pot with drainage holes, or drill some if possible.",
                ("light_exposure", "low_light"): "If the plant type prefers more light, move it closer to a bright window.",
                ("humidity_level", "low_humidity"): "Consider a humidity tray, pebble tray, or humidifier nearby.",
                ("fertilizing_frequency", "never"): "Start a light, regular feeding schedule during the growing season.",
                ("time_since_last_repot", "5_plus_years_or_never"): "Consider repotting to refresh depleted soil and check the roots.",
                ("pest_presence", "yes"): "Treat visible pests promptly with insecticidal soap or neem oil.",
                ("leaf_symptom_count", "three_or_more"): "Multiple visible symptoms suggest checking the Plant Symptom Diagnosis page for a specific issue.",
                ("watering_frequency", "daily"): "Daily watering may be excessive for this plant type — check soil moisture before each watering.",
                ("watering_frequency", "rarely"): "Infrequent watering may be too little for this plant type — check soil moisture more often.",
                ("temperature_range", "cold_below_12C"): "Move to a warmer spot away from cold drafts or windows.",
                ("temperature_range", "hot_above_27C"): "Move away from direct heat sources or intense afternoon sun.",
            }

            shown = 0
            for feature in agg.index:
                key = (feature, str(input_data.get(feature)))
                if key in tip_map and shown < 6:
                    st.write(f"- **{FRIENDLY_NAMES.get(feature, feature)}** ({agg[feature]:.1f}% of the model's decision): {tip_map[key]}")
                    shown += 1

            if shown == 0:
                st.write("- No single factor stands out — conditions look broadly reasonable for this plant type.")

        with row8_2:
            st.write("#### Contribution to Risk Score")
            pie_df = pd.DataFrame({
                "Feature": [FRIENDLY_NAMES.get(f, f) for f in agg.index],
                "Importance": agg.values,
            })
            fig = px.pie(pie_df, names="Feature", values="Importance")
            st.plotly_chart(fig)

    except Exception as e:
        row8_1.error(e)

    st.write("""
        ###### ***Disclaimer***
        *This tool gives general, educational guidance based on common plant-care patterns. It isn't
        a substitute for advice from a local nursery, horticulturist, or agricultural extension service —
        especially for valuable, rare, or commercial plants.*
    """)
