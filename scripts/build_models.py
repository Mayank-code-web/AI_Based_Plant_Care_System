"""
Builds every dataset + trained model used by the app, from scratch, offline.

Run once from the project root:
    python scripts/build_models.py

Everything here is generated programmatically (no scraped/copyrighted data),
so it's safe to commit and safe to regenerate at any time.
"""

import os
import json
import random
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

random.seed(42)
np.random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =========================================================================
# 1) PLANT SYMPTOM -> ISSUE DIAGNOSIS MODEL
#    (mirrors the old "symptoms -> disease" RandomForest classifier)
# =========================================================================

SYMPTOMS = [
    "yellowing_leaves", "brown_leaf_spots", "wilting", "leaf_curling",
    "white_powdery_coating", "black_sooty_residue", "holes_in_leaves",
    "sticky_leaf_residue", "stunted_growth", "root_rot_smell", "leaf_drop",
    "mold_on_soil_surface", "fine_webbing_on_leaves", "distorted_new_growth",
    "pale_leaves", "dry_crispy_leaf_edges", "leggy_stretched_growth",
    "no_flowering", "mushy_stem_base", "visible_tiny_insects",
    "cottony_white_clusters", "leaf_scorch", "droopy_leaves",
    "slow_growth", "waterlogged_soil",
]

ISSUES = {
    "Powdery Mildew": ["white_powdery_coating", "leaf_curling", "stunted_growth"],
    "Root Rot": ["root_rot_smell", "wilting", "mushy_stem_base", "waterlogged_soil", "yellowing_leaves"],
    "Spider Mite Infestation": ["fine_webbing_on_leaves", "pale_leaves", "leaf_drop", "visible_tiny_insects"],
    "Aphid Infestation": ["sticky_leaf_residue", "distorted_new_growth", "visible_tiny_insects", "black_sooty_residue"],
    "Fungal Leaf Spot": ["brown_leaf_spots", "leaf_drop", "yellowing_leaves"],
    "Nitrogen Deficiency": ["yellowing_leaves", "slow_growth", "pale_leaves"],
    "Overwatering Stress": ["waterlogged_soil", "wilting", "yellowing_leaves", "mold_on_soil_surface"],
    "Underwatering Stress": ["dry_crispy_leaf_edges", "droopy_leaves", "wilting", "leaf_scorch"],
    "Light Deficiency (Etiolation)": ["leggy_stretched_growth", "no_flowering", "pale_leaves", "slow_growth"],
    "Sooty Mold": ["black_sooty_residue", "sticky_leaf_residue", "leaf_drop"],
    "Bacterial Leaf Blight": ["brown_leaf_spots", "leaf_scorch", "wilting", "leaf_drop"],
    "Mealybug Infestation": ["cottony_white_clusters", "sticky_leaf_residue", "stunted_growth", "leaf_curling"],
}

ISSUE_LIST = sorted(ISSUES.keys())
ISSUE_INDEX = {name: i for i, name in enumerate(ISSUE_LIST)}
SYMPTOM_INDEX = {s: i for i, s in enumerate(SYMPTOMS)}


def make_symptom_training_rows(rows_per_issue=60):
    rows = []
    labels = []
    for issue, core_symptoms in ISSUES.items():
        for _ in range(rows_per_issue):
            vec = np.zeros(len(SYMPTOMS), dtype=int)
            # core symptoms mostly present (some noise / missed reporting)
            for s in core_symptoms:
                if random.random() < 0.85:
                    vec[SYMPTOM_INDEX[s]] = 1
            # small chance of 1-2 unrelated symptoms too (real-world noise)
            if random.random() < 0.3:
                extra = random.choice(SYMPTOMS)
                vec[SYMPTOM_INDEX[extra]] = 1
            rows.append(vec)
            labels.append(ISSUE_INDEX[issue])
    return np.array(rows), np.array(labels)


def build_symptom_model():
    X, y = make_symptom_training_rows()
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    out_dir = os.path.join(ROOT, "models", "symptom_model")
    with open(os.path.join(out_dir, "RandomForest.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(out_dir, "symptoms_list.json"), "w") as f:
        json.dump(SYMPTOM_INDEX, f, indent=2)

    with open(os.path.join(out_dir, "issues_list.json"), "w") as f:
        json.dump({str(v): k for k, v in ISSUE_INDEX.items()}, f, indent=2)

    # ---- supporting reference tables (description / care / treatment / etc.) ----
    data_dir = os.path.join(ROOT, "data", "plant_care_dataset")

    descriptions = {
        "Powdery Mildew": "A fungal disease that coats leaves and stems in a white, powdery layer. Thrives in humid air with poor circulation.",
        "Root Rot": "Roots decay from prolonged waterlogging or fungal pathogens, cutting off the plant's ability to take up water and nutrients.",
        "Spider Mite Infestation": "Tiny sap-sucking arachnids that thrive in hot, dry conditions and leave fine webbing between leaves.",
        "Aphid Infestation": "Small soft-bodied insects that cluster on new growth and buds, feeding on plant sap and secreting honeydew.",
        "Fungal Leaf Spot": "A group of fungal pathogens causing discrete brown or black spots on foliage, often spreading in wet conditions.",
        "Nitrogen Deficiency": "A lack of available nitrogen in the soil, most visible as pale, uniformly yellowing older leaves.",
        "Overwatering Stress": "Roots sit in saturated soil for too long, starving them of oxygen and inviting rot-causing fungi.",
        "Underwatering Stress": "Prolonged dryness causes the plant to pull moisture from its own leaves, leading to crisping and wilting.",
        "Light Deficiency (Etiolation)": "Insufficient light causes the plant to stretch toward any light source, producing weak, elongated growth.",
        "Sooty Mold": "A black fungus that grows on the sugary honeydew left behind by sap-sucking pests like aphids or scale.",
        "Bacterial Leaf Blight": "A bacterial infection that causes water-soaked lesions which turn brown and can spread rapidly in wet, warm weather.",
        "Mealybug Infestation": "Small, soft insects covered in a white waxy, cotton-like coating that cluster in leaf joints and undersides.",
    }

    care_actions = {
        "Powdery Mildew": ["Improve air circulation around the plant", "Avoid wetting the leaves when watering", "Remove and discard heavily infected leaves", "Space plants further apart"],
        "Root Rot": ["Unpot and trim away black, mushy roots", "Repot into fresh, well-draining soil", "Reduce watering frequency going forward", "Ensure the pot has drainage holes"],
        "Spider Mite Infestation": ["Rinse leaves thoroughly under running water", "Raise humidity around the plant", "Isolate the plant from others nearby", "Inspect weekly for a couple of weeks"],
        "Aphid Infestation": ["Wipe or spray off aphids with a strong stream of water", "Prune heavily infested new growth", "Introduce or attract natural predators like ladybugs", "Recheck every few days until clear"],
        "Fungal Leaf Spot": ["Remove and dispose of affected leaves", "Avoid overhead watering", "Improve airflow between plants", "Sterilize pruning tools between cuts"],
        "Nitrogen Deficiency": ["Feed with a balanced, nitrogen-rich fertilizer", "Check that soil pH is in a normal range", "Avoid overfeeding, which can burn roots", "Refresh depleted potting mix"],
        "Overwatering Stress": ["Let the topsoil dry out before watering again", "Check the pot has proper drainage", "Move to a location with better air circulation", "Avoid saucers that trap standing water"],
        "Underwatering Stress": ["Water thoroughly until it drains from the bottom", "Set a consistent watering reminder", "Group with a humidity tray if in dry air", "Trim back severely scorched leaf tips"],
        "Light Deficiency (Etiolation)": ["Move closer to a bright, indirect light source", "Rotate the plant regularly for even growth", "Consider a grow light in darker rooms", "Prune leggy stems to encourage fuller growth"],
        "Sooty Mold": ["Treat the underlying pest problem first", "Gently wipe mold off leaves with a damp cloth", "Improve airflow around the plant", "Recheck for returning pests after cleaning"],
        "Bacterial Leaf Blight": ["Remove infected foliage promptly", "Avoid working with the plant when leaves are wet", "Water at the soil line, not overhead", "Space plants to reduce humidity buildup"],
        "Mealybug Infestation": ["Dab visible clusters with rubbing alcohol on a cotton swab", "Rinse the plant well afterward", "Check leaf joints and undersides carefully", "Isolate from other plants until resolved"],
    }

    treatments = {
        "Powdery Mildew": ["Neem oil spray", "Potassium bicarbonate spray", "Sulfur-based fungicide"],
        "Root Rot": ["Fungicidal root soak", "Fresh well-draining potting mix", "Cinnamon as a natural antifungal on cut roots"],
        "Spider Mite Infestation": ["Insecticidal soap", "Neem oil spray", "Predatory mites (biological control)"],
        "Aphid Infestation": ["Insecticidal soap", "Neem oil spray", "Ladybug release (biological control)"],
        "Fungal Leaf Spot": ["Copper-based fungicide", "Neem oil spray"],
        "Nitrogen Deficiency": ["Balanced NPK liquid fertilizer", "Compost top-dressing"],
        "Overwatering Stress": ["Root rot fungicide (only if rot is present)", "Well-draining amended soil mix"],
        "Underwatering Stress": ["Deep soak watering", "Mulch to retain soil moisture"],
        "Light Deficiency (Etiolation)": ["Full-spectrum grow light", "Reflective plant saucer/tray"],
        "Sooty Mold": ["Insecticidal soap for the source pest", "Mild soapy water wipe-down"],
        "Bacterial Leaf Blight": ["Copper-based bactericide", "Removal of infected debris"],
        "Mealybug Infestation": ["Isopropyl alcohol spot treatment", "Neem oil spray", "Insecticidal soap"],
    }

    soil_nutrition = {
        "Powdery Mildew": ["Well-draining soil", "Avoid excess nitrogen which encourages soft growth"],
        "Root Rot": ["Chunky, fast-draining mix (add perlite/bark)", "Avoid compacted, water-retentive soil"],
        "Spider Mite Infestation": ["Standard balanced potting mix", "Keep soil evenly moist to raise ambient humidity"],
        "Aphid Infestation": ["Standard balanced potting mix", "Avoid over-fertilizing with nitrogen"],
        "Fungal Leaf Spot": ["Well-draining soil", "Top-dress with compost"],
        "Nitrogen Deficiency": ["Nitrogen-rich balanced fertilizer", "Organic compost amendment"],
        "Overwatering Stress": ["Chunky, fast-draining mix (add perlite/bark)", "Terracotta pots to aid evaporation"],
        "Underwatering Stress": ["Moisture-retentive mix (add coco coir)", "Mulching the topsoil"],
        "Light Deficiency (Etiolation)": ["Standard balanced potting mix", "Balanced fertilizer once growth resumes"],
        "Sooty Mold": ["Standard balanced potting mix", "No special amendment needed"],
        "Bacterial Leaf Blight": ["Well-draining soil", "Avoid overhead irrigation"],
        "Mealybug Infestation": ["Standard balanced potting mix", "Avoid excess nitrogen"],
    }

    care_routine = {
        "Powdery Mildew": ["Water at soil level, early in the day", "Increase spacing/airflow weekly"],
        "Root Rot": ["Water only when top 2 inches of soil are dry", "Check roots at each repot"],
        "Spider Mite Infestation": ["Mist or humidify a few times a week", "Wipe leaves monthly to deter mites"],
        "Aphid Infestation": ["Inspect new growth weekly", "Prune spent flowers/buds regularly"],
        "Fungal Leaf Spot": ["Water at soil level only", "Remove fallen leaf debris promptly"],
        "Nitrogen Deficiency": ["Feed every 2-4 weeks during growing season", "Flush soil occasionally to prevent salt buildup"],
        "Overwatering Stress": ["Check soil moisture before every watering", "Use a moisture meter if unsure"],
        "Underwatering Stress": ["Water on a consistent schedule", "Increase frequency in hot/dry seasons"],
        "Light Deficiency (Etiolation)": ["Track hours of direct/indirect light received", "Reposition seasonally as light angle changes"],
        "Sooty Mold": ["Wipe leaves periodically", "Monitor for returning pests"],
        "Bacterial Leaf Blight": ["Water early in the day so foliage dries fast", "Sanitize tools between plants"],
        "Mealybug Infestation": ["Inspect leaf joints weekly", "Quarantine new plants before introducing them"],
    }

    pd.DataFrame({"Issue": list(descriptions.keys()), "Description": list(descriptions.values())}) \
        .to_csv(os.path.join(data_dir, "description.csv"), index=False)

    rows = []
    for issue, acts in care_actions.items():
        acts = (acts + [""] * 4)[:4]
        rows.append([issue] + acts)
    pd.DataFrame(rows, columns=["Issue", "Care_Action_1", "Care_Action_2", "Care_Action_3", "Care_Action_4"]) \
        .to_csv(os.path.join(data_dir, "care_actions.csv"), index=False)

    pd.DataFrame({"Issue": list(treatments.keys()), "Treatment": [str(v) for v in treatments.values()]}) \
        .to_csv(os.path.join(data_dir, "treatments.csv"), index=False)

    pd.DataFrame({"Issue": list(soil_nutrition.keys()), "Soil_Nutrition": [str(v) for v in soil_nutrition.values()]}) \
        .to_csv(os.path.join(data_dir, "soil_nutrition.csv"), index=False)

    rows = []
    for issue, routines in care_routine.items():
        for r in routines:
            rows.append([issue, r])
    pd.DataFrame(rows, columns=["issue", "routine"]).to_csv(os.path.join(data_dir, "care_routine.csv"), index=False)

    # symptoms reference table (for the "search by issue" view, mirrors old symptoms_df.csv)
    sym_rows = []
    for issue, syms in ISSUES.items():
        padded = (syms + [""] * 4)[:4]
        sym_rows.append([issue] + padded)
    pd.DataFrame(sym_rows, columns=["Issue", "Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4"]) \
        .to_csv(os.path.join(data_dir, "symptoms_df.csv"), index=False)

    print(f"[symptom model] trained on {X.shape[0]} rows, {len(ISSUE_LIST)} issues, {len(SYMPTOMS)} symptoms")


# =========================================================================
# 2) FERTILIZER / PLANT-CARE PRODUCT RECOMMENDATION
#    (mirrors the old content-based "drug recommendation" similarity model)
# =========================================================================

FERTILIZERS = [
    ("GreenGro All-Purpose 10-10-10", "Granular", "balanced npk slow release granular fertilizer for general houseplant and garden feeding all purpose"),
    ("GreenGro Bloom Booster 5-15-10", "Granular", "phosphorus rich bloom booster granular fertilizer for flowering plants and vegetables encourages blooms"),
    ("LeafyLife Liquid Balanced 10-10-10", "Liquid", "balanced liquid npk fertilizer fast acting for houseplants easy to dilute with water weekly feeding"),
    ("LeafyLife Nitrogen Boost 20-5-5", "Liquid", "high nitrogen liquid fertilizer for leafy green growth lawns and foliage plants promotes lush leaves"),
    ("RootRight Root Stimulator 2-10-8", "Liquid", "phosphorus potassium root stimulator liquid for transplants and new cuttings encourages root development"),
    ("SoilSafe Slow-Release Pellets 14-14-14", "Granular", "slow release coated pellets balanced npk feeds plants steadily over several months low burn risk"),
    ("BloomBright Orchid Formula 6-30-30", "Liquid", "specialty orchid fertilizer high phosphorus potassium for epiphytic flowering plants weak dose weekly"),
    ("VeggiePower Tomato & Vegetable 5-10-10", "Granular", "vegetable garden fertilizer formulated for tomatoes peppers and fruiting plants supports fruit set"),
    ("CactusCare Succulent Formula 2-7-7", "Liquid", "low nitrogen light feeding formula for cacti and succulents diluted liquid prevents leggy growth"),
    ("CitrusShine Fruit Tree Blend 8-3-9", "Granular", "fruit tree and citrus fertilizer with micronutrients magnesium iron zinc for healthy fruiting trees"),
    ("EverGreen Lawn Feed 24-0-10", "Granular", "high nitrogen lawn fertilizer promotes thick green turf grass low phosphorus formula"),
    ("MicroMend Micronutrient Mix", "Liquid", "chelated micronutrient supplement iron manganese zinc copper corrects deficiency yellowing"),
    ("OrganoBoost Fish Emulsion 5-1-1", "Liquid", "organic fish emulsion liquid fertilizer gentle nitrogen source improves soil microbial life"),
    ("OrganoBoost Seaweed Extract", "Liquid", "organic seaweed kelp extract liquid biostimulant improves stress tolerance and root health"),
    ("OrganoBoost Worm Castings", "Organic Amendment", "organic worm castings soil amendment improves soil structure slow nutrient release beneficial microbes"),
    ("OrganoBoost Compost Tea Concentrate", "Liquid", "organic compost tea concentrate liquid soil drench improves microbial diversity in soil"),
    ("HydroGrow A+B Base Nutrients", "Liquid", "two part hydroponic base nutrient solution balanced npk with micronutrients for soilless growing"),
    ("HydroGrow Cal-Mag Supplement", "Liquid", "calcium magnesium supplement for hydroponic and coco coir growing prevents blossom end rot"),
    ("EpsomEase Magnesium Sulfate", "Mineral Supplement", "magnesium sulfate mineral supplement corrects magnesium deficiency yellowing between leaf veins"),
    ("IronFix Chelated Iron Spray", "Liquid", "chelated iron foliar spray corrects iron deficiency chlorosis interveinal yellowing on new growth"),
    ("MycoRoot Mycorrhizal Inoculant", "Soil Amendment", "beneficial mycorrhizal fungi inoculant powder improves root nutrient and water uptake at planting"),
    ("SlowFeed Osmocote-Style Granules 15-9-12", "Granular", "controlled release resin coated granules feeds continuously for several months minimal maintenance"),
    ("PalmPro Palm & Tropical Blend 8-2-12", "Granular", "specialty fertilizer for palms and tropical plants with magnesium manganese prevents frizzle top"),
    ("RoseRadiance Rose Food 4-6-2", "Granular", "specialty rose fertilizer with added calcium and sulfur promotes blooms and disease resistance"),
    ("HerbHarvest Culinary Herb Feed 3-4-6", "Liquid", "gentle balanced liquid feed for culinary herbs preserves flavor oils low nitrogen formula"),
    ("BerryBoost Acid Lover Blend 4-3-6", "Granular", "acidifying fertilizer for blueberries azaleas and other acid loving plants lowers soil ph"),
    ("SeedStart Starter Solution 4-10-3", "Liquid", "gentle starter solution for seedlings and new transplants low burn risk phosphorus for roots"),
    ("TurfTech Winterizer 10-0-20", "Granular", "high potassium lawn winterizer fertilizer builds cold and drought tolerance before dormancy"),
    ("HouseplantHero Complete Feed 3-1-2", "Liquid", "gentle everyday liquid fertilizer for all houseplants safe for frequent light feeding"),
    ("BambooBoost Silica Supplement", "Mineral Supplement", "silica supplement strengthens cell walls improves pest and disease resistance in many plant types"),
]


def build_fertilizer_model():
    names = [f[0] for f in FERTILIZERS]
    categories = [f[1] for f in FERTILIZERS]
    descriptions = [f[2] for f in FERTILIZERS]

    df = pd.DataFrame({"Product_Name": names, "Category": categories, "Description": descriptions})

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["Description"])
    similarity = cosine_similarity(tfidf_matrix)

    out_dir = os.path.join(ROOT, "models", "fertilizer_model")
    joblib.dump(similarity, os.path.join(out_dir, "similarity.joblib"))

    product_dict = df.to_dict("list")
    with open(os.path.join(out_dir, "product_dict.pkl"), "wb") as f:
        pickle.dump(product_dict, f)

    data_dir = os.path.join(ROOT, "data", "fertilizer_recommendation")
    readable_descriptions = [d.replace("_", " ").capitalize() + "." for d in descriptions]
    pd.DataFrame({
        "Product_Name": names,
        "Category": categories,
        "Description": readable_descriptions,
    }).to_csv(os.path.join(data_dir, "fertilizer.csv"), index=False)

    print(f"[fertilizer model] {len(names)} products, similarity matrix {similarity.shape}")


# =========================================================================
# 3) PLANT STRESS / DECLINE RISK ASSESSMENT MODEL
#    (mirrors the old lifestyle-inputs -> heart disease risk model)
# =========================================================================

PLANT_TYPES = ["tropical_foliage", "succulent_or_cactus", "flowering", "fern", "herb", "vegetable"]
LIGHT = ["low_light", "medium_indirect_light", "bright_indirect_light", "direct_sunlight"]
WATERING_FREQ = ["daily", "every_2_3_days", "weekly", "every_2_weeks", "rarely"]
HUMIDITY = ["low_humidity", "moderate_humidity", "high_humidity"]
SOIL = ["well_draining_mix", "standard_potting_mix", "dense_garden_soil", "waterlogged_soil"]
DRAINAGE = ["good_drainage_holes", "poor_or_no_drainage"]
FERTILIZING = ["never", "rarely", "monthly", "every_2_weeks_growing_season"]
TEMP_RANGE = ["cold_below_12C", "cool_12_18C", "ideal_18_27C", "hot_above_27C"]
REPOT_TIME = ["repotted_this_year", "1_2_years_ago", "2_5_years_ago", "5_plus_years_or_never"]
PEST_PRESENCE = ["yes", "no"]
LEAF_SYMPTOM_COUNT = ["none", "one_or_two", "three_or_more"]
INDOOR_OUTDOOR = ["indoor", "outdoor", "both_seasonal"]

RISK_FEATURE_COLUMNS = [
    "plant_type", "light_exposure", "watering_frequency", "humidity_level",
    "soil_type", "pot_drainage", "fertilizing_frequency", "temperature_range",
    "time_since_last_repot", "pest_presence", "leaf_symptom_count", "indoor_or_outdoor",
]


def _sample_row():
    return {
        "plant_type": random.choice(PLANT_TYPES),
        "light_exposure": random.choice(LIGHT),
        "watering_frequency": random.choice(WATERING_FREQ),
        "humidity_level": random.choice(HUMIDITY),
        "soil_type": random.choice(SOIL),
        "pot_drainage": random.choice(DRAINAGE),
        "fertilizing_frequency": random.choice(FERTILIZING),
        "temperature_range": random.choice(TEMP_RANGE),
        "time_since_last_repot": random.choice(REPOT_TIME),
        "pest_presence": random.choice(PEST_PRESENCE),
        "leaf_symptom_count": random.choice(LEAF_SYMPTOM_COUNT),
        "indoor_or_outdoor": random.choice(INDOOR_OUTDOOR),
    }


def _risk_score(row):
    """Hand-authored ground-truth rule used only to *label* synthetic training data."""
    score = 0.0
    if row["soil_type"] == "waterlogged_soil":
        score += 0.25
    if row["pot_drainage"] == "poor_or_no_drainage":
        score += 0.20
    if row["watering_frequency"] == "daily" and row["plant_type"] == "succulent_or_cactus":
        score += 0.20
    if row["watering_frequency"] == "rarely" and row["plant_type"] in ("fern", "tropical_foliage"):
        score += 0.15
    if row["light_exposure"] == "low_light" and row["plant_type"] in ("flowering", "vegetable", "succulent_or_cactus"):
        score += 0.15
    if row["light_exposure"] == "direct_sunlight" and row["plant_type"] == "fern":
        score += 0.10
    if row["humidity_level"] == "low_humidity" and row["plant_type"] in ("tropical_foliage", "fern"):
        score += 0.10
    if row["fertilizing_frequency"] == "never":
        score += 0.08
    if row["temperature_range"] in ("cold_below_12C", "hot_above_27C"):
        score += 0.10
    if row["time_since_last_repot"] == "5_plus_years_or_never":
        score += 0.08
    if row["pest_presence"] == "yes":
        score += 0.15
    if row["leaf_symptom_count"] == "three_or_more":
        score += 0.20
    elif row["leaf_symptom_count"] == "one_or_two":
        score += 0.08

    score += np.random.normal(0, 0.05)  # measurement noise
    return float(np.clip(score, 0, 1))


def build_risk_model(n_rows=4000):
    rows = [_sample_row() for _ in range(n_rows)]
    df = pd.DataFrame(rows)
    scores = df.apply(_risk_score, axis=1)
    df["at_risk"] = (scores > 0.45).astype(int)

    X = df[RISK_FEATURE_COLUMNS]
    y = df["at_risk"]

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), RISK_FEATURE_COLUMNS)]
    )
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)),
    ])
    pipeline.fit(X, y)

    out_dir = os.path.join(ROOT, "models", "risk_model")
    with open(os.path.join(out_dir, "risk_pipeline.pkl"), "wb") as f:
        pickle.dump(pipeline, f)

    with open(os.path.join(out_dir, "feature_options.json"), "w") as f:
        json.dump({
            "plant_type": PLANT_TYPES, "light_exposure": LIGHT, "watering_frequency": WATERING_FREQ,
            "humidity_level": HUMIDITY, "soil_type": SOIL, "pot_drainage": DRAINAGE,
            "fertilizing_frequency": FERTILIZING, "temperature_range": TEMP_RANGE,
            "time_since_last_repot": REPOT_TIME, "pest_presence": PEST_PRESENCE,
            "leaf_symptom_count": LEAF_SYMPTOM_COUNT, "indoor_or_outdoor": INDOOR_OUTDOOR,
        }, f, indent=2)

    train_acc = pipeline.score(X, y)
    print(f"[risk model] trained on {n_rows} synthetic rows, train accuracy={train_acc:.3f}, positive rate={y.mean():.2f}")


if __name__ == "__main__":
    build_symptom_model()
    build_fertilizer_model()
    build_risk_model()
    print("\nAll models + datasets built successfully.")
