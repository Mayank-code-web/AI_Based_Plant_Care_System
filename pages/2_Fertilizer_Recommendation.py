import streamlit as st
from utils.theme import apply_base_theme

# ---------- LOGIN PROTECTION ---------- #

if not st.session_state.get("logged_in", False):
    st.warning("Please login first")
    st.rerun()

# st.set_page_config is called once, in the router (0_home.py).

apply_base_theme()

import pandas as pd
import pickle
import joblib
from datetime import datetime
from garden_db import add_diagnosis

st.sidebar.markdown("<h2 style='color: #ffffff;'>📌  Description</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p class='sidebar-text'>Our AI-powered Fertilizer Recommendation System uses content-based similarity to analyze plant-care products and recommend the most relevant alternatives.</p>", unsafe_allow_html=True)


# Cache loading of models and data for faster execution
@st.cache_resource()
def load_models():
    with open('models/fertilizer_model/product_dict.pkl', 'rb') as file:
        product_dict = pickle.load(file)
    similarity = joblib.load('models/fertilizer_model/similarity.joblib')

    return pd.DataFrame(product_dict), similarity

@st.cache_resource()
def load_description_data():
    return pd.read_csv('data/fertilizer_recommendation/fertilizer.csv')

# Load models and data
products, similarity = load_models()
description_data = load_description_data()

# Recommendation function
@st.cache_data()
def recommend(product):
    try:
        product_index = products[products['Product_Name'] == product].index[0]
    except IndexError:
        return []
    distances = similarity[product_index]
    products_list = sorted(enumerate(distances), reverse=True, key=lambda x: x[1])[1:6]
    return [products.iloc[i[0]].Product_Name for i in products_list]

# Title
st.markdown("""
    <h1 style='text-align: center; color: #1F5233; font-size: 36px;'>🧪 Fertilizer & Plant-Care Product Recommendation</h1>
""", unsafe_allow_html=True)

# Searchbox
st.markdown("""
    <h3 style='color: #1F5233;'>🔍 Find Similar Products:</h3>
""", unsafe_allow_html=True)

col1, col2 = st.columns([4, 1])
with col1:
    selected_product_name = st.selectbox('Select a product:', sorted(products['Product_Name'].values))
with col2:
    # Matches the selectbox's label height so the button sits level with the
    # input field itself, not the label above it.
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    recommend_btn = st.button('🔍 Recommend Product', use_container_width=True)

# CSS styling
st.markdown("""
    <style>
        div[data-testid="stButton"] > button {
            width: 100%;
            background-color: #2F7A4D;
            color: white;
            border-radius: 10px;
            padding: 8px;
            font-size: 16px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Display Description
desc = description_data.loc[description_data['Product_Name'] == selected_product_name, 'Description']
category = description_data.loc[description_data['Product_Name'] == selected_product_name, 'Category']
if not desc.empty:
    st.markdown(f"""
        <p style='background-color: #2F7A4D; color: #ffffff; padding: 12px; border-radius: 10px; font-size: 16px;'>
        <strong>Category:</strong> {category.values[0]}<br>
        <strong>Description:</strong> {desc.values[0]}</p>
    """, unsafe_allow_html=True)

# Display Recommendations
if recommend_btn:
    recommendations = recommend(selected_product_name)
    if recommendations:
        add_diagnosis(st.session_state.username, {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "type": "Fertilizer Recommendation",
            "summary": selected_product_name,
            "details": ", ".join(recommendations),
        })

        st.markdown("""
            <h3 style='color: #1F5233;'>📌 Recommended Products:</h3>
        """, unsafe_allow_html=True)

        for product in recommendations:
            product_category = description_data.loc[
                description_data['Product_Name'] == product, 'Category'
            ].values
            product_category = product_category[0] if len(product_category) else ""
            st.markdown(f"""
                <p style='font-size:18px; border:2px solid #E4DCC8; padding: 10px; border-radius:10px; display: flex; justify-content: space-between;'>
                    <span style='color:#1F5233; font-weight:bold;'>{product}</span>
                    <span style='background-color: #D97A4D; color: white; padding: 5px 10px; border-radius: 5px;'>
                    {product_category}</span>
                </p>
            """, unsafe_allow_html=True)
    else:
        st.error("No recommendations found. Please select another product.")

st.caption(
    "Formulation guidance only — always check a product's own label for the recommended dose "
    "and application frequency for your specific plant."
)
