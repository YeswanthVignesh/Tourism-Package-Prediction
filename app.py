import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import HfApi
import os
import numpy as np

# --- Configuration for Model Loading ---
PROCESSED_HF_MODEL_ID = "Yesh0608/data-model" # Derived from previous steps
MODEL_FILENAME = "random_forest_model.joblib"
LOCAL_MODEL_PATH = "random_forest_model.joblib"

# Set page title and layout
st.set_page_config(page_title="Wellness Tourism Package Prediction", layout="centered")

st.title("🌴 Wellness Tourism Package Prediction")
st.write("Enter customer details to predict if they will purchase the Wellness Tourism Package.")

# --- Load Model from Hugging Face Model Hub ---
@st.cache_resource
def load_model_from_hf():
    api = HfApi()
    try:
        # Ensure HF_TOKEN is set as an environment variable or via login for write access
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
             st.warning("HF_TOKEN environment variable not found. Model download might fail if the repository is private.")
             # Fallback to try without token if not set, might work for public repos

        # Download the model file
        api.hf_hub_download(
            repo_id=PROCESSED_HF_MODEL_ID,
            filename=MODEL_FILENAME,
            local_dir=".", # Download to the current directory
            token=hf_token # Pass the token if available
        )
        model = joblib.load(LOCAL_MODEL_PATH)
        st.success("Model loaded successfully from Hugging Face!")
        return model
    except Exception as e:
        st.error(f"Error loading model from Hugging Face: {e}")
        st.stop() # Stop execution if model can't be loaded

model = load_model_from_hf()

# --- Input Features (based on X_train columns, excluding 'Unnamed: 0') ---
# The order and names of these features must match the training data
# and the preprocessing pipeline used in the model.

# Group features for better UI organization
st.header("Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=90, value=35)
    city_tier = st.selectbox("City Tier", options=[1, 2, 3], index=0)
    occupation = st.selectbox("Occupation", options=["Salaried", "Small Business", "Large Business", "Free Lancer", "Government Sector"], index=0)
    gender = st.selectbox("Gender", options=[
