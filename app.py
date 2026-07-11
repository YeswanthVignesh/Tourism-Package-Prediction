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
    gender = st.selectbox("Gender", options=["Male", "Female"], index=0)
    marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced"], index=0)

with col2:
    number_of_person_visiting = st.number_input("Number of People Visiting", min_value=1, max_value=10, value=2)
    preferred_property_star = st.selectbox("Preferred Property Star", options=[1.0, 2.0, 3.0, 4.0, 5.0], index=2)
    number_of_trips = st.number_input("NumberOfTrips (Annually)", min_value=0, max_value=20, value=1)
    passport = st.selectbox("Has Passport?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
    own_car = st.selectbox("Owns Car?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=1)

with col3:
    number_of_children_visiting = st.number_input("NumberOfChildrenVisiting (Below 5)", min_value=0, max_value=5, value=0)
    designation = st.selectbox("Designation", options=["Manager", "Executive", "Senior Manager", "AVP", "VP", "Director"], index=0)
    monthly_income = st.number_input("MonthlyIncome", min_value=5000, max_value=100000, value=25000)

st.header("Interaction Details")
col4, col5 = st.columns(2)

with col4:
    type_of_contact = st.selectbox("TypeofContact", options=["Self Enquiry", "Company Invited"], index=0)
    product_pitched = st.selectbox("ProductPitched", options=["Deluxe", "Basic", "Standard", "Super Deluxe", "King"], index=0)

with col5:
    duration_of_pitch = st.number_input("DurationOfPitch (minutes)", min_value=1, max_value=60, value=10)
    pitch_satisfaction_score = st.slider("PitchSatisfactionScore", min_value=1, max_value=5, value=3)

# --- Create DataFrame for Prediction ---
def create_input_df():
    data = {
        'Age': [age],
        'TypeofContact': [type_of_contact],
        'CityTier': [city_tier],
        'DurationOfPitch': [duration_of_pitch],
        'Occupation': [occupation],
        'Gender': [gender],
        'NumberOfPersonVisiting': [number_of_person_visiting],
        'ProductPitched': [product_pitched],
        'PreferredPropertyStar': [preferred_property_star],
        'MaritalStatus': [marital_status],
        'NumberOfTrips': [number_of_trips],
        'Passport': [passport],
        'PitchSatisfactionScore': [pitch_satisfaction_score],
        'OwnCar': [own_car],
        'NumberOfChildrenVisiting': [number_of_children_visiting],
        'Designation': [designation],
        'MonthlyIncome': [monthly_income]
    }
    # Note: 'Unnamed: 0' was dropped during data cleaning in the training notebook, so it's not included here.
    # Ensure the order matches the training data features if the model expects specific order (pipeline handles this).
    return pd.DataFrame(data)

input_df = create_input_df()

# --- Make Prediction ---
if st.button("Predict Purchase"): # Moved the predict button to the bottom
    try:
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.success(f"The customer is LIKELY to purchase the Wellness Tourism Package! (Probability: {prediction_proba[1]:.2f})")
        else:
            st.info(f"The customer is UNLIKELY to purchase the Wellness Tourism Package. (Probability: {prediction_proba[1]:.2f})")

        st.write("--- Debug Info ---")
        st.write("Input Data:")
        st.dataframe(input_df)
        # st.write("Prediction Probabilities:", prediction_proba)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure all input fields are correctly filled.")
