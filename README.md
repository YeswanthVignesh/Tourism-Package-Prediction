%%writefile tourism_project/deployment/README.md
# Wellness Tourism Package Prediction: An MLOps Pipeline

## Project Overview
This project implements an end-to-end MLOps pipeline for predicting customer purchases of a "Wellness Tourism Package." The goal is to automate data preparation, model training, evaluation, and deployment to a Streamlit application, leveraging GitHub for version control and CI/CD, and Hugging Face for data and model hosting.

## Business Context
"Visit with Us," a travel company, aims to efficiently target potential customers for its new Wellness Tourism Package. By automating the prediction of customer purchasing likelihood, the project seeks to improve marketing strategies, reduce manual effort, and drive customer acquisition.

## Data Description
The dataset includes customer demographics and interaction details:
*   **Customer Details:** CustomerID, ProdTaken (target variable), Age, TypeofContact, CityTier, Occupation, Gender, NumberOfPersonVisiting, PreferredPropertyStar, MaritalStatus, NumberOfTrips, Passport, OwnCar, NumberOfChildrenVisiting, Designation, MonthlyIncome.
*   **Customer Interaction Data:** PitchSatisfactionScore, ProductPitched, NumberOfFollowups, DurationOfPitch.

## MLOps Pipeline Stages

The pipeline is structured into several key stages:

1.  **Data Registration:** Raw `tourism.csv` data is loaded, cleaned (handling missing values, dropping `CustomerID`), and then split into training and testing sets. These processed datasets are then uploaded to the Hugging Face Data Hub.
2.  **Model Training & Experiment Tracking:**
    *   Processed data (`train.csv`, `test.csv`) is loaded from Hugging Face.
    *   A `RandomForestClassifier` is used, integrated into a `scikit-learn` pipeline with `OneHotEncoder` for categorical features.
    *   `GridSearchCV` performs hyperparameter tuning.
    *   MLflow tracks experiments, logging parameters, metrics (accuracy, precision, recall, F1-score, ROC AUC), and the best model artifact.
3.  **Model Registration:** The best-performing model is saved locally (`random_forest_model.joblib`) and then uploaded to the Hugging Face Model Hub for versioning and easy access.
4.  **Deployment to Streamlit Cloud:**
    *   A Streamlit web application (`app.py`) is developed to serve predictions.
    *   `requirements.txt` lists all necessary Python dependencies.
    *   The `app.py`, `requirements.txt`, and the `random_forest_model.joblib` are copied into the GitHub repository.
    *   The application is deployed to Streamlit Cloud via GitHub, which automatically installs dependencies and runs the `app.py`.

## Deployment Instructions

### Prerequisites
*   **GitHub Account:** For repository hosting and CI/CD.
*   **Hugging Face Account:** For hosting datasets and models.
*   **GitHub Personal Access Token (PAT):** With `repo` scope to push changes from Colab.
*   **Hugging Face API Token:** With write access to upload datasets and models, stored as a Colab secret `HF_TOKEN`.

### Steps to Deploy

1.  **Clone the Repository:** Clone the GitHub repository locally (or in your Colab environment).
2.  **Prepare `app.py` and `requirements.txt`:** Ensure `app.py` is configured to load the model from Hugging Face and `requirements.txt` lists all necessary dependencies, including specific versions if compatibility issues arise.
3.  **Copy Files:** Copy the `app.py`, `requirements.txt`, and the `random_forest_model.joblib` (downloaded from Hugging Face Model Hub during local execution) into the root of your cloned GitHub repository.
4.  **Commit and Push:** Commit these files to your `main` branch and push them to your GitHub repository.
5.  **Deploy to Streamlit Cloud:**
    *   Go to Streamlit Cloud (share.streamlit.io).
    *   Click "New app" and select your GitHub repository.
    *   Choose the `main` branch and specify `app.py` as the main file path.
    *   Click "Deploy!" Streamlit Cloud will automatically detect `requirements.txt` and install dependencies.

## Repository Structure

```
Tourism-Package-Prediction/
├── tourism_project/
│   ├── data/
│   │   ├── tourism.csv             # Raw data
│   │   ├── train.csv               # Processed training data
│   │   └── test.csv                # Processed testing data
│   ├── model_building/
│   │   └── best_model/
│   │       └── random_forest_model.joblib # Trained model artifact
│   ├── deployment/
│   │   ├── app.py                  # Streamlit application
│   │   ├── requirements.txt        # Python dependencies
│   │   └── Dockerfile              # (Optional) Dockerfile for containerization
│   └── mlruns/                     # MLflow experiment tracking data
├── .github/
│   └── workflows/
│       └── pipeline.yml            # GitHub Actions CI/CD workflow
├── README.md                       # Project documentation (this file)
└── .gitignore
```

## Access the Deployed App

*   **GitHub Repository:** [https://github.com/YeswanthVignesh/Tourism-Package-Prediction.git](https://github.com/YeswanthVignesh/Tourism-Package-Prediction.git)
*   **Streamlit Cloud App:** (Link will appear here once successfully deployed)
