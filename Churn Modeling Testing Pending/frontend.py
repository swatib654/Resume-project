import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Customer Churn Prediction")

# Paths to your saved files
model_path = 'best_model.pkl'
scaler_path = 'scaler.joblib'
encoder_path = 'encoder.joblib'

# Check if files exist
if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(encoder_path):
    st.error("Required files not found. Please ensure 'best_model.pkl', 'scaler.joblib', and 'encoder.joblib' are present.")
else:
    # Load model, scaler, and encoder
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)

    st.title("Customer Churn Prediction")

    # Define features
    numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance',
                          'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    categorical_features = ['Geography', 'Gender']

    # Collect user input
    input_data = {}
    for feature in numerical_features:
        if feature in ['HasCrCard', 'IsActiveMember']:
            input_data[feature] = st.selectbox(f"Enter {feature}", options=["Yes", "No"])
        else:
            input_data[feature] = st.number_input(
                f"Enter {feature}", value=0.0 if feature == 'Balance' else 1.0, step=1.0
            )

    for i, feature in enumerate(categorical_features):
        # Use encoder categories if available
        options = encoder.categories_[i] if hasattr(encoder, "categories_") else []
        input_data[feature] = st.selectbox(f"Enter {feature}", options=options)

    # Prepare numerical data
    numerical_df = pd.DataFrame([{k: input_data[k] for k in numerical_features}])
    numerical_df['HasCrCard'] = numerical_df['HasCrCard'].apply(lambda x: 1 if x == "Yes" else 0)
    numerical_df['IsActiveMember'] = numerical_df['IsActiveMember'].apply(lambda x: 1 if x == "Yes" else 0)
    scaled_data = scaler.transform(numerical_df)  # shape: (1, n_num_features)

    # Prepare categorical data
    categorical_df = pd.DataFrame([{k: input_data[k] for k in categorical_features}])
    encoded_data = encoder.transform(categorical_df)
    if hasattr(encoded_data, "toarray"):  # Convert sparse matrix to dense
        encoded_data = encoded_data.toarray()

    # Combine numerical and categorical features
    final_input = np.hstack([scaled_data, encoded_data])  # shape: (1, total_features)

    # Prediction
    if st.button("Predict"):
        prediction = model.predict(final_input)
        prediction_text = "Yes" if prediction[0] == 1 else "No"
        st.success(f"Churn Prediction: {prediction_text}")

    # Display raw input data
    st.write("### Input Data (before encoding/scaling)")
    st.write(pd.DataFrame([input_data]))
