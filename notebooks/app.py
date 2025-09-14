import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="CSPX Next-Day Price Predictor",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS for Minimalist Design ---
st.markdown("""
<style>
    /* General Styles */
    body {
        font-family: 'Inter', sans-serif;
    }

    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Title */
    h1 {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    /* Subheader/Description */
    .st-emotion-cache-1629p8f a {
        color: #4B8BBE;
    }

    .st-emotion-cache-1629p8f {
        color: #4a4a4a;
        text-align: center;
        margin-bottom: 2.5rem;
    }

    /* File Uploader */
    .st-emotion-cache-1jicfl2 {
        border: 2px dashed #d0d0d0;
        border-radius: 0.5rem;
        padding: 1.5rem;
        background-color: #fafafa;
    }

    .st-emotion-cache-1jicfl2 .st-emotion-cache-1tpl0xr p{
        font-size: 1rem;
        font-weight: 500;
    }

    /* Dataframe */
    .stDataFrame {
        border: 1px solid #e6e6e6;
        border-radius: 0.5rem;
    }

    /* Prediction Card */
    .prediction-card {
        background-color: #ffffff;
        border-radius: 0.75rem;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid #e6e6e6;
    }

    .prediction-card h2 {
        font-size: 1.25rem;
        color: #4a4a4a;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }

    .prediction-card p {
        font-size: 3rem;
        font-weight: 700;
        color: #007aff;
        margin: 0;
    }

</style>
""", unsafe_allow_html=True)


# --- Caching for Performance ---
@st.cache_resource
def load_prediction_model():
    """Loads and caches the LSTM model."""
    try:
        model = load_model("notebooks/lstm_model.h5", compile=False)
        return model
    except (FileNotFoundError, IOError):
        st.error("Model file 'lstm_model.h5' not found. Please ensure it's in the 'notebooks' directory.")
        return None

@st.cache_resource
def load_scalers():
    """Loads and caches the feature and target scalers."""
    try:
        feature_scaler = joblib.load("notebooks/feature_scaler.pkl")
        target_scaler = joblib.load("notebooks/target_scaler.pkl")
        return feature_scaler, target_scaler
    except (FileNotFoundError, IOError):
        st.error("Scaler files not found in 'notebooks' directory. Please ensure 'feature_scaler.pkl' and 'target_scaler.pkl' are present.")
        return None, None

def run_prediction(df, model, feature_scaler, target_scaler):
    """
    Prepares data and runs prediction on the last sequence.
    Returns the predicted close price.
    """
    SEQ_LEN = 60
    FEATURE_COLUMNS = ['Close', 'High', 'Low', 'Open', 'Volume', 'MA5', 'MA20',
                       'Return', 'MA50', 'Volatility20', 'Lag1', 'Lag2', 'Lag3']

    # --- Data Validation ---
    if not all(col in df.columns for col in FEATURE_COLUMNS):
        st.error("The uploaded CSV is missing one or more required columns.")
        st.info(f"Required columns: {', '.join(FEATURE_COLUMNS)}")
        return None

    if len(df) < SEQ_LEN:
        st.warning(f"Insufficient data. The model requires at least {SEQ_LEN} historical records for an accurate prediction. Found: {len(df)}.")
        return None

    try:
        # --- Prepare Features ---
        features = df[FEATURE_COLUMNS].values
        features_scaled = feature_scaler.transform(features)

        # --- Get Last Sequence for Prediction ---
        last_sequence = features_scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, len(FEATURE_COLUMNS))

        # --- Predict ---
        prediction_scaled = model.predict(last_sequence)
        prediction = target_scaler.inverse_transform(prediction_scaled)

        return prediction.flatten()[0]

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None


# --- App UI ---
st.title("CSPX Next-Day Close Price")
st.markdown("Upload your historical market data CSV file to predict the next closing price using our trained LSTM model.")

# --- Load Model and Scalers ---
model = load_prediction_model()
feature_scaler, target_scaler = load_scalers()

if model and feature_scaler and target_scaler:
    uploaded_file = st.file_uploader(
        "Upload Historical Data",
        type=["csv"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            with st.expander("View Uploaded Data (Last 5 Rows)"):
                 st.dataframe(df.tail())

            # --- Run Prediction ---
            predicted_price = run_prediction(df, model, feature_scaler, target_scaler)

            if predicted_price is not None:
                st.markdown(f"""
                <div class="prediction-card">
                    <h2>Predicted Next-Day Close Price</h2>
                    <p>${predicted_price:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Failed to process the uploaded file. Please ensure it is a valid CSV. Error: {e}")
