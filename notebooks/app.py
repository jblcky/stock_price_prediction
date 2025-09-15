import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from pathlib import Path
from typing import Tuple, Optional, Any

# --- Constants & Configuration ---
# Use pathlib for robust path handling
BASE_DIR = Path("notebooks")
MODEL_PATH = BASE_DIR / "lstm_model_v0.h5"
FEATURE_SCALER_PATH = BASE_DIR / "feature_scaler_v0.pkl"
TARGET_SCALER_PATH = BASE_DIR / "target_scaler_v0.pkl"

# Centralize sequence length and feature columns to avoid repetition
SEQ_LEN = 60
FEATURE_COLUMNS = [
    "Close", "High", "Low", "Open", "Volume", "MA5", "MA20",
    "Return", "MA50", "Volatility20", "Lag1", "Lag2", "Lag3",
    "Month_sin", "Month_cos"
]

# --- Page Configuration ---
st.set_page_config(
    page_title="CSPX Next-Day Price Predictor",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS (unchanged) ---
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
    .st-emotion-cache-1629p8f {
        color: #4a4a4a;
        text-align: center;
        margin-bottom: 2.5rem;
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
def load_assets() -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """Loads and caches the ML model and scalers."""
    try:
        model = load_model(MODEL_PATH, compile=False)
        feature_scaler = joblib.load(FEATURE_SCALER_PATH)
        target_scaler = joblib.load(TARGET_SCALER_PATH)
        return model, feature_scaler, target_scaler
    except (FileNotFoundError, IOError) as e:
        st.error(f"Error loading model assets: {e}. Please ensure the files are in the '{BASE_DIR}' directory.")
        return None, None, None


def get_prediction(
    uploaded_file,
    model: tf.keras.Model,
    feature_scaler: Any,
    target_scaler: Any
) -> Optional[float]:
    """
    Preprocesses data from an uploaded file and runs a prediction.
    Returns the predicted price or None if an error occurs.
    """
    try:
        # --- 1. Load and Preprocess Data ---
        df = pd.read_csv(uploaded_file, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        # Check for required raw columns before feature engineering
        required_raw_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
        if not required_raw_cols.issubset(df.columns):
            missing = required_raw_cols - set(df.columns)
            st.error(f"The uploaded CSV is missing required columns: {', '.join(missing)}")
            return None

        # --- 2. Feature Engineering ---
        df["MA5"] = df["Close"].rolling(window=5).mean()
        df["MA20"] = df["Close"].rolling(window=20).mean()
        df["MA50"] = df["Close"].rolling(window=50).mean()
        df["Return"] = df["Close"].pct_change()
        df["Volatility20"] = df["Return"].rolling(window=20).std()
        for i in range(1, 4):
            df[f"Lag{i}"] = df["Close"].shift(i)
        df["Month"] = df["Date"].dt.month
        df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

        # Drop rows with NaN values created by rolling/lagging features
        df = df.dropna().reset_index(drop=True)
        df_model = df[FEATURE_COLUMNS]

        # --- 3. Data Validation (after preprocessing) ---
        if len(df_model) < SEQ_LEN:
            st.warning(
                f"Insufficient data after processing ({len(df_model)} rows). "
                f"The model requires at least {SEQ_LEN} historical records to predict."
            )
            return None

        # --- 4. Scale, Predict, and Inverse Transform ---
        features = df_model.values
        features_scaled = feature_scaler.transform(features)
        last_sequence = features_scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, len(FEATURE_COLUMNS))

        prediction_scaled = model.predict(last_sequence)
        prediction = target_scaler.inverse_transform(prediction_scaled)

        return prediction.flatten()[0]

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None


# --- App UI ---
st.title("CSPX Next-Day Close Price")
st.markdown("Upload your historical market data CSV file to predict the next closing price using our trained LSTM model.")

# Load model and scalers once
model, feature_scaler, target_scaler = load_assets()

if all([model, feature_scaler, target_scaler]):
    uploaded_file = st.file_uploader(
        "Upload Historical Data (CSV)",
        type=["csv"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        # Run the entire pipeline
        predicted_price = get_prediction(uploaded_file, model, feature_scaler, target_scaler)

        if predicted_price is not None:
            st.markdown(f"""
            <div class="prediction-card">
                <h2>Predicted Next-Day Close Price</h2>
                <p>${predicted_price:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        # Optional: Show a preview of the uploaded data
        with st.expander("View Uploaded Raw Data (Last 5 Rows)"):
            # Reread the file for display to avoid issues with the file buffer
            df_display = pd.read_csv(uploaded_file)
            st.dataframe(df_display.tail())
