import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# --- Load model and scalers ---
model = load_model("lstm_model.h5")
feature_scaler = joblib.load("feature_scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")

SEQ_LEN = 60  # sequence length used in training
feature_columns = ['Close', 'High', 'Low', 'Open', 'Volume', 'MA5', 'MA20',
                   'Return', 'MA50', 'Volatility20', 'Lag1', 'Lag2', 'Lag3']

st.title("CSPX Next-Day Close Price Prediction")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload historical CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Last 5 rows of uploaded data:")
    st.dataframe(df.tail())

    # --- Prepare features ---
    features = df[feature_columns].values
    features_scaled = feature_scaler.transform(features)

    # --- Take last SEQ_LEN rows for prediction ---
    last_seq = features_scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, len(feature_columns))

    # --- Predict ---
    y_pred_scaled = model.predict(last_seq)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)

    st.subheader("Predicted Next-Day Close Price")
    st.write(f"${y_pred.flatten()[0]:.2f}")
