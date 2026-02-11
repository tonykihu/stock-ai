import os
import numpy as np
import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime
from utils.alerts import send_email_alert, send_discord_alert

# --- Load models safely ---
@st.cache_resource
def load_models():
    """Load trained models with error handling."""
    models = {}
    try:
        models["technical"] = joblib.load("models/technical_model.pkl")
    except FileNotFoundError:
        st.warning("Technical model not found. Train it first with: python models/train_technical.py")
    try:
        models["hybrid"] = joblib.load("models/hybrid_model.pkl")
    except FileNotFoundError:
        st.warning("Hybrid model not found. Train it first with: python models/train_hybrid.py")
    return models

models = load_models()

# --- Helper functions ---
def compute_rsi(prices, period=14):
    """Calculate RSI from price series."""
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def fetch_latest_data(ticker):
    """Fetch latest stock data and compute indicators."""
    if ".NSE" in ticker:
        # Kenyan ticker - return mock data (replace with real NSE data source)
        return pd.DataFrame({
            "rsi_14": [45.0],
            "sma_50": [15.50],
            "sentiment_score": [0.6]
        })
    else:
        # US ticker - fetch 3 months for enough data for SMA-50
        data = yf.download(ticker, period="3mo")
        if data.empty:
            st.error(f"No data returned for {ticker}")
            return None

        data["rsi_14"] = compute_rsi(data["Close"])
        data["sma_50"] = data["Close"].rolling(50).mean()
        return data.iloc[-1:].reset_index()

# --- UI ---
st.title("AI Stock Signal Dashboard")
st.markdown("""
**US & Kenyan Market Predictions**
*Data updates EOD | Models retrain weekly*
""")

# Inputs
ticker = st.selectbox("Select Ticker", ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "SCOM.NSE"])
model_type = st.radio("Model", ["Technical Only", "Hybrid (Technical + Sentiment)"])

# Fetch real data
data = fetch_latest_data(ticker)

# Predict
if data is not None:
    if model_type == "Technical Only" and "technical" in models:
        features = data[["rsi_14", "sma_50"]].dropna()
        if not features.empty:
            prediction = models["technical"].predict(features)
            st.metric("Signal", "BUY" if prediction[0] == 1 else "HOLD/SELL")
        else:
            st.warning("Not enough data to compute indicators.")
    elif model_type == "Hybrid (Technical + Sentiment)" and "hybrid" in models:
        # Add a default sentiment score if not present
        if "sentiment_score" not in data.columns:
            data["sentiment_score"] = 0.5  # Neutral default
        features = data[["rsi_14", "sma_50", "sentiment_score"]].dropna()
        if not features.empty:
            prediction = models["hybrid"].predict(features)
            st.metric("Signal", "BUY" if prediction[0] == 1 else "HOLD/SELL")
        else:
            st.warning("Not enough data to compute indicators.")
    else:
        st.info("Model not available. Please train the model first.")

    # Show price chart
    if ".NSE" not in ticker:
        chart_data = yf.download(ticker, period="1mo")
        if not chart_data.empty:
            st.line_chart(chart_data["Close"])

# --- Upload NSE Data ---
st.subheader("Upload NSE Kenya Data")
uploaded_file = st.file_uploader("Upload NSE Kenya Data (CSV)")
if uploaded_file:
    nse_data = pd.read_csv(uploaded_file)
    st.write("Latest NSE Data:", nse_data.tail())

# --- Feedback ---
st.subheader("Feedback")
feedback = st.text_area("How can we improve?")
if st.button("Submit Feedback"):
    os.makedirs("feedback", exist_ok=True)
    with open("feedback/log.txt", "a") as f:
        f.write(f"{datetime.now()}: {feedback}\n")
    st.success("Thanks for your input!")

# --- Alert section ---
st.subheader("Set Up Alerts")
alert_type = st.radio("Alert Channel", ["Email", "Discord"], key="alert_channel")

if alert_type == "Email":
    if st.button("Enable Email Alerts", key="email_alert_btn"):
        send_email_alert("BUY", ticker)
elif alert_type == "Discord":
    webhook_url = st.text_input("Enter Discord Webhook URL")
    if st.button("Test Discord Alert", key="discord_alert_btn"):
        send_discord_alert("BUY", ticker, webhook_url)
