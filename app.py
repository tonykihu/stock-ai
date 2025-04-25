import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from utils.alerts import send_email_alert, send_discord_alert

# Load models
tech_model = joblib.load("models/technical_model.pkl")
hybrid_model = joblib.load("models/hybrid_model.pkl")

# UI
st.title("📈 AI Stock Signal Dashboard")
st.markdown("""
**US & Kenyan Market Predictions**  
*Data updates EOD | Models retrain weekly*
""")

# Inputs
ticker = st.selectbox("Select Ticker", ["AAPL", "MSFT", "SCOM.NSE"])
model_type = st.radio("Model", ["Technical Only", "Hybrid (Technical + Sentiment)"])

# Fetch latest data (mock example)
data = pd.DataFrame({
    "Date": [datetime.now().strftime("%Y-%m-%d")],
    "rsi_14": [32.5],  # Replace with real-time data
    "sma_50": [175.20],
    "sentiment_score": [0.7]
})

# Predict
if model_type == "Technical Only":
    prediction = tech_model.predict(data[["rsi_14", "sma_50"]])
else:
    prediction = hybrid_model.predict(data[["rsi_14", "sma_50", "sentiment_score"]])

# Display
st.metric("Signal", "BUY 🟢" if prediction[0] == 1 else "HOLD/SELL 🔴")
st.line_chart(pd.DataFrame({"Close": [170, 172, 171, 175]}))  # Replace with real data

import yfinance as yf

def fetch_latest_data(ticker):
    if ".NSE" in ticker:  # Kenyan ticker (mock)
        return pd.DataFrame({"rsi_14": [45], "sma_50": [15.50], "sentiment_score": [0.6]})
    else:  # US ticker
        data = yf.download(ticker, period="1mo")
        data["rsi_14"] = data["Close"].rolling(14).apply(lambda x: 100 - (100 / (1 + (x[x > 0].mean() / x[x < 0].mean()))))
        data["sma_50"] = data["Close"].rolling(50).mean()
        return data.iloc[-1:].reset_index()

# Replace mock data in the dashboard
data = fetch_latest_data(ticker)

uploaded_file = st.file_uploader("Upload NSE Kenya Data (CSV)")
if uploaded_file:
    nse_data = pd.read_csv(uploaded_file)
    st.write("Latest NSE Data:", nse_data.tail())

# Feedback
feedback = st.text_area("How can we improve?")
if st.button("Submit Feedback"):
    with open("feedback/log.txt", "a") as f:
        f.write(f"{datetime.now()}: {feedback}\n")
    st.success("Thanks for your input!")

# Alert section
st.subheader("Set Up Alerts")
alert_type = st.radio("Alert Channel", ["Email", "Discord"])

if alert_type == "Email":
    if st.button("Enable Email Alerts"):
        send_email_alert("BUY", "AAPL")  # Replace with real signal
elif alert_type == "Discord":
    webhook_url = st.text_input("Enter Discord Webhook URL")
    if st.button("Test Discord Alert"):
        send_discord_alert("BUY", "AAPL", webhook_url)