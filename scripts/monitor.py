import os
import streamlit as st
import pandas as pd
import plotly.express as px

def load_logs(log_path="logs/performance.csv"):
    """Load performance logs, with error handling."""
    if not os.path.exists(log_path):
        st.error(f"Log file not found: {log_path}")
        st.info("Run the training pipeline first to generate performance logs.")
        st.stop()

    logs = pd.read_csv(log_path)

    required_cols = ["date", "accuracy", "last_updated"]
    missing = [col for col in required_cols if col not in logs.columns]
    if missing:
        st.error(f"Missing columns in log file: {missing}")
        st.stop()

    return logs

# Dashboard
st.title("System Health Monitor")

logs = load_logs()

st.metric("Model Accuracy", f"{logs['accuracy'].iloc[-1]:.2%}")
st.metric("Data Freshness", logs["last_updated"].iloc[-1])

# Plot accuracy over time
fig = px.line(logs, x="date", y="accuracy", title="Model Accuracy Trend")
st.plotly_chart(fig)
