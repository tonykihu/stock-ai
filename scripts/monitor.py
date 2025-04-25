import streamlit as st
import pandas as pd
import plotly.express as px

# Load log data
logs = pd.read_csv("logs/performance.csv")

# Dashboard
st.title("📊 System Health Monitor")
st.metric("Model Accuracy", f"{logs['accuracy'].iloc[-1]:.2%}")
st.metric("Data Freshness", logs["last_updated"].iloc[-1])

# Plot accuracy over time
fig = px.line(logs, x="date", y="accuracy", title="Model Accuracy Trend")
st.plotly_chart(fig)