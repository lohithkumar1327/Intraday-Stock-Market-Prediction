
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from textblob import TextBlob
from datetime import datetime
import os
import requests
from streamlit_autorefresh import st_autorefresh

# --- Config ---
st.set_page_config(page_title="Integrated Stock App", layout="wide")
st.title(" Stock Market Dashboard")
dark_mode = st.sidebar.checkbox(" Dark Mode")
plt.style.use('dark_background' if dark_mode else 'ggplot')

# --- Finnhub API ---
FINNHUB_API_KEY = "d14r3g1r01qop9mf0b9gd14r3g1r01qop9mf0ba0"  # 🔁 Replace with your actual key

def load_data_finnhub(symbol):
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if "c" in data and data["c"] != 0:
            df = pd.DataFrame([{
                "Current": data["c"],
                "High": data["h"],
                "Low": data["l"],
                "Open": data["o"],
                "Prev Close": data["pc"],
                "Time": datetime.fromtimestamp(data["t"]).strftime("%Y-%m-%d %H:%M:%S")
            }])
            return df
        else:
            st.error("⚠️ Invalid symbol or no data received.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error fetching real-time data: {e}")
        return pd.DataFrame()

# --- Load Model ---
MODEL_PATH = "Stock Predictions Model.keras"
model = tf.keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# --- Input ---
stock = st.text_input("Enter Stock Symbol", "AAPL").strip().upper()

# --- Auto Refresh Every 1 Minute ---
st_autorefresh(interval=60_000, key="live_refresh")

# --- Real-Time Quote ---
st.subheader("📡 Real-Time Stock Quote")
quote = load_data_finnhub(stock)
if not quote.empty:
    st.dataframe(quote)

# --- Prediction Stub (placeholder) ---
st.subheader("🔮 Price Prediction")
if model is not None:
    st.info("🔧 This section will require recent historical data. Add historical support if needed.")
else:
    st.warning("Prediction model not found.")

# --- Sentiment Analysis ---
st.subheader("🗞️ Simulated News Sentiment")

def get_mock_news_sentiment():
    news = [
        "Stock price hits all-time high",
        "CEO resigns due to internal conflicts",
        "New product line receives positive reviews",
        "Market uncertainty increases volatility"
    ]
    return [(n, TextBlob(n).sentiment.polarity) for n in news]

for headline, polarity in get_mock_news_sentiment():
    icon = "🔺" if polarity > 0 else "🔻" if polarity < 0 else "⚖️"
    st.markdown(f"- {headline} — Sentiment: {icon} ({polarity:.2f})")
