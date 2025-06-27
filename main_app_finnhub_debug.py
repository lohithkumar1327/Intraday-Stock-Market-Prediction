
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
st.title("📊 Stock Market Dashboard")
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")
plt.style.use('dark_background' if dark_mode else 'ggplot')

# --- Finnhub API ---
FINNHUB_API_KEY = "d14r3g1r01qop9mf0b9gd14r3g1r01qop9mf0ba0"

def load_data_finnhub(symbol):
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
    try:
        response = requests.get(url)
        st.code(f"🔁 Real-Time API Response: {response.text}", language='json')  # Debug response
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
            st.error("⚠️ Invalid symbol or no price data received.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Real-time fetch error: {e}")
        return pd.DataFrame()

def get_historical_data(symbol, resolution="D", count=120):
    url = "https://finnhub.io/api/v1/stock/candle"
    now = int(datetime.now().timestamp())
    past = now - count * 24 * 60 * 60
    params = {
        "symbol": symbol,
        "resolution": resolution,
        "from": past,
        "to": now,
        "token": FINNHUB_API_KEY
    }
    response = requests.get(url, params=params)
    st.code(f"📦 Historical API Response: {response.text}", language='json')  # Debug response
    data = response.json()
    if data.get("s") == "ok":
        df = pd.DataFrame({
            "Date": pd.to_datetime(data["t"], unit="s"),
            "Close": data["c"]
        })
        return df
    else:
        st.warning("⚠️ Could not retrieve historical candle data.")
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

# --- Price Prediction using Historical Data ---
st.subheader("🔮 Price Prediction")

if model is not None:
    hist_data = get_historical_data(stock, count=120)
    if not hist_data.empty:
        df = hist_data.copy()
        close_data = df['Close'].values.reshape(-1, 1)

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_data)

        if len(scaled_data) >= 60:
            X_input = scaled_data[-60:]
            X_input = np.reshape(X_input, (1, 60, 1))

            prediction = model.predict(X_input)
            predicted_price = scaler.inverse_transform(prediction)[0][0]

            st.success(f"📈 Predicted Next Closing Price: **${predicted_price:.2f}**")

            df_plot = df[-60:].copy()
            df_plot = df_plot.append({
                "Date": df_plot["Date"].iloc[-1] + pd.Timedelta(days=1),
                "Close": predicted_price
            }, ignore_index=True)

            fig, ax = plt.subplots()
            ax.plot(df_plot["Date"][:-1], df_plot["Close"][:-1], label="Historical Close", linewidth=2)
            ax.plot(df_plot["Date"].iloc[-1], df_plot["Close"].iloc[-1], 'ro', label="Predicted Price")
            ax.set_title("Historical Close & Predicted Next Price")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("Not enough data to make a prediction (need at least 60 days).")
    else:
        st.warning("⚠️ Could not fetch historical data for prediction.")
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
