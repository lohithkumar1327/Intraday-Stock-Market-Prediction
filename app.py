import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
import math

model = load_model('Stock Predictions Model.keras')

st.header('Stock Market Predictor')

stock =st.text_input('Enter Stock Symbol','AAPL')
start = '2022-01-01'
end = '2025-4-10'

data = yf.download(stock, start ,end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)


# Define a function to run prediction and calculate metrics
def run_prediction():
    # Download fresh data
    new_data = yf.download(stock, start, end)

    # Preprocessing
    data_train = pd.DataFrame(new_data.Close[0: int(len(new_data)*0.80)])
    data_test = pd.DataFrame(new_data.Close[int(len(new_data)*0.80): len(new_data)])

    scaler = MinMaxScaler(feature_range=(0,1))

    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)

    x = []
    y_true = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y_true.append(data_test_scale[i,0])

    x, y_true = np.array(x), np.array(y_true)

    predict = model.predict(x)

    scale = 1/scaler.scale_
    predict = predict * scale
    y_true = y_true * scale

    # Calculate metrics
    mse = mean_squared_error(y_true, predict)
    mae = mean_absolute_error(y_true, predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, predict)

    return predict, y_true, mse, mae, rmse, r2

# Add a button to rerun prediction
# Run prediction every time app runs or button is pressed
if 'predict' not in st.session_state or st.button("🔄 Predict Again"):
    predict, y, mse, mae, rmse, r2 = run_prediction()
    st.session_state.predict = predict
    st.session_state.y = y
    st.session_state.mse = mse
    st.session_state.mae = mae
    st.session_state.rmse = rmse
    st.session_state.r2 = r2
    st.success("Prediction updated with fresh data! 🎯")
else:
    predict = st.session_state.predict
    y = st.session_state.y
    mse = st.session_state.mse
    mae = st.session_state.mae
    rmse = st.session_state.rmse
    r2 = st.session_state.r2


# Display the fancy metrics
st.subheader('📈 Regression Metrics')

col1, col2, col3, col4 = st.columns(4)

col1.metric(label="📉 MAE", value=f"{mae:.2f}")
col2.metric(label="📉 MSE", value=f"{mse:.2f}")
col3.metric(label="📉 RMSE", value=f"{rmse:.2f}")
col4.metric(label="📈 R² Score", value=f"{r2:.2f}")

# Plot Original vs Predicted Prices again (optional refresh)
st.subheader('Original Price vs Predicted Price (Live)')
fig5 = plt.figure(figsize=(8,6))
plt.plot(y, 'r', label='Original Price')
plt.plot(predict, 'g', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig5)

st.header("Live Stock Trend Tracker")

stock = st.text_input("Symbol", "AAPL")
start = "2022-01-01"
end = datetime.now().strftime("%Y-%m-%d")

# Fetch live data (e.g. 5-minute bars)
data = yf.download(stock, start, end, interval="1d")

# Trend Summary
ma20 = data['Close'].rolling(20).mean()
ma50 = data['Close'].rolling(50).mean()

# Safely extract floats
ma20_last = float(ma20.iloc[-1])
ma50_last = float(ma50.iloc[-1])

if not (math.isnan(ma20_last) or math.isnan(ma50_last)):
    if ma20_last > ma50_last:
        st.success(f"Uptrend 📈 — MA20: {ma20_last:.2f} > MA50: {ma50_last:.2f}")
    elif ma20_last < ma50_last:
        st.error(f"Downtrend 📉 — MA20: {ma20_last:.2f} < MA50: {ma50_last:.2f}")
    else:
        st.info(f"Sideways ➡️ — MA20 = MA50 = {ma20_last:.2f}")
else:
    st.warning("Waiting for enough data to compute MA20 & MA50.")

# Optional: auto-refresh every minute
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=60_000, key="live_refresh")

import time

# After you fetch & display everything:
refresh_interval = 60  # seconds

# Show countdown
remaining = refresh_interval - ((time.time() - st.session_state.get("last_run", time.time())) % refresh_interval)
st.info(f"Auto-refresh in {int(remaining)}s …")

# When interval elapses, rerun
if remaining < 1:
    st.session_state.last_run = time.time()
    st.experimental_rerun()


