import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import plotly.graph_objects as go

# Load model
model = joblib.load("../models/xgboost_model.pkl")
df = pd.read_csv("../data/AAPL_stock_data.csv")

# Predict latest
latest = df.iloc[-1:]
features = ['Open', 'High', 'Low', 'Volume', 'MA10', 'MA20']
latest_X = latest[features]
prediction = model.predict(latest_X)[0]

st.title("📈 SmartPrice Predictor - Apple Stock")
st.write("### Latest Close:", latest['Close'].values[0])
st.write("### Predicted Next Close:", round(prediction, 2))

# Plot chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Actual'))
fig.add_trace(go.Scatter(x=df['Date'], y=df['MA10'], name='MA10'))
fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], name='MA20'))
st.plotly_chart(fig)