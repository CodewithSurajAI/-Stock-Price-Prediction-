import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close']])

X = []
y = []

for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32)

model.save("../models/lstm_model.h5")
🔹 5. Web App – app/streamlit_app.py
python
Copy
Edit
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