import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/AAPL_stock_data.csv", parse_dates=['Date'])
df['MA10'] = df['Close'].rolling(10).mean()
df['MA20'] = df['Close'].rolling(20).mean()

plt.plot(df['Date'], df['Close'], label='Close')
plt.plot(df['Date'], df['MA10'], label='MA10')
plt.plot(df['Date'], df['MA20'], label='MA20')
plt.legend()
plt.title("Apple Stock with Moving Averages")
plt.show()
