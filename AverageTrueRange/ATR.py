import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Set a random seed for reproducibility
np.random.seed(42)

# Generate synthetic trading data
num_days = 100
date_today = datetime.date.today()
dates = [date_today - datetime.timedelta(days=x) for x in range(num_days)]
prices = np.random.normal(loc=100, scale=5, size=num_days).cumsum()

# Create a DataFrame
data = pd.DataFrame({'Date': dates, 'Close': prices})

# Function to calculate Average True Range (ATR)
def calculate_atr(data, window=14):
    data['High-Low'] = data['High'] - data['Low']
    data['High-PrevClose'] = np.abs(data['High'] - data['Close'].shift(1))
    data['Low-PrevClose'] = np.abs(data['Low'] - data['Close'].shift(1))

    true_range = data[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    atr = true_range.rolling(window=window).mean()

    return atr

# Adding High and Low columns for synthetic data
data['High'] = data['Close'] + np.random.normal(scale=2, size=num_days)
data['Low'] = data['Close'] - np.random.normal(scale=2, size=num_days)

# Calculate ATR for the generated data
atr = calculate_atr(data)

# Plot the data and ATR
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(data['Close'], label='Close Price')
ax1.set_title('Price and Average True Range (ATR)')
ax1.set_ylabel('Price')
ax1.legend()

ax2.plot(atr, label='ATR', color='purple')
ax2.set_xlabel('Date')
ax2.set_ylabel('ATR')
ax2.legend()

plt.show()
