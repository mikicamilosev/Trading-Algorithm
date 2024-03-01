# Bollinger Bands:
# Description: Bollinger Bands consist of a middle band being an N-period simple moving average and upper/lower bands representing N standard deviations away from the moving average, indicating volatility.

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

# Calculate Bollinger Bands
def calculate_bollinger_bands(data, window_size=20, num_std_dev=2):
    rolling_mean = data['Close'].rolling(window=window_size).mean()
    rolling_std = data['Close'].rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

# Calculate Bollinger Bands for the generated data
upper_band, lower_band = calculate_bollinger_bands(data)

# Plot the data and Bollinger Bands
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Close Price')
plt.plot(upper_band, label='Upper Bollinger Band', color='red')
plt.plot(lower_band, label='Lower Bollinger Band', color='green')
plt.title('Bollinger Bands')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()