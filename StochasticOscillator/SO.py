# Stochastic Oscillator:
# Description: Stochastic Oscillator measures the current price relative to its range over a recent period, identifying overbought or oversold conditions.

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

# Function to calculate Stochastic Oscillator
def calculate_stochastic_oscillator(data, k=14, d=3):
    data['Lowest_Low'] = data['Close'].rolling(window=k).min()
    data['Highest_High'] = data['Close'].rolling(window=k).max()
    
    data['_K'] = ((data['Close'] - data['Lowest_Low']) / (data['Highest_High'] - data['Lowest_Low'])) * 100
    data['_D'] = data['_K'].rolling(window=d).mean()
    
    return data['_K'], data['_D']

# Calculate Stochastic Oscillator for the generated data
_K, _D = calculate_stochastic_oscillator(data)

# Plot the data and Stochastic Oscillator
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(data['Close'], label='Close Price')
ax1.set_title('Price and Stochastic Oscillator')
ax1.set_ylabel('Price')
ax1.legend()

ax2.plot(_K, label='%K', color='blue')
ax2.plot(_D, label='%D', color='orange')
ax2.axhline(80, color='red', linestyle='--', label='Overbought (80)')
ax2.axhline(20, color='green', linestyle='--', label='Oversold (20)')
ax2.set_xlabel('Date')
ax2.set_ylabel('Percentage')
ax2.legend()

plt.show()