# Pairs Trading:
# Description: Pairs trading involves trading a long position in one security and a short position in another related security to exploit relative price movements.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate synthetic price data for two instruments
np.random.seed(0)  # For reproducibility
n_periods = 1000
t = np.arange(n_periods)
price1 = np.cumsum(np.random.randn(n_periods))  # Random walk for instrument 1
price2 = np.cumsum(np.random.randn(n_periods))  # Random walk for instrument 2

# Create DataFrame with synthetic price data
df1 = pd.DataFrame({'Close': price1}, index=pd.date_range('2022-01-01', periods=n_periods))
df2 = pd.DataFrame({'Close': price2}, index=pd.date_range('2022-01-01', periods=n_periods))

# Calculate spread between the two instruments (e.g., price ratio)
spread = df1['Close'] / df2['Close']

# Define rolling mean and standard deviation of spread
spread_mean = spread.rolling(window=20).mean()
spread_std = spread.rolling(window=20).std()

# Calculate z-score of spread
z_score = (spread - spread_mean) / spread_std

# Define entry and exit thresholds
entry_threshold = 1.0
exit_threshold = 0.0

# Initialize positions
position1 = 0  # Long position for instrument 1
position2 = 0  # Short position for instrument 2

# Plot spread and z-score
plt.figure(figsize=(10, 6))
plt.plot(df1.index, spread, label='Spread')
plt.plot(df1.index, spread_mean, label='Rolling Mean')
plt.plot(df1.index, z_score, label='Z-Score')
plt.axhline(y=entry_threshold, color='r', linestyle='--', label='Entry Threshold')
plt.axhline(y=exit_threshold, color='g', linestyle='--', label='Exit Threshold')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Pairs Trading Strategy')
plt.legend()
plt.grid(True)
plt.show()

# Implement pairs trading strategy
for i in range(len(df1)):
    # Entry condition: z-score exceeds entry threshold
    if z_score.iloc[i] > entry_threshold:
        if position1 == 0 and position2 == 0:
            position1 = 1
            position2 = -1
            print("Entry Long:", df1.index[i], "Short:", df2.index[i])
    
    # Exit condition: z-score falls below exit threshold
    elif z_score.iloc[i] < exit_threshold:
        if position1 == 1 and position2 == -1:
            position1 = 0
            position2 = 0
            print("Exit Long:", df1.index[i], "Short:", df2.index[i])

# Close any open positions at the end
if position1 == 1 and position2 == -1:
    print("Close Long:", df1.index[-1], "Short:", df2.index[-1])