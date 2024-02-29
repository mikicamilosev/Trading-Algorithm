import pandas as pd
import numpy as np

def calculate_sma(data, window):
    """
    Calculate the Simple Moving Average (SMA) for a given window size.
    
    Parameters:
        data (pd.Series): The input data series.
        window (int): The window size for the SMA.
    
    Returns:
        pd.Series: The SMA values.
    """
    return data.rolling(window=window).mean()

def calculate_ema(data, window):
    """
    Calculate the Exponential Moving Average (EMA) for a given window size.
    
    Parameters:
        data (pd.Series): The input data series.
        window (int): The window size for the EMA.
    
    Returns:
        pd.Series: The EMA values.
    """
    return data.ewm(span=window, adjust=False).mean()

# Sample data (replace this with your actual trading data)
dates = pd.date_range(start='2024-01-01', end='2024-02-28')
prices = np.random.randint(50, 150, size=len(dates))
data = pd.Series(prices, index=dates)

# Window sizes for moving averages
sma_window = 20
ema_window = 20

# Calculate moving averages
sma = calculate_sma(data, sma_window)
ema = calculate_ema(data, ema_window)

# Print the results
print("Simple Moving Average (SMA):")
print(sma.tail())  # Print the last few SMA values
print("\nExponential Moving Average (EMA):")
print(ema.tail())  # Print the last few EMA values