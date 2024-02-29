
# Moving Average Convergence Divergence (MACD) Algorithm
import pandas as pd
import numpy as np

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) indicator.

    Args:
        df (pandas.DataFrame): DataFrame containing 'Close' prices.
        short_window (int): Short window period (default is 12).
        long_window (int): Long window period (default is 26).
        signal_window (int): Signal window period (default is 9).

    Returns:
        pandas.DataFrame: DataFrame with MACD values and signal line.
    """
    # Calculate short-term and long-term exponential moving averages
    short_ema = df['Close'].ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = df['Close'].ewm(span=long_window, min_periods=1, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = short_ema - long_ema
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_window, min_periods=1, adjust=False).mean()
    
    # Create DataFrame to store MACD and signal line values
    macd_df = pd.DataFrame({'MACD': macd_line, 'Signal': signal_line})
    
    return macd_df

# Sample data (closing prices)
data = {
    'Close': [50, 55, 60, 65, 70, 65, 60, 55, 50, 45]
}
df = pd.DataFrame(data)

# Calculate MACD using the sample data
macd_df = calculate_macd(df)

# Display the MACD DataFrame
print("MACD and Signal Line:")
print(macd_df)
