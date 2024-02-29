import numpy as np

def calculate_rsi(data, window=14):
    """
    Calculate the Relative Strength Index (RSI) for a given dataset.
    
    Args:
        data (numpy.ndarray): Array of closing prices.
        window (int): Window size for calculating RSI (default is 14).
        
    Returns:
        numpy.ndarray: Array of RSI values.
    """
    delta = np.diff(data)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.convolve(gain, np.ones(window)/window, mode="full")[:len(data)]
    avg_loss = np.convolve(loss, np.ones(window)/window, mode="full")[:len(data)]
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Generate sample closing prices using NumPy
np.random.seed(0)  # For reproducibility
closing_prices = np.random.randint(50, 150, size=50)

# Calculate RSI for the sample data
rsi_values = calculate_rsi(closing_prices)

# Print the RSI values
print("Sample Closing Prices:", closing_prices)
print("RSI Values:", rsi_values)