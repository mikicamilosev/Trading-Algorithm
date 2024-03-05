# Markowitz Modern Portfolio Theory:
# Description: Markowitz's theory focuses on portfolio optimization by balancing risk and return through diversification.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define historical returns and covariance matrix of assets
returns = pd.DataFrame({
    'Stock_A': [0.05, 0.06, 0.04, 0.03, 0.02],
    'Stock_B': [0.07, 0.08, 0.06, 0.05, 0.04],
    'Stock_C': [0.10, 0.12, 0.11, 0.09, 0.08]
})

cov_matrix = returns.cov()

# Define function to calculate portfolio returns and volatility
def portfolio_performance(weights, returns, cov_matrix):
    port_returns = np.sum(returns.mean() * weights) * 252
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (port_returns - risk_free_rate) / port_volatility
    return -sharpe_ratio

# Define function to minimize negative Sharpe ratio
def negative_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
    port_returns, port_volatility = portfolio_performance(weights, returns, cov_matrix)
    sharpe_ratio = (port_returns - risk_free_rate) / port_volatility
    return -sharpe_ratio

# Define constraints for optimization
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Define bounds for weights (0 <= weight <= 1)
bounds = tuple((0, 1) for _ in range(len(returns.columns)))

# Define initial guess for weights
initial_guess = [1.0 / len(returns.columns)] * len(returns.columns)

# Define risk-free rate
risk_free_rate = 0.02

# Perform optimization to find optimal portfolio weights
optimal_weights = minimize(negative_sharpe_ratio, initial_guess,
                           args=(returns, cov_matrix, risk_free_rate),
                           method='SLSQP', bounds=bounds, constraints=constraints)

# Calculate optimal portfolio returns and volatility
optimal_returns, optimal_volatility = portfolio_performance(optimal_weights.x, returns, cov_matrix)

# Print results
print("Optimal Portfolio Weights:")
for asset, weight in zip(returns.columns, optimal_weights.x):
    print(f"{asset}: {weight:.2f}")

print(f"Optimal Portfolio Returns: {optimal_returns:.2f}")
print(f"Optimal Portfolio Volatility: {optimal_volatility:.2f}")

# Plot efficient frontier
def efficient_frontier(returns, cov_matrix, risk_free_rate):
    target_returns = np.linspace(returns.min(), returns.max(), num=100)
    efficient_portfolios = []
    for target_return in target_returns:
        constraints = ({'type': 'eq', 'fun': lambda x: portfolio_performance(x, returns, cov_matrix)[0] - target_return},
                       {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(returns.columns)))
        result = minimize(portfolio_performance, initial_guess,
                          args=(returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)
        efficient_portfolios.append(result['fun'])
    return efficient_portfolios

efficient_portfolios = efficient_frontier(returns, cov_matrix, risk_free_rate)
plt.plot(efficient_portfolios, target_returns)
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.title('Efficient Frontier')
plt.show()