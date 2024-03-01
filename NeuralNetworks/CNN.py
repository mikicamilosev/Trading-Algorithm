import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# Generate some sample data
np.random.seed(0)
X = np.linspace(-1, 1, 100)
y = 2 * X + np.random.normal(0, 0.2, 100)

# Create a sequential model
model = Sequential()

# Add input and hidden layers
model.add(Dense(units=64, activation='relu', input_dim=1))
model.add(Dense(units=64, activation='relu'))

# Add output layer
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=50, batch_size=10, verbose=0)

# Predictions
y_pred = model.predict(X)

# Plot the data and the predictions
plt.scatter(X, y, label='Actual data')
plt.plot(X, y_pred, color='red', label='Predictions')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Neural Network Regression')
plt.legend()
plt.show()