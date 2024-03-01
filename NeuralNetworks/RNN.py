import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

# Create a sequential model
model = Sequential()

# Add a simple RNN layer
model.add(SimpleRNN(units=64, input_shape=(10, 32)))  # 10 timesteps, 32 input dimensions

# Add a fully connected layer
model.add(Dense(units=10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Generate dummy data
x_train = np.random.random((1000, 10, 32))  # 1000 samples, 10 timesteps, 32 input dimensions
y_train = np.random.randint(2, size=(1000, 10))  # 1000 samples, 10 output dimensions

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)