import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Create a sequential model
model = Sequential()

# Add input and hidden layers
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=64, activation='relu'))

# Add output layer
model.add(Dense(units=10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# Generate dummy data
x_train = np.random.random((1000, 100))
y_train = np.random.randint(10, size=(1000, 1))

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)