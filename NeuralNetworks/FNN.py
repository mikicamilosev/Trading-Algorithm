import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Generate dummy data
x_train = np.random.random((1000, 100))
y_train = np.random.randint(10, size=(1000))  # Remove the extra dimension

# Convert class vectors to binary class matrices (one-hot encoding)
y_train = to_categorical(y_train, num_classes=10)

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

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

# Plot training loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot training accuracy
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
