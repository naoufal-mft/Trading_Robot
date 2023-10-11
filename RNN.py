# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:08:06 2023

@author: Kaoutar
"""
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# Load your data
df = pd.read_csv('test.csv')
df = df.rename(columns={'timestamp': 'Timestamp', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
# Feature selection and scaling
features = df[['Open', 'High', 'Low', 'Close', 'Volume']]
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Sequence length and initialization
sequence_length = 10  # Adjust as needed
sequences = []
labels = []

# Create sequences and labels
for i in range(len(features_scaled) - sequence_length):
    # Create a sequence of length sequence_length
    sequence = features_scaled[i : i + sequence_length]
    sequences.append(sequence)

    # Determine the label based on the next price movement
    if df['Close'].iloc[i + sequence_length] > df['Close'].iloc[i + sequence_length - 1]:
        labels.append('Up')
    else:
        labels.append('Down')

# Encoding Labels
# Convert 'Up' to 1 and 'Down' to 0
labels = [1 if label == 'Up' else 0 for label in labels]

sequences = np.array(sequences)

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, shuffle=False)
X_train = np.array(X_train)
y_train = np.array(y_train)
# Print training and testing data
"""print("X_train:")
print(X_train)
print("y_train:")
print(y_train)

print("Testing Data:")
print("X_test:")
print(X_test)
print("y_test:")
print(y_test)"""

# Define the RNN model
"""model = keras.Sequential()
model.add(keras.layers.LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2]))
model.add(keras.layers.Dense(1, activation='sigmoid'))
"""
# ...

# Define the RNN model using TensorFlow Keras
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units=20, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(tf.keras.layers.Dense(1, activation='relu'))

# ...

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
X_test = np.array(X_test)
y_test = np.array(y_test)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")