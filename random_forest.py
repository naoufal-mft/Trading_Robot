# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:28:30 2023

@author: Kaoutar
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# Load your CSV data into a DataFrame
df = pd.read_csv('test5.csv')
#df = df.rename(columns={'date': 'Timestamp', '1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'})
print(df)
# Check for missing values in the entire dataset
missing_values = df.isnull().sum()

# Print the results
print("Missing values in the entire dataset:")
print(missing_values)


# Calculate price change from current time step to the next time step
df['PriceChange'] = df['Close'].diff().shift(0)

# Create a target variable based on price change
df['PriceMovement'] = df['PriceChange'].apply(lambda x: 'Up' if x > 0 else 'Down')

# Drop rows with missing values (NaN) in the target variable
df.dropna(subset=['PriceMovement'], inplace=True)


# Define features (OHLCV)
features = ['Open', 'High', 'Low', 'Close', 'Volume']

# Define the target variable
target = 'PriceMovement' 

# Create X (features) and y (target)
X = df[features]
y = df[target]


split_point = 258608

X_train, y_train = X.iloc[:split_point], y.iloc[:split_point]
X_test, y_test = X.iloc[split_point:], y.iloc[split_point:]

# Display the shapes of the training and testing sets
print("X_train:")
print(X_train)
print("y_train:")
print(y_train)

print("Testing Data:")
print("X_test:")
print(X_test)
print("y_test:")
print(y_test)

# Create an instance of the Random Forest model
random_forest_model = RandomForestClassifier(random_state=42)

# Fit the model to your training data
random_forest_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = random_forest_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
# Calculate the win rate
win_rate = accuracy * 100  # Convert to a percentage

# Print the win rate
print(f"Win Rate: {win_rate:.2f}%")
# Print the evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", classification_report_str)
