# -*- coding: utf-8 -*-
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Generate Dummy Data
np.random.seed(42)

# Create synthetic data for S&P 1500 and CRSP indices
num_samples = 1000
data = pd.DataFrame({
    'Index': np.random.choice(['S&P 1500', 'CRSP'], size=num_samples),
    'Market_Cap': np.random.uniform(1e9, 1e12, num_samples),  # Market cap in dollars
    'Volume': np.random.randint(1e5, 1e7, num_samples),  # Daily trading volume
    'Float_Shares': np.random.uniform(1e6, 1e9, num_samples),  # Number of float shares
    'Previous_Change': np.random.uniform(-5, 5, num_samples),  # Previous share change percentage
    'Sentiment_Score': np.random.uniform(-1, 1, num_samples),  # News sentiment
    'Prediction_Target': np.random.choice([0, 1], size=num_samples, p=[0.5, 0.5])  # 1: Change, 0: No Change
})

# Preview the data
print("Dummy Data Sample:")
print(data.head())

# Step 2: Data Preprocessing
# Convert 'Index' to numeric
data['Index'] = data['Index'].map({'S&P 1500': 1, 'CRSP': 0})

# Split the data into features and target
X = data.drop(columns=['Prediction_Target'])
y = data['Prediction_Target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a Predictive Model
# Use Random Forest for simplicity
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Step 4: Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")

# Hit Rate Evaluation
hit_rate = np.mean(y_pred == y_test)
print(f"Hit Rate: {hit_rate:.2%}")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Step 5: Strategy Implementation
# Simulate a trading strategy based on the model's predictions
test_data = X_test.copy()
test_data['Actual'] = y_test
test_data['Predicted'] = y_pred

# Assume each correct prediction generates a profit of $1,000, and incorrect predictions result in a loss of $500.
test_data['Profit'] = np.where(
    test_data['Actual'] == test_data['Predicted'], 1000, -500
)

# Calculate total profit
total_profit = test_data['Profit'].sum()
print(f"\nTotal Profit from Strategy: ${total_profit}")

# Step 6: Visualize Strategy Outcomes
# Profit Distribution
plt.figure(figsize=(10, 6))
plt.hist(test_data['Profit'], bins=10, color='blue', alpha=0.7, edgecolor='black')
plt.title("Profit Distribution")
plt.xlabel("Profit ($)")
plt.ylabel("Frequency")
plt.show()

# Plot Actual vs Predicted Changes
plt.figure(figsize=(10, 6))
plt.scatter(test_data.index, test_data['Actual'], label='Actual Changes', alpha=0.6)
plt.scatter(test_data.index, test_data['Predicted'], label='Predicted Changes', alpha=0.6, color='red')
plt.title("Actual vs Predicted Changes")
plt.xlabel("Index")
plt.ylabel("Change (0 or 1)")
plt.legend()
plt.show()

