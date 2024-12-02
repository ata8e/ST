import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Simulate Data

np.random.seed(42)  # For reproducibility

# Simulating 100 data points for X1, X2, X3
n_samples = 100

X1 = np.random.uniform(1, 10, n_samples)  # Random values between 1 and 10
X2 = np.random.uniform(1, 10, n_samples)
X3 = np.random.uniform(1, 10, n_samples)

# True coefficients for the regression model
beta_0 = 5
beta_1 = 2
beta_2 = -3
beta_3 = 1

# Random noise (epsilon)
noise = np.random.normal(0, 2, n_samples)

# Simulate the target variable Y
Y = beta_0 + beta_1 * X1 + beta_2 * X2 + beta_3 * X3 + noise

# Step 2: Create a DataFrame
data = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'Y': Y})

# Step 3: Split the data into training and testing sets
X = data[['X1', 'X2', 'X3']]
y = data['Y']

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the results
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# Step 7: Plot the results (Actual vs Predicted)
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.title('Actual vs Predicted Values')
plt.show()
