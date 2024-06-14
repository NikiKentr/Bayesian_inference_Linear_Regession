import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os

# Load the data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'salary.csv')
data = pd.read_csv(data_path)

# Separate features (X) and target variable (y)
X = data[['Experience']]  
y = data['Salary']  

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Print model parameters
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Predict salaries for the test data
y_pred = model.predict(X_test)

# Calculate and print performance metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("R^2:", r2)
print("Mean Squared Error:", mse)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Salaries')
plt.scatter(X_test, y_pred, color='red', label='Predicted Salaries')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.title('Actual vs Predicted Salaries')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()

# Save the plot
figures_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'salary_prediction.png')
os.makedirs(os.path.dirname(figures_path), exist_ok=True)
plt.savefig(figures_path)

# Show the plot
plt.show()