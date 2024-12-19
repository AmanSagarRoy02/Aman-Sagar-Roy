import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset to check its structure
file_path = r"C:/Users/amans/OneDrive/Documents/BostonHousing.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
df.head()
print(df.head())

# Check for missing values in the dataset
missing_values = df.isnull().sum()
missing_values

# Impute missing values in 'rm' column with the median value using numpy
df['rm'] = df['rm'].fillna(np.median(df['rm'].dropna()))

# Split the data into features (X) and target (y)
X = df.drop('medv', axis=1) # Features (all columns except 'medv')
y = df['medv']  # Target (house prices)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the Mean Squared Error and R-squared values
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# Plot True vs Predicted values
plt.figure(figsize=(10,6))

# Scatter plot of true vs predicted values
plt.scatter(y_test, y_pred, color='blue', label='Predicted Values')

# Add a diagonal line representing the perfect prediction (y_test = y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Fit')

# Labeling the axes and title
plt.xlabel("True Values (y_test)")
plt.ylabel("Predicted Values (y_pred)")
plt.title("True vs Predicted House Prices")

# Show legend to differentiate between actual and predicted data
plt.legend()

# Display the plot
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(10,6))
sns.histplot(residuals, kde=True)
plt.title("Distribution of Residuals")
plt.xlabel("Residuals")
plt.show()

coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)


