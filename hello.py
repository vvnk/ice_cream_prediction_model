# Simple Random Forest Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Importing the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/mk-gurucharan/Regression/master/IceCreamData.csv')

X = dataset["Temperature"].values
y = dataset["Revenue"].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)

# Training the Random Forest Regression model on the training set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train.reshape(-1, 1), y_train)

# Predicting a new result
y_pred = regressor.predict(X_test.reshape(-1, 1))

# Comparing the predicted result with the actual result
# Display the values of y_test (Real Values) and y_pred (Predicted Values) in a Pandas DataFrame
df = pd.DataFrame({'Actual': y_test.reshape(-1), 'Predicted': y_pred.reshape(-1)})
print(df)

# Evaluating the model performance using R2 Score and Mean Squared Error
print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Smooth X grid for plotting
X_grid = np.arange(min(X), max(X), 0.1)  # 0.1 step for smooth curve
y_grid_pred = regressor.predict(X_grid.reshape(-1,1))

# Plot
plt.figure(figsize=(8, 5))

# Training Points 
plt.scatter (X_train, y_train, color='green', label='Training Data (Predicted)', alpha=0.6)

# Test Points 
plt.scatter(X_test, y_test, color='red', label='Test Data (Actual)', s=50)

# Predicted Line
plt.plot(X_grid, y_grid_pred, color='blue', label='Predicted', linewidth=2)

# Titles and labels
plt.title('Ice Cream Revenue Prediction (Random Forest Regression)', fontsize=14)
plt.xlabel('Temperature', fontsize=12)
plt.ylabel('Revenue', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save the plot as an image
plt.savefig("ice_cream_prediction.png")  # saves in your current folder
print("Plot saved as ice_cream_prediction.png")