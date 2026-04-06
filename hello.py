# Simple Random Forest Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/mk-gurucharan/Regression/master/IceCreamData.csv')

X = dataset["Temperature"].values
y = dataset["Revenue"].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)

# Training the Random Forest Regression model on the training set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X_train.reshape(-1, 1), y_train)

# Predicting a new result
y_pred = regressor.predict(X_test.reshape(-1, 1))

# Comparing the predicted result with the actual result
# Display the values of y_test (Real Values) and y_pred (Predicted Values) in a Pandas DataFrame
df = pd.DataFrame({'Actual': y_test.reshape(-1), 'Predicted': y_pred.reshape(-1)})
print(df)
