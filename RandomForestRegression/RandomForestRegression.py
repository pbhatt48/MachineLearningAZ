# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values

#applying random forest regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X,y)

#predicting
y_pred = regressor.predict(np.array([[6.5]]))

#visualizing
X_grid = np.arange(min(X),max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Experience Vs Salay (Random Forest Regressor)")
plt.show()

