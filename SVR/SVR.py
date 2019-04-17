# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#get dataset
dataset = pd.read_csv('Position_Salaries.csv')

#get X and Y values
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values

#apply feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_Y.fit_transform(y)

#get the SVR
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

#predicting the result
y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))


#Visualizing the SVR results
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title("SVR regression")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()