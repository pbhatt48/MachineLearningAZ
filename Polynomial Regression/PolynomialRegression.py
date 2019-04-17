# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#get prediction val
y_pred = lin_reg.predict(X)

#visualize linear regression
plt.scatter(X, y, color='red')
plt.plot(X, y_pred, color='blue')
plt.xlabel("No of years")
plt.ylabel("Salary")
plt.title("Linear regression (Salary VS Year)")
plt.show()


#fitting polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
poly_reg = poly_reg.fit(X_poly, y)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)


#visualize polyregression regression
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='blue')
plt.xlabel("No of years")
plt.ylabel("Salary")
plt.title("Linear regression (Salary VS Year)")
plt.show()
 
