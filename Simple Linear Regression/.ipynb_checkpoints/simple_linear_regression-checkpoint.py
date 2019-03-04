# Simple Linear Regression

# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Get data
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values

#splitting training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0) 

#create a linear regressor
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predict from linear regressor
y_pred = regressor.predict(X_test)

#visualization of the Training data
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Vs Experience (Training Set)")
plt.show()

#Visualization of the Test data
plt.scatter(X_test, y_test, color="red")
plt.plot(X_test, y_pred, color='blue')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary Vs Experience (Test Set)')
plt.show()


