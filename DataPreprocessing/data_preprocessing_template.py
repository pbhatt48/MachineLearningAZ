# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#dataset = pd.read_csv('/Users/sadichha/UdacityClasses/UdemyClass/ML_Practices/MachineLearningAZ/DataPreprocessing/Data.csv')

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
X[:, 1:] = imputer.fit_transform(X[:, 1:])

#lets add label encoder. 
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

#creating labelencoder for Y
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#create test and train data
from sk
