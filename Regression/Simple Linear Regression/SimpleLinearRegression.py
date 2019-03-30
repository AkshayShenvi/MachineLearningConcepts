# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:19:10 2019

@author: Akshay Shenvi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

#Importing dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#imputer = Imputer(missing_values= 'NaN', strategy ='mean', axis = 0)
#imputer =imputer.fit(X[:, 1:3])
#X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encoding Categorical data
#labelencoder_X = LabelEncoder()
#X[:,0] =labelencoder_X.fit_transform(X[:,0])
#onehotencoder = OneHotEncoder(categorical_features=[0])
#X = onehotencoder.fit_transform(X).toarray()
#labelencoder_y= LabelEncoder()
#y = labelencoder_y.fit_transform(y)

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1/3, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the Test set
y_pred = regressor.predict(X_test)

#Feature scaling
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)