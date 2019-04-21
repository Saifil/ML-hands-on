#cmd + i for info

#import the libraries
import numpy as np #for mathematics
import matplotlib.pyplot as plt #for plotting
import pandas as pd # for inporting and managing dataset

#import the dataset
dataset = pd.read_csv("Data.csv")

#extract part of dataset in a variable
X = dataset.iloc[:, :-1].values 
#first param is rows, second is column
# : signifies all, :-1 is all except last one 
y = dataset.iloc[:, 3].values #input index as param

#missing data
from sklearn.preprocessing import Imputer #preprocessing library
#create a object for missing values
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#param1 = what is identified as missing, param2 = stratergy to get new value, param3 = along rows or columns (1/0)
#fit imputer to the matrix
imputer = imputer.fit(X[:, 1:3]) #all rows, column 1 to 3 - 1 , i.e., 1 and 2
#replace the values in X
X[:, 1:3] = imputer.transform(X[:, 1:3])
#approach 2
#new imputer for age to be the most frequent and salary as mean
imputer_age = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer_age = imputer_age.fit(X[:, 1:2]) 
X[:, 1:2] = imputer_age.transform(X[:, 1:2]) #use arnge here for single element
#for salary
imputer_sal = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer_sal = imputer_sal.fit(X[:, 2:3]) 
X[:, 2:3] = imputer_sal.transform(X[:, 2:3]) #use arnge here for single element

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#label encoder converts the categorial values to numbers
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#onehotencoder create separate columns for each type, and assigns 1 to the corresponding column in the row
#used when do not want to rank, but want to make from category to values
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
#incase of the dependent variable (y)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#splitting into training and testing set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#random state is the seend number, same seed number results in same selection and splitting

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
