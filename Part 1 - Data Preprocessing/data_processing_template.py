#cmd + i for info

#import the libraries
import numpy as np #for mathematics
import matplotlib.pyplot as plt #for plotting
import pandas as pd # for inporting and managing dataset

#import the dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, 3].values #input index as param

#splitting into training and testing set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
