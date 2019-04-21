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
Y = dataset.iloc[:, 3].values #input index as param