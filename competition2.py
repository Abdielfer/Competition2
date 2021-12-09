'''
.git/Competition2 
Author: Abdiel Fernandez 
Cours: IFT 6390 Machine Learning
'''
# imports

import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import math, decimal
from math import exp
import sklearn as sk
import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import metrics
# from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import OneHotEncoder # One can do it by hand too..See Notes at the end of the code
# from sklearn.linear_model import LogisticRegression, SGDClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report
# from scipy.special import expit  #  to prevent overflow in exp(-z)
# import joblib
seed1 = 0

# Data importing
train = pd.read_csv("train.csv", index_col = None)
y = train[['LABELS']]
x = train.drop('LABELS', axis=1)
# test_nolabels = pd.read_csv("test_nolabels.csv", index_col = None)



# Splitting trainig/Validation
# x_train, x_validation, y_train, y_validation = train_test_split( x,y, test_size=0.3, random_state=4)

# print(y_train.columns())
# create grill for RF training

# Random Forest


# predict and format prediction


# Save model and sve predisction





