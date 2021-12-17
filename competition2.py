'''
.git/Competition2 
Author: Abdiel Fernandez 
Cours: IFT 6390 Machine Learning
'''
import numpy as np
from numpy import random
import pandas as pd
import joblib
import pickle
import matplotlib.pyplot as plt
import math, decimal
from math import exp
import seaborn as sns
import sklearn as sk
from sklearn import metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_curve, auc, roc_auc_score
from sklearn.calibration import calibration_curve, CalibrationDisplay
from xgboost import XGBClassifier
from collections import Counter
seedRF = 50

def meanStd(dataset):
    '''
    dataset_minmax(dataset)
    return a list like {min:#,max:#}
    # Find the min and std values for each column
    '''
    col = dataset.shape[1]
    meanVal, stdVal = 0,0
    stats = list()
    for i in range(col):
        val = dataset.iloc[:, i]
        meanVal = np.mean(val)
        stdVal = np.std(val)
        stats.append([meanVal,stdVal])
    return stats

def standardize_data(dataset, mean_std):
    '''
    standardize_data(dataset, mean_std)
    @mean_std: @arguent: list of min/max valuer per column {min:#,max:#}
    # Rescale dataset columns to the range 0-1
    '''
    col = dataset.shape[1]
    row = dataset.shape[0]
    for i in range(1,col):
        for n in range(row):
            dataset.iloc[n,i] -= mean_std[i][0]
            dataset.iloc[n,i] /= mean_std[i][1]
    return dataset


###Import Data
# train = pd.read_csv("train.csv", index_col = None)
# y = train[['LABELS']]
# x = train.drop('LABELS', axis=1)
# xMean = x.mean()
# x = x.fillna(xMean)
test_nolabels = pd.read_csv("test_nolabels.csv", index_col = None)
test_nolabels_means = test_nolabels.mean()
test_nolabels = test_nolabels.fillna(test_nolabels_means)
toposElevation = {"topo_elevation_jan",'topo_elevation_feb','topo_elevation_mar','topo_elevation_apr','topo_elevation_may','topo_elevation_jun','topo_elevation_jul','topo_elevation_aug','topo_elevation_sep','topo_elevation_oct','topo_elevation_nov','topo_elevation_dec'}
topoSlope = {'topo_slope_jan','topo_slope_feb','topo_slope_mar','topo_slope_apr','topo_slope_may','topo_slope_aug','topo_slope_jun','topo_slope_jul','topo_slope_sep','topo_slope_oct','topo_slope_nov','topo_slope_dec'}
topoElevationDF = test_nolabels[toposElevation]
topoElevationMean = topoElevationDF.mean(axis=1)
test_nolabels = test_nolabels.drop(toposElevation,axis=1)
test_nolabels['topoElevationMean'] = topoElevationMean
topoSlopeDF = test_nolabels[topoSlope]
topoSlopeMean = topoSlopeDF.mean(axis=1)
test_nolabels = test_nolabels.drop(topoSlope, axis=1)
test_nolabels['topoSlope'] = topoSlopeMean
mean_std_x_train = meanStd(test_nolabels)
test_nolabelsx = standardize_data(test_nolabels, mean_std_x_train)
x.to_csv('nonLabeled_standardized.csv')




