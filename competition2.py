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

###Import Data
train = pd.read_csv("train.csv", index_col = None)
y = train[['LABELS']]
x = train.drop('LABELS', axis=1)
xMean = x.mean()
test_nolabels = pd.read_csv("test_nolabels.csv", index_col = None)
# Splitting trainig/Validation
x_train, x_validation, y_train, y_validation = train_test_split( x,y, test_size=0.2)

## Replacing possible missing values
x_train = x_train.fillna(xMean)
x_validation = x_validation.fillna(xMean)

## adapt the prediction to Kaggel format of submission 
def formating_prediction(predictions): 
        '''
        return de predicted classes from the hypotesis function result (sigmoid(W,X))
        @hypotesis : matrix of probablilities 
        '''
        y_hat = pd.DataFrame({'S.No' : [],'LABELS' : []}, dtype=np.int8, index=None) 
        for i in range(len(predictions)):
            y_hat.loc[i] = [i,predictions[i]]
        return pd.DataFrame(data = y_hat, index=None) 

### Some helpers function
def predictOnSet(modelFilename, x_test):
        # # load the model from disk to predict new dataSet
    loaded_model = pickle.load(open(modelFilename, 'rb'))
    prediction = loaded_model.predict(x_test)
    return prediction

def savingModels(classifier, modelFileName):
    '''
    NOTE: Do not forget the extention = *.pkl
    Save as : 'modelFileName.pkl'
    '''
    joblib.dump(classifier, modelFileName)


def importModel(modefname):
    model = pickle.load(open(modefname,'rb'))
    return model

def savePrediction(prediction, filename):
    '''
    Save predictions
    @argument: filename: Remenber EXTENTION 'filename.csv'
    '''
    prediction = prediction.astype('int32') #exsure prediction as integer
    predictions_DF = formating_prediction(prediction)
    return predictions_DF.to_csv(filename, index = None)

# ## modle evaluation
def metric_RocAuc(y_probability, y_validation, estimator_name):
    '''
    Calculate and plt ROC metric
    @argument: y_probability : the probability class=1.
    @argument: y_validation: True labels.
    fpr, tpr = false_positive, true_positive.
    Return: "false_positive" and "true_positive", ROC_auc metric.
    '''
    fpr, tpr, _ = roc_curve(y_validation, y_probability) 
    roc_auc = auc(fpr, tpr)
    fig, axes = plt.subplots(constrained_layout=True,figsize=(5,3), dpi=150)
    fig.suptitle(estimator_name)
    axes.plot([0, 1], [0, 1], color= 'k',linestyle="--") # perfect fit 
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                       estimator_name=estimator_name)
    display.plot(ax=axes)
    return fpr, tpr, roc_auc

## Show some evaluation criteria on the clasifier
def evaluate_model(x_train, y_train, x_validation, y_validation, classifier):
    features = x_train.columns
    # validation_Prediction = classifier.predict(x_validation)
    validation_PredictedProb = classifier.predict_proba(x_validation)[:, 1]
    ### ROC metric and curve #####
    clasifierName = type(classifier).__name__
    metric_RocAuc(validation_PredictedProb, y_validation,clasifierName)
    fi_model = pd.DataFrame({'feature': features,
                   'importance': classifier.feature_importances_}).\
                    sort_values('importance', ascending = False)
    clasifierNameExtended = clasifierName + "_info_fi"     
    fi_model.to_csv(clasifierNameExtended, index = None)
    return fi_model

#     ## XGBOOST
def xgboost(x_train, y_train, x_validation, y_validation):    
    # fit model no training data
    model = XGBClassifier(use_label_encoder=False, eval_metric ='error')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_validation)
    # predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_validation, y_pred)
    print("Accuracy xgboost: %.2f%%" % (accuracy * 100.0))
    predictionsDF = formating_prediction(y_pred)
    nameToSavePrediction = type(model).__name__ + '.csv'
    predictionsDF.to_csv(nameToSavePrediction)
    return model, accuracy, predictionsDF

<<<<<<< HEAD
model = XGBClassifier(use_label_encoder=False)
x_train = x_train[0:1000]
y_train = y_train[0:1000]
model.fit(x_train, y_train)
savingModels(model, "xgboost_singleTest.pkl")
# y_pred = model.predict(x_validation)

evaluate_model(x_train, y_train, x_validation, y_validation, model)
=======
model, accuracy, predictionsDF = xgboost(x_train, y_train, x_validation, y_validation)

# savingModels(model, "xgboost_singleTest.pkl")
# y_pred = model.predict(x_validation)
# fi_model = evaluate_model(x_train, y_train, x_validation, y_validation, model)
# fi_model.to_csv('fi_xgboost.csv')
>>>>>>> ae19ee440bd9f9890ddb9e7cae8268b127d80cd1
