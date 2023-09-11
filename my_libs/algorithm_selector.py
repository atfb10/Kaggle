'''
TODO
Class does the following:
- parameters
- dataset
- list of models to try
- list of features to exclude

Methods to do following:
- Creates, trains, scales (as needed), fits model
- Models evaluated & ranked according to scoring metrics
- Returns best scoring model

NOTE: Classification Only
NOTE: Not for simple models. If all I think I need is a simple KNN model, I should just make one for the problem at hand
NOTE: SVM not included. Too long to run. If I have a problem SVM's are a good selection for, just create one  
'''

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, RocCurveDisplay, auc, roc_curve
)
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.preprocessing import StandardScaler

'''
TODO:
models
- logistic regression
- randomforest
- xgboost
- lightgbm
- catboost
-  
'''
# Constants
LOG = 'logistic'
F = 'randomforest'
XG = 'xgboost'
L = 'light'
CAT = 'catboost'


class ClassificationModelSelect():

    def __init__(self, dataset: pd.DataFrame, classifiers: list, exclude: list) -> None:
        self.data = dataset
        self.classifiers = classifiers
        self.features_to_exclude = exclude
        self.train_logistic_classifier = False
        self.train_randomforest_classifier = False
        self.train_xgboost = False
        self.train_lightgbm = False
        self.train_catboost = False

    def __compare_models(self) -> None:
        pass

    def __check_models(self) -> None:
        '''
        update boolean attributes appropriately
        '''
        if LOG in self.classifiers:
            self.train_logistic_classifier = True
        if F in self.classifiers:
            self.train_randomforest_classifier = True
        if XG in self.classifiers:
            self.train_xgboost = True
        if L in self.classifiers:
            self.train_lightgbm = True
        if CAT in self.classifiers:
            self.train_catboost = True
        return

    def __train_logistic_classifier(self) -> tuple:
        '''
        returns tuple of metrics 
        '''
        if self.train_logistic_classifier:
            pass
        return (0, 0, 0, 0)
    
    def __train_forest_classifier(self) -> tuple:
        '''
        returns tuple of metrics 
        '''
        if self.train_randomforest_classifier:
            pass
        return (0, 0, 0, 0)
    

    def __train_xgboost_classifier(self) -> tuple:
        '''
        returns tuple of metrics 
        '''
        if self.train_logistic_classifier:
            pass
        return (0, 0, 0, 0)
    
    def __train_lightgbm_classifier(self) -> tuple:
        '''
        returns tuple of metrics 
        '''
        if self.train_lightgbm:
            pass
        return (0, 0, 0, 0)

    def __train_catboost_classifier(self) -> tuple:
        '''
        returns tuple of metrics 
        '''
        if self.train_logistic_classifier:
            pass
        return (0, 0, 0, 0)