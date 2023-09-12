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

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, RocCurveDisplay, auc, roc_curve
)
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.preprocessing import StandardScaler

'''
TODO:
models
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

# Prep models for training
class ModelPrep():

    def __init__(self, dataset: pd.DataFrame, label: str, classifiers: list, exclude=None) -> None: # Exclude is list, but if not included defaults to none
        self.data = dataset
        self.label = label
        self.classifiers = classifiers
        self.features_to_exclude = exclude
        self.train_logistic_classifier = False
        self.train_randomforest_classifier = False
        self.train_xgboost = False
        self.train_lightgbm = False
        self.train_catboost = False

        if classifiers:
            self.__check_models()
        if exclude:
            self.__drop_exclude_features()
        return
    
    def __drop_exclude_features(self) -> None:
        self.data = self.data.drop(self.features_to_exclude)
        return
    
    def __check_models(self) -> None:
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
    
# Train models, assign them to parameters
class ClassifierCreate(ModelPrep):
    def __init__(self, dataset: pd.DataFrame, label: str, classifiers: list, exclude: list) -> None:
        super().__init__(dataset, label, classifiers, exclude)
        self.log_clf = None
        self.forest_clf = None
        self.xgboost_clf = None
        self.lightgbm_clf = None
        self.cat_clf = None
        return 

    def __train_logistic_classifier(self) -> tuple(LogisticRegression, pd.Series):
        '''
        return created model its predictions on test data and actual y values
        '''
        trained_model = None
        predictions = None
        y_test = None
        if self.train_logistic_classifier:
            X = self.data.drop([self.label], axis=1)
            y = self.data[self.label]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
            scaler = StandardScaler()
            scaled_X_Train = scaler.fit_transform(X=X_train)
            scaled_X_Test = scaler.transform(X=X_test)
            base_clf = LogisticRegression(solver='saga', multi_class='auto', max_iter=5000)
            param_grid = {
                'penalty': ['l1', 'l2', 'elasticnet'],
                'l1_ratio': [.1, .5, .9, .99],
                'C': np.logspace(0, 10, 5),
            }
            grid_clf = GridSearchCV(estimator=base_clf, param_grid=param_grid)
            grid_clf.fit(X=scaled_X_Train, y=y_train)
            predictions = grid_clf.predict(X=scaled_X_Test)
            trained_model = grid_clf
            self.log_clf = grid_clf
        return (trained_model, predictions, y_test)
    
    def __train_forest_classifier(self) -> tuple:
        '''
        return created model its predictions on test data and actual y values
        '''
        trained_model = None
        predictions = None
        y_test = None
        if self.train_randomforest_classifier:
            X = self.data.drop([self.label], axis=1)
            y = self.data[self.label]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
            base_clf = RandomForestClassifier(n_estimators=100, oob_score=False, bootstrap=True)
            param_grid = {
                'max_features': [3, 4],
                'max_depth': [3, 4]
            }
            grid_clf = GridSearchCV(estimator=base_clf, param_grid=param_grid)
            grid_clf.fit(X=X_train, y=y_train)
            predictions = grid_clf.predict(X_test)
            trained_model = grid_clf
            self.forest_clf = grid_clf
        return (trained_model, predictions, y_test)
    

    def __train_xgboost_classifier(self) -> tuple:
        '''
        return created model
        '''
        trained_model = None
        predictions = None
        if self.train_logistic_classifier:
            pass
        return
    
    def __train_lightgbm_classifier(self) -> tuple:
        '''
        return created model
        '''
        if self.train_lightgbm:
            pass
        return (0, 0, 0, 0)

    def __train_catboost_classifier(self) -> None:
        '''
        return created model
        '''
        if self.train_logistic_classifier:
            pass
        return (0, 0, 0, 0)
    

# select best model
class ClassifierSelection(ClassifierCreate):
    def __init__(self, dataset: pd.DataFrame, label: str, classifiers: list, exclude: list) -> None:
        super().__init__(dataset, label, classifiers, exclude)    

    # TODO - returns list of dictionaries. key is name of model, value is classification report 
    # TODO - or return dataframe. index is precision, recall & f1-score. Columns are model types
    def compare_and_select_model(self) -> None:
        log_model, log_preds, log_y_values = self.__train_logistic_classifier()
        forest_model, forest_preds, forest_y_values = self.__train_forest_classifier()
        xg_model, xg_preds, xg_y_values = self.__train_xgboost_classifier()
        lgbm_model, lgbm_preds, lgbm_y_values = self.__train_lightgbm_classifier()
        cat_model, cat_preds, cat_y_values = self.__train_catboost_classifier()
        pass