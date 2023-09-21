
'''
Adam Forestier
September 20, 2023
This file contains classes to compare multiple classification algorithms for a given dataset and rank the models
'''


'''
NOTE: Classification Only
NOTE: Not for simple models. If all I think I need is a simple KNN model, I should just make one for the problem at hand
NOTE: SVM not included. Too long to run. If I have a problem SVM's are a good selection for, just create one  
'''

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import (cross_val_score, cross_val_predict, GridSearchCV, train_test_split)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Constants
LOG = 'logistic'
F = 'randomforest'
XG = 'xgboost'
L = 'light'
CAT = 'catboost'

# Prep models for training
class ModelPrep():

    def __init__(self, dataset: pd.DataFrame, label: str, cross_validation_folds: int, classifiers: list, exclude=None) -> None: # Exclude is list, but if not included defaults to none
        self.data = dataset
        self.label = label
        self.classifiers = classifiers
        self.cv_folds = cross_validation_folds
        self.features_to_exclude = exclude
        self.train_logistic_classifier = True
        self.train_randomforest_classifier = True
        self.train_xgboost = True
        self.train_lightgbm = True
        self.train_catboost = True

        if classifiers:
            self.__check_models()
        if exclude:
            self.__drop_exclude_features()
        return
    
    def __drop_exclude_features(self) -> None:
        self.data = self.data.drop(self.features_to_exclude, axis=1)
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
    def __init__(self, dataset: pd.DataFrame, label: str, cross_validation_folds: int, classifiers: list, exclude: list) -> None:
        super().__init__(dataset, label, cross_validation_folds, classifiers, exclude)
        self.log_clf = None
        self.forest_clf = None
        self.xgboost_clf = None
        self.lightgbm_clf = None
        self.cat_clf = None

    def __train_logistic_classifier(self) -> tuple:
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
            predictions = cross_val_predict(estimator=grid_clf, X=scaled_X_Test, y=y_test, cv=self.cv_folds)
            # predictions = grid_clf.predict(X=scaled_X_Test)
            trained_model = grid_clf
            self.log_clf = grid_clf
        return (trained_model, predictions, y_test)
    
    def __train_forest_classifier(self) -> tuple:
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
                'max_depth': [3, 5]
            }
            grid_clf = GridSearchCV(estimator=base_clf, param_grid=param_grid)
            grid_clf.fit(X=X_train, y=y_train)
            predictions = cross_val_predict(estimator=grid_clf, X=X_test, y=y_test, cv=self.cv_folds)
            # predictions = grid_clf.predict(X_test)
            trained_model = grid_clf
            self.forest_clf = grid_clf
        return (trained_model, predictions, y_test)
    

    def __train_xgboost_classifier(self) -> tuple:
        trained_model = None
        predictions = None
        y_test = None
        if self.train_logistic_classifier:
            X = self.data.drop([self.label], axis=1)
            y = self.data[self.label]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
            base_xgboost = XGBClassifier(n_estimators=100, subsample=1.0)
            param_grid = {
                'learning_rate': [.05, .1, .3],
                'max_depth': [3, 5]
            }
            grid_clf = GridSearchCV(estimator=base_xgboost, param_grid=param_grid)
            grid_clf.fit(X=X_train, y=y_train)
            predictions = cross_val_predict(estimator=grid_clf, X=X_test, y=y_test, cv=self.cv_folds)
            # predictions = grid_clf.predict(X=X_test)
            trained_model = grid_clf
            self.xgboost_clf = grid_clf
        return (trained_model, predictions, y_test)
    
    def __train_lightgbm_classifier(self) -> tuple:
        '''
        return created model
        '''
        trained_model = None
        predictions = None
        y_test = None
        if self.train_lightgbm:
            X = self.data.drop([self.label], axis=1)
            y = self.data[self.label]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
            base_lightgbm = LGBMClassifier(n_estimators=100, subsample=1.0)
            param_grid = {
                'learning_rate': [.05, .1, .3],
                'max_depth': [3, 5]
            }
            grid_clf = GridSearchCV(estimator=base_lightgbm, param_grid=param_grid)
            grid_clf.fit(X=X_train, y=y_train)
            predictions = cross_val_predict(estimator=grid_clf, X=X_test, y=y_test, cv=self.cv_folds)
            # predictions = grid_clf.predict(X=X_test)
            trained_model = grid_clf
            self.lightgbm_clf = grid_clf
        return (trained_model, predictions)

    def __train_catboost_classifier(self) -> None:
        '''
        return created model
        '''
        trained_model = None
        predictions = None
        y_test = None
        if self.train_logistic_classifier:
            X = self.data.drop([self.label], axis=1)
            y = self.data[self.label]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
            base_cat = CatBoostClassifier(iterations=100, subsample=1.0)
            param_grid = {
                'learning_rate': [.05, .1, .3],
                'depth': [3, 5]
            }
            grid_clf = GridSearchCV(estimator=base_cat, param_grid=param_grid)
            grid_clf.fit(X=X_train, y=y_train)
            predictions = cross_val_predict(estimator=grid_clf, X=X_test, y=y_test, cv=self.cv_folds)
            # predictions = grid_clf.predict(X=X_test)
            trained_model = grid_clf
            self.cat_clf = grid_clf
        return (trained_model, predictions, y_test)
    
class ClassifierSelection(ClassifierCreate):
    def __init__(self, dataset: pd.DataFrame, label: str, cross_validation_folds: int, classifiers: list, exclude: list) -> None:
        super().__init__(dataset, label, cross_validation_folds, classifiers, exclude)
    
    def classification_report_scores(self) -> None:
        c_reports = []
        if self.log_clf:
            log_model, log_preds, log_y_test = self.__train_logistic_classifier()
            logistic_regression_classification_report = classification_report(y_true=log_y_test, y_pred=log_preds)
            c_reports.append(logistic_regression_classification_report)
        if self.forest_clf:
            forest_model, forest_preds, forest_y_test = self.__train_forest_classifier()
            random_forest_classification_report = classification_report(y_true=forest_y_test, y_pred=forest_preds)
            c_reports.append(random_forest_classification_report)
        if self.xgboost_clf:    
            xg_model, xg_preds, xg_y_test = self.__train_xgboost_classifier()
            xg_boost_classification_report = classification_report(y_true=xg_y_test, y_pred=xg_preds)
            c_reports.append(xg_boost_classification_report)
        if self.lightgbm_clf:
            lgbm_model, lgbm_preds, lgbm_y_test = self.__train_lightgbm_classifier()
            lightgbm_classification_report = classification_report(y_true=lgbm_y_test, y_pred=lgbm_preds)
            c_reports.append(lightgbm_classification_report)
        if self.cat_clf:
            cat_model, cat_preds, cat_y_test = self.__train_catboost_classifier()
            catboost_classification_report = classification_report(y_true=cat_y_test, y_pred=cat_preds)
            c_reports.append(catboost_classification_report)

        for report in c_reports:
            print(report)