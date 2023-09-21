
'''
Adam Forestier
September 21, 2023
This file contains a class to train multiple classification algorithms and display the classification reports for comparison
'''

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import (cross_val_predict, GridSearchCV, train_test_split)
from sklearn.preprocessing import (LabelEncoder, StandardScaler)
from xgboost import XGBClassifier

# Constants
LOG = 'logistic'
F = 'randomforest'
XG = 'xgboost'
L = 'light'
CAT = 'catboost'
    
class ClassifierSelect():
    def __init__(self, dataset: pd.DataFrame, label: str, cross_validation_folds: int, classifiers: list, exclude: list) -> None:
        """
        Initialize the ClassifierSelect class.

        Args:
            dataset (pd.DataFrame): The input dataset containing features and labels.
            label (str): The name of the label column in the dataset.
            cross_validation_folds (int): The number of cross-validation folds to use.
            classifiers (list): A list of classifier names to train and evaluate.
            exclude (list): A list of feature names to exclude from the dataset.
        """
        self.data = dataset
        self.label = label
        self.cv_folds = cross_validation_folds
        self.models = classifiers
        self.excluded_features = exclude
        self.log_clf = None
        self.forest_clf = None
        self.xgboost_clf = None
        self.lightgbm_clf = None
        self.cat_clf = None
        if self.excluded_features:
            self.__drop_exclude_features()
        return
    
    def __drop_exclude_features(self) -> None:
        """
        Drop excluded features from the dataset.
        """
        self.data = self.data.drop(self.features_to_exclude, axis=1)
        return
    
    def __train_logistic_classifier(self) -> tuple:
        """
        Train a logistic regression classifier.

        Returns:
            tuple: A tuple containing the trained model, predictions, and true labels.
        """
        trained_model = None
        predictions = None
        y_test = None
        print(self.models)
        if "logistic" in self.models:
            label_encoder = LabelEncoder()
            self.data[self.label] = label_encoder.fit_transform(self.data[self.label])
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
            trained_model = grid_clf
            self.log_clf = grid_clf
        return (trained_model, predictions, y_test)
    
    def __train_forest_classifier(self) -> tuple:
        """
        Train a random forest classifier.

        Returns:
            tuple: A tuple containing the trained model, predictions, and true labels.
        """
        trained_model = None
        predictions = None
        y_test = None
        if F in self.models:
            label_encoder = LabelEncoder()
            self.data[self.label] = label_encoder.fit_transform(self.data[self.label])
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
            trained_model = grid_clf
            self.forest_clf = grid_clf
        return (trained_model, predictions, y_test)
    

    def __train_xgboost_classifier(self) -> tuple:
        """
        Train a xgboost classifier.

        Returns:
            tuple: A tuple containing the trained model, predictions, and true labels.
        """
        trained_model = None
        predictions = None
        y_test = None
        if XG in self.models:
            label_encoder = LabelEncoder()
            self.data[self.label] = label_encoder.fit_transform(self.data[self.label])
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
            trained_model = grid_clf
            self.xgboost_clf = grid_clf
        return (trained_model, predictions, y_test)
    
    def __train_lightgbm_classifier(self) -> tuple:
        """
        Train a LightGBM classifier.

        Returns:
            tuple: A tuple containing the trained model, predictions, and true labels.
        """
        trained_model = None
        predictions = None
        y_test = None
        if L in self.models:
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
            trained_model = grid_clf
            self.lightgbm_clf = grid_clf
        return (trained_model, predictions, y_test)

    def __train_catboost_classifier(self) -> None:
        """
        Train a catboost classifier.

        Returns:
            tuple: A tuple containing the trained model, predictions, and true labels.
        """
        trained_model = None
        predictions = None
        y_test = None
        if CAT in self.models:
            print('CAT Found')
            X = self.data.drop([self.label], axis=1)
            y = self.data[self.label]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
            base_cat = CatBoostClassifier(iterations=100)
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
    
    def classification_report_scores(self) -> None:
        '''
        Display classification reports for multiple classifiers
        '''
        c_reports = []
        log_model, log_preds, log_y_test = self.__train_logistic_classifier()
        logistic_regression_classification_report = classification_report(y_true=log_y_test, y_pred=log_preds)
        c_reports.append(logistic_regression_classification_report)
        forest_model, forest_preds, forest_y_test = self.__train_forest_classifier()
        random_forest_classification_report = classification_report(y_true=forest_y_test, y_pred=forest_preds)
        c_reports.append(random_forest_classification_report)
        xg_model, xg_preds, xg_y_test = self.__train_xgboost_classifier()
        xg_boost_classification_report = classification_report(y_true=xg_y_test, y_pred=xg_preds)
        c_reports.append(xg_boost_classification_report)
        lgbm_model, lgbm_preds, lgbm_y_test = self.__train_lightgbm_classifier()
        lightgbm_classification_report = classification_report(y_true=lgbm_y_test, y_pred=lgbm_preds)
        c_reports.append(lightgbm_classification_report)
        cat_model, cat_preds, cat_y_test = self.__train_catboost_classifier()
        catboost_classification_report = classification_report(y_true=cat_y_test, y_pred=cat_preds)
        c_reports.append(catboost_classification_report)

        print('\n')
        for report in c_reports:
            print(f'{report}\n')