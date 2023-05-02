'''
Adam Forestier
Last Updated: May 2, 2023
File used to write code in before transferring to Jupyter Notebook
'''

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier) 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay)
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Read in dataset
df = pd.read_csv('diabetes_prediction_dataset.csv')

# Information

# 