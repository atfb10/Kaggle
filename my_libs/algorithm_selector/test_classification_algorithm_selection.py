import pandas as pd
from classification_algorithm_selector import ClassifierSelection

# Constants
LOG = 'logistic'
F = 'randomforest'
XG = 'xgboost'
L = 'light'
CAT = 'catboost'
N_FOLDS = 5
LABEL = 'test_result'
FEATURES_TO_EXCLUDE = None



# Initialize & test
df = pd.read_csv('hearing_test.csv')
models = [LOG, F, XG, L, CAT]
model_comparison = ClassifierSelection(dataset=df, label=LABEL, cross_validation_folds=N_FOLDS, classifiers=models, exclude=None)
model_comparison.classification_report_scores()

# BUG: not instantiating models - need to instantiate attributes in parent class BEFORE calling train in ClassifierSelection