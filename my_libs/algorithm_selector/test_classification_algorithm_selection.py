import pandas as pd
from classification_algorithm_selector import ClassifierSelect

# Constants
LOG = 'logistic'
F = 'randomforest'
XG = 'xgboost'
L = 'light'
CAT = 'catboost'
N_FOLDS = 5
TEST1_LABEL = 'test_result'
TEST2_LABEL = 'species'
FEATURES_TO_EXCLUDE = None
TEST1 = 'hearing_test.csv'
TEST2 = 'iris.csv'

# Initialize & test
models = [LOG, LOG, F, XG, L, CAT]
df = pd.read_csv(TEST2)
model_comparison = ClassifierSelect(dataset=df, label=TEST2_LABEL, classifiers=models, cross_validation_folds=N_FOLDS, exclude=None)
model_comparison.classification_report_scores()