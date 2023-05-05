'''
Adam Forestier
Last Updated: May 5, 2023
File used to write code in before transferring to Kaggle Notebook.
'''

# Imports
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns 

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier) 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, recall_score, confusion_matrix, classification_report, ConfusionMatrixDisplay)
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Read in dataset
df = pd.read_csv('diabetes_prediction_dataset.csv')

# Information
df.info()

# Can see there are no null values. Let's ensure there are no duplicates
df = df.drop_duplicates()

# Make Gender Binary instead of string
# df['gender'] = df['gender'].map({'Female': int(0), 'Male': int(1)})
# df = df.rename(columns={'gender':'gender_male'})

# Every age has .0 in decimal place. Let's convert to an integer as there is no month or day data; only age in years 
df['age'] = np.vectorize(lambda age: int(age))(df['age'])

# general data statistics
df.describe().transpose()

# Transform dataset
final_df = pd.get_dummies(df, drop_first=True)

# Let's investigate the correlation of each feature to the label we are trying to predict: diabetes
diabetes_correlation = final_df.corr()['diabetes'].sort_values()[:-1]
diabetes_correlation

print('-----------------------------------')
print(diabetes_correlation)

# **It appears blood_glucose_level and HbA1c_level are the two highest correlated features to diabetes**

# See how the label is distributed, in order to determine if class weight may need to be assigned to the classifier models
diabetic_count = df['diabetes'].value_counts()
print(diabetic_count)

# ** There appears to be approximately 1 diabetic for every 10 non diabetics in this dataset

# Visualization

# View Relationship between two strongest correlated feature and the label
sns.scatterplot(x='HbA1c_level', y='blood_glucose_level', data=final_df, hue='diabetes')
plt.xlabel('HbA1c Level')
plt.ylabel('Blood Glucose Level')
plt.title('Diabetic vs Non-Diabetic by HbA1c and Blood Glucose')
plt.show()

# ** We can see a very clear seperation between diabetics and non-diabetics based on HbA1c level and blood glucose level.

# ** This indicates to me, that a simple KNN model, or may be a good algorithms for this classification task. Lets Explore this relationship further.

# Boxplot of blood glucose and diabetes
sns.boxplot(x='diabetes', y='blood_glucose_level', data=final_df)
plt.xlabel('Diabetic')
plt.ylabel('Blood Glucose Level')
plt.title('Blood Glucose for Diabetic and Non-Diabetic')
plt.show()

# Boxplot of blood glucose and diabetes
sns.boxplot(x='diabetes', y='HbA1c_level', data=final_df)
plt.xlabel('Diabetic')
plt.ylabel('HbA1c Level')
plt.title('HbA1c for Diabetic and Non-Diabetic')
plt.show()

# ** high blood glucose and HbA1C_level are strong indicators of diabetes

# ** Let's checkout the remaining features

# Distribution of ages for those with and without diabetes
sns.displot(data=final_df, x='age', bins=50, col='diabetes', hue='diabetes')
plt.title('Distribution of Ages for Diabetics and Non-Diabetics')
plt.show()

# Distribution of BMI for those with and without diabetes
sns.displot(data=final_df, x='bmi', bins=50, col='diabetes', hue='diabetes')
plt.title('Distribution of Ages for Diabetics and Non-Diabetics')
plt.show()

# View count of diabetics and non-diabetics by hypertension 
sns.countplot(data=final_df, x='hypertension', hue='diabetes')
plt.xlabel('Hypertension')
plt.ylabel('Total Count')
plt.title('Hypertension and Diabetes')
plt.show()

# View count of diabetics and non-diabetics by heart disease 
sns.countplot(data=final_df, x='heart_disease', hue='diabetes')
plt.xlabel('Diabetic')
plt.ylabel('Total Count')
plt.title('Heart Disease and Diabetes')
plt.show()

# Count of diabetics and non diabetics by gender
sns.countplot(data=df, x='gender', hue='diabetes')
plt.xlabel('Diabetic')
plt.ylabel('Total Count')
plt.title('Gender and Diabetes')
plt.show()

# Count of diabetics and non diabetics by gender
sns.countplot(data=df, x='smoking_history', hue='diabetes')
plt.xlabel('Diabetic')
plt.ylabel('Total Count')
plt.title('Smoking History and Diabetes')
plt.show()

# ** Count for those with heart disease and hypertension with no diabetes exceeds those with both and heart disease...
# ** HOWEVER - we must remember the unbalanced dataset. They are near equal AND there are only 1/10 the amount of those with diabetes in the dataset!

# Classification Models

# ** With all of the following visualized. Let's start training some models with the findings we have gathered

# ** We are going to start with high bias and low variance (low complexity) and increase complexity
# ..The first model will be a very simple K Nearest Neighbors model utilizing only Blood Glucose and HbA1c

# Seperate features and label. 
X = final_df[['blood_glucose_level', 'HbA1c_level']]
y = final_df['diabetes']

# Perform train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=101)

# Scale data. Only fit training data to prevent data leakage
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# Create a K Nearest Neighbors model with search for optimum amount of neighbors between 1 - 10. Use Minkowski algorithm for distance calculation
param_grid = {
    'n_neighbors': list(range(1, 11)),
    'metric':['minkowski']
}
knn_clf = KNeighborsClassifier()
knn_clf = GridSearchCV(knn_clf, param_grid=param_grid, cv=5, scoring='accuracy')
knn_clf.fit(scaled_X_train, y_train)
y_pred = knn_clf.predict(scaled_X_test)
knn_params = knn_clf.best_estimator_.get_params()
knn_params
print(knn_params)

# Now that we have the best n_neighbors parameter. Create a model with those parameters
knn_clf = KNeighborsClassifier(n_neighbors=6, metric='minkowski')
knn_clf.fit(scaled_X_train, y_train)
y_pred = knn_clf.predict(scaled_X_test)

# Confusion matrix to display precision and recall. 
knn_cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
p = ConfusionMatrixDisplay(confusion_matrix=knn_cm, display_labels=knn_clf.classes_)
p.plot()
plt.show()

# Classification Report
knn_cr = classification_report(y_true=y_test, y_pred=y_pred)
print(knn_cr)

# .. This classifier performs very well in most facets. High precision and recall for non-diabetics. Overall accuracy of 97%
# .. The classifier also always 100% correct when assigning the diabetic label to a patient. However it is missing 1/3 of the positive cases
# .. There are too many false negatives. Considering the task of this model, to identify when individuals have diabetes. We need to have higher recall for class 1, even if it is at the expense of other scores
# .. Let us see if more complex models can perform recall better

sm = SMOTE(random_state=101)
X, y = sm.fit_resample(X, y)

# Now that we have the best n_neighbors parameter. Create a model with those parameters
knn_clf = KNeighborsClassifier(n_neighbors=6, metric='minkowski')
knn_clf.fit(scaled_X_train, y_train)
y_pred = knn_clf.predict(scaled_X_test)

# Confusion matrix to display precision and recall. 
knn_cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
p = ConfusionMatrixDisplay(confusion_matrix=knn_cm, display_labels=knn_clf.classes_)
p.plot()
plt.show()

# Classification Report
knn_cr = classification_report(y_true=y_test, y_pred=y_pred)
print(knn_cr)

# .. Let's try another distance based classifier. Support Vector Classifier using the 5 strongest correlated features
strongest_correlated_features = ['heart_disease', 'hypertension', 'bmi', 'age', 'HbA1c_level', 'blood_glucose_level']
X = final_df[strongest_correlated_features]
y = final_df['diabetes']
X, y = sm.fit_resample(X, y)

# Perform train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=101)

# Scale data. Only fit training data to prevent data leakage
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# Create a Support Vector Machine Classifier with cross validation
c = [.5, .75, .95, .99, 1]
degree = [1, 2, 3, 4]
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': c,
    'gamma': ['scale', 'auto'],
    'degree': degree,
}
svm_clf = SVC()
svm_clf = GridSearchCV(svm_clf, param_grid=param_grid, cv=2, scoring='accuracy')
svm_clf.fit(scaled_X_train, y_train)
y_pred = svm_clf.predict(scaled_X_test)
svm_params = svm_clf.best_estimator_.get_params()
svm_params
print(svm_params)

# {'C': 1, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0, 
# 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'poly', 
# 'max_iter': -1, 'probability': False, 'random_state': None, 'shrinking': True, 'tol': 0.001, 'verbose': False}
# Now that we have found the optimal parameters. From the first grid search, let's do 1 more fine-tuned grid search, 
# let's also search for best class-weight, as I hypothosize we need to increase the class weight to lower the number of false negatives

# ** Now that we have found the optimal parameters. Create two models model with those parameters: 1 with class weight=None and 2 with class weight='balanced' 
# ** NOTE: The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
# ** I hypothesize that balancing the class weight will result in better results; aka less False Negatives, as this is an unbalanced data set: 10x more people are non-diabetic as opposed to diabetic

# With no balancing
svm_clf = SVC(class_weight=None, degree=3, gamma='scale', C=.99999, kernel='poly')
svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)

# Confusion matrix to display precision and recall. 
svm_clf_cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
p = ConfusionMatrixDisplay(confusion_matrix=svm_clf_cm, display_labels=svm_clf.classes_)
p.plot()
plt.show()

# Classification Report
svm_clf_cr = classification_report(y_true=y_test, y_pred=y_pred)
print(svm_clf_cr)

# With balancing
svm_balanced_clf = SVC(class_weight='balanced', degree=3, gamma='scale', C=.99999, kernel='poly')
svm_balanced_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)

# Confusion matrix to display precision and recall. 
svm_balanced_clf_cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
p = ConfusionMatrixDisplay(confusion_matrix=svm_balanced_clf_cm, display_labels=svm_balanced_clf.classes_)
p.plot()
plt.show()

# Classification Report
svm_balanced_clf_cr = classification_report(y_true=y_test, y_pred=y_pred)
print(svm_balanced_clf_cr)

# **Slightly disappointing Results from the Support Vector Machine Classifiers. Let's try some Ensemble approaches**

# Seperate features and label
X = final_df.drop('diabetes', axis=1)
y = final_df['diabetes']
X, y = sm.fit_resample(X, y)

# Cross validated random forest NOTE: Always do this to get best parameters. commented out to not cook my poor computer
forest_clf = RandomForestClassifier(random_state=101, oob_score=True)
param_grid = {
    'n_estimators': [64, 100, 128, 200],
    'max_features': [2, 3, 4],
    'max_depth': [2, 3, 4],
    'bootstrap': [True, False],
    'criterion': ['entropy', 'gini']
}
grid_clf = GridSearchCV(estimator=forest_clf, param_grid=param_grid)
grid_clf.fit(X=X_train, y=y_train)

# See best model
best_params = grid_clf.best_params_ 
best_params #

# **Now that we have found the optimal parameters**
forest_clf = RandomForestClassifier(n_estimators=64, max_features=4, max_depth=3, bootstrap=True, criterion='entropy')

forest_clf.fit(X_train, y_train)
y_pred = forest_clf.predict(X_test)

# Confusion matrix to display precision and recall. 
forest_clf_cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
p = ConfusionMatrixDisplay(confusion_matrix=forest_clf_cm, display_labels=forest_clf.classes_)
p.plot()
plt.show()

# Classification Report
forest_clf_cr = classification_report(y_true=y_test, y_pred=y_pred)
print(forest_clf_cr)

# ** FINDINGS**

# **Let's try a AdaBoost meta learning approach, selecting the number of weak learnings through the Elbow Method**

# Here we create n models, 1 for each number of columns. We determine the best model based on having the lowest false negative score
recall_error = []
for i in range(1, len(final_df.columns)):
    clf = AdaBoostClassifier(n_estimators=i)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    recall_error.append(1 - recall_score(y_true=y_test, y_pred=y_pred))
plt.plot(range(1, len(final_df.columns)), recall_error)
plt.title('Recall Error by N_Estimators')
plt.xlabel('Number of Estimators')
plt.ylabel('Recall Error')

# Classifier with the lowest recall error
ada_clf = AdaBoostClassifier(n_estimators=14)
ada_clf.fit(X_train, y_train)
y_pred = ada_clf.predict(X_test)

# Show the features with importance > 0
feature_imp = pd.DataFrame(index=X, y=ada_clf.feature_importances_, columns=['Importance'])
feature_imp = feature_imp[feature_imp['Importance'] > 0.0001]
feature_imp = feature_imp.sort_values('Importance')
sns.barplot(x=feature_imp.index, y='Importance', data=feature_imp)
plt.xlabel('Feature')
plt.xlabel('Importance')
plt.title('Importance by Feature')

# Confusion matrix to display precision and recall. 
forest_clf_cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
p = ConfusionMatrixDisplay(confusion_matrix=forest_clf_cm, display_labels=forest_clf.classes_)
p.plot()
plt.show()

# Classification Report
forest_clf_cr = classification_report(y_true=y_test, y_pred=y_pred)
print(forest_clf_cr)

# **The model performs almost identically to the KNN model, but is much more complex:
#  We see again and again that HbA1c and Blood Glucose are the most important determiners.**

# **Let's try a Cross-Validated GradientBoost**


# Search for optimal hyperparameters
base_clf = GradientBoostingClassifier()
param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [.05, .1, .15, .2],
    'max_depth': [3, 4, 5]
}
grid_clf = GridSearchCV(estimator=base_clf, param_grid=param_grid)
grid_clf.fit(X_train, y_train)
best_params = grid_clf.best_params_ 
best_params

# Model with the best parameters
gradient_clf = GradientBoostingClassifier(n_estimators=, learning_rate=, max_depth=, random_state=101)
gradient_clf.fit(X_train, y_train)
y_pred = ada_clf.predict(X_test)

# Confusion matrix to display precision and recall. 
gradient_cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
p = ConfusionMatrixDisplay(confusion_matrix=gradient_cm, display_labels=forest_clf.classes_)
p.plot()
plt.show()

# Classification Report
gradient_cr = classification_report(y_true=y_test, y_pred=y_pred)
print(gradient_cr)

# **Another identical performer to KNN**

# **Final Model: Cross Validated LogisticRegression utizing ElasticNet Regularization**

# Grid search best hyperparameters
model = LogisticRegression(penalty='elasticnet', solver='saga', multi_class='ovr', max_iter=1000)
param_grid = {
    'l1_ratio': [.1, .75, .85, 9, .99, .999, 1],
    'C': np.logspace(0, 10, 10) # SciKit Learn recommends logrithmic spacing!
}
grid_model = GridSearchCV(estimator=model, param_grid=param_grid)
grid_model.fit(X=scaled_X_train, y=y_train)
best_params = grid_model.best_params_
best_params

# Model w/ best hyperparameters and balanced class weight
model = LogisticRegression(l1_ratio=.1, C=1, class_weight='balanced', penalty='elasticnet', solver='saga', multi_class='ovr')
model.fit(scaled_X_train, y_train)
y_pred = model.predict(scaled_X_test)

# Confusion matrix to display precision and recall. 
log_cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
p = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=model.classes_)
p.plot()
plt.show()

# Classification Report
log_cr = classification_report(y_true=y_test, y_pred=y_pred)
print(log_cr)

# Visualized ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
p = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Diabetes Result Estimator')
p.plot()


# Findings
'''
add Codeadd Markdown
Most models performed with a high level of accuracy, but low recall. This is likely the result of an imbalanced dataset, i.e. many more non-diabetic individuals than diabetic.

Depending on context, the two models I would consider utilizing are the K Nearest Neighbors Classifier, and the Random Forest Ensemble with balanced weight.

The K Nearest Neighbors Classifier has an accuracy of 97% and is wonderfully simple. It only relies upon two features: blood glucose and HbA1c levels to predict diabetes. It has high bias, meaning it is not overfitted and would perform similarly on new blood glucose and HbA1c data.

Likely, the model I would recommend the most, is the Random Forest Meta Learning Model. If the goal of the data set, is to predict when an individual has diabetes, this model performs that task the best; it has the highest recall. It does come at the cost of significantly higher precision, however, meaning that the false positive rate is much higher.

In practice, what the Random Forest Classifier would allow a physician to do is, view a patient's age, blood glucose and HbA1c levels. If those statistics fell below a certain percentage, it could be inferred from the model they are not at risk of diabetes. It they were predicted to be diabetic, further medical tests could be performed on the patient to determine whether or not they are in fact diabetic, or if the model falsely predicted them to be so (false positive).
'''