import numpy as np
import pandas as pd

data = pd.read_csv("./heart.csv")
data.head(5)

X = data.drop('target', 1)
X = X.drop('ca', 1)
X = X.drop('thal', 1)
X = X.drop('slope', 1)
y = data['target']
### train test split
from sklearn.model_selection import train_test_split
# using test_size to specify the ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
train_set = pd.concat([X_train, y_train], axis = 1)
train_set['target'].describe()
train_set['target'].value_counts()

### feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)


X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

### Kernel SVM Trained on heart.csv file (original dataset)
import numpy as np
### Kernel SVM
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_C = [0.1, 1.0, 10.0, 100.0, 1000.0]
param_gamma = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
svm = SVC(random_state=0)

param_grid = [{'C': param_C,
               'gamma' : param_gamma,
               'kernel': ['rbf']}]
gs = GridSearchCV(estimator=svm,
                  param_grid=param_grid,
                  scoring='accuracy')
X_train_std = X_train
X_test_std = X_test
gs = gs.fit(X_train_std, y_train)
print(gs.best_score_)
print(gs.best_params_)

from sklearn.metrics import accuracy_score
svm = gs.best_estimator_
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)
y_test = np.array(y_test)

FN = 0
FP = 0
TN = 0
TP = 0
for i in range(len(y_pred)):
    if y_test[i] == 0 and y_pred[i] == 0:
        TN += 1
    if y_test[i] == 0 and y_pred[i] == 1:
        FP += 1
    if y_test[i] == 1 and y_pred[i] == 0:
        FN += 1
    if y_test[i] == 1 and y_pred[i] == 1:
        TP += 1
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1_score = 2 * (recall * precision) / (recall + precision)

print("Results for the Kernel SVM model trained on heart.csv")
print("Precision: {}".format(precision))
print("Recall: {}".format(recall))
print("F1_Score: {}".format(F1_score))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print()

### use previous model to test new dataset
Test_Data = pd.read_csv("./hungarian_dataset.csv")
X2 = Test_Data.drop('target', 1)
X2 = X2.drop('ca', 1)
X2 = X2.drop('thal', 1)
X2 = X2.drop('slope', 1)
y2 = Test_Data['target']
y2 = np.array(y2)
for i in range(len(y2)):
    if y2[i] >= 1:
        y2[i] = 1
    else:
        y2[i] = 0
sc2 = StandardScaler()
sc2.fit(X2)
X2_std = sc2.transform(X2)

y_pred = svm.predict(X2_std)
y_test = y2
y_test = np.array(y_test)

FN = 0
FP = 0
TN = 0
TP = 0
for i in range(len(y_pred)):
    if y_test[i] == 0 and y_pred[i] == 0:
        TN += 1
    if y_test[i] == 0 and y_pred[i] == 1:
        FP += 1
    if y_test[i] == 1 and y_pred[i] == 0:
        FN += 1
    if y_test[i] == 1 and y_pred[i] == 1:
        TP += 1

print("Results for the Kernel SVM model tested on hungarian_dataset.csv")
print("FN: {}".format(FN))
print("FP: {}".format(FP))
print("TN: {}".format(TN))
print("TP: {}".format(TP))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print()



data = pd.read_csv("./all.csv")
data.head(5)

X = data.drop('target', 1)
X = X.drop('ca', 1)
X = X.drop('thal', 1)
X = X.drop('slope', 1)
y = data['target']
### train test split
from sklearn.model_selection import train_test_split
# using test_size to specify the ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
train_set = pd.concat([X_train, y_train], axis = 1)
train_set['target'].describe()
train_set['target'].value_counts()

### feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

### Kernel SVM Trained on all.csv file (combined heart.csv and hungarian_dataset.csv)
import numpy as np
### Kernel SVM
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
param_C = [0.1, 1.0, 10.0, 100.0, 1000.0]
param_gamma = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
svm = SVC(random_state=0)

param_grid = [{'C': param_C,
               'gamma' : param_gamma,
               'kernel': ['rbf']}]
gs = GridSearchCV(estimator=svm,
                  param_grid=param_grid,
                  scoring='accuracy')

gs = gs.fit(X_train_std, y_train)
print(gs.best_score_)
print(gs.best_params_)

from sklearn.metrics import accuracy_score
svm = gs.best_estimator_
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)
y_test = np.array(y_test)

FN = 0
FP = 0
TN = 0
TP = 0
for i in range(len(y_pred)):
    if y_test[i] == 0 and y_pred[i] == 0:
        TN += 1
    if y_test[i] == 0 and y_pred[i] == 1:
        FP += 1
    if y_test[i] == 1 and y_pred[i] == 0:
        FN += 1
    if y_test[i] == 1 and y_pred[i] == 1:
        TP += 1
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1_score = 2 * (recall * precision) / (recall + precision)

print("Results for the Kernel SVM model trained on all.csv (combined heart.csv and hungarian_dataset.csv)")
print("Precision: {}".format(precision))
print("Recall: {}".format(recall))
print("F1_Score: {}".format(F1_score))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
