#techniques.py

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    print("loading x")
    X_train = np.load('x-train-data.npy')
    X_test = np.load('x-test-data.npy')
    print("loading y")
    y_train = np.load('y-train-data.npy')
    y_test = np.load('y-test-data.npy')

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_data()
"""
print("fitting log regression model")
# fit a logistic regression model
clf = LogisticRegression(solver='newton-cg', verbose=1).fit(X_train, y_train)
# computing the accuracy
accuracy = accuracy_score(clf.predict(X_test), y_test)
print('my obtained accuracy on Iris dataset using log-regression is: {0}'.format(accuracy))
# computing the confusion matrix
confMat = confusion_matrix(clf.predict(X_test), y_test)
print('my obtained confusion matrix on Iris dataset using log-regression is:\n {0}'.format(confMat))
# classification report
print(classification_report(clf.predict(X_test), y_test))
"""
print("fitting knn model")
# fitting kNN model
knnModel = KNeighborsClassifier(n_neighbors=5).fit(X_train,y_train)
accuracy = accuracy_score(knnModel.predict(X_test), y_test)
print('my obtained accuracy on Iris dataset using KNN is: {0}'.format(accuracy))
confMat = confusion_matrix(knnModel.predict(X_test), y_test)
print('my obtained confusion matrix on Iris dataset using knn is:\n {0}'.format(confMat))
print(classification_report(knnModel.predict(X_test), y_test))
