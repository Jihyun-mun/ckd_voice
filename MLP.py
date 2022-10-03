#!/usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy as cp
from typing import Tuple

import os
import warnings
warnings.filterwarnings('ignore')

data = '/home/jihyeon1202/ML/data/eGeMAPS.csv'
df = pd.read_csv(data)
df = df.drop(['filename'], axis=1)
#df = df.drop(['eGFR'], axis=1)
df = df.drop(['label_s'], axis=1)
# df = df.drop(['label'], axis=1)

df_X = df.drop(['label'], axis=1)
y = df['label']

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold, StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectFromModel

str_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=66)
#kfold = KFold(n_splits=5, random_state=66, shuffle=True)

def cross_val_predict(model, kfold : str_kf, X : np.array, y : np.array) -> Tuple[np.array, np.array, np.array]:
    model_ = cp.deepcopy(model)
    no_classes = len(np.unique(y))

    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, no_classes])

    for train_ndx, test_ndx in kfold.split(X, y):
        train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]

        actual_classes = np.append(actual_classes, test_y)
        #print(actual_classes)

        model_.fit(train_X, train_y)
        predicted_classes = np.append(predicted_classes, model_.predict(test_X))
        #print(predicted_classes)

        try:
            predicted_proba = np.append(predicted_proba, model_.predict_proba(test_X), axis=0)
        except:
            predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), no_classes), dtype=float), axis=0)

    return actual_classes, predicted_classes, predicted_proba

def plot_confusion_matrix(task, actual_classes : np.array, predicted_classes : np.array, sorted_labels : list):

    matrix = confusion_matrix(actual_classes, predicted_classes, labels=sorted_labels)
    print(classification_report(actual_classes, predicted_classes))
    
    if task == 'detection':
        new_labels = ['non-CKD', 'CKD']
        plt.figure(figsize=(12.8, 6))
        sns.heatmap(matrix, annot=True, xticklabels=new_labels, yticklabels = new_labels, cmap='Blues', fmt = 'g')
        plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
        plt.savefig('MLP_detection.png')

    else:
        new_labels = ['non-CKD', 'stage3', 'stage4']
        plt.figure(figsize=(12.8, 6))
        sns.heatmap(matrix, annot=True, xticklabels=new_labels, yticklabels = new_labels, cmap='Blues', fmt = 'g')
        plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion matrix')
        plt.savefig('MLP_prediction.png')


std = StandardScaler()
std_X = std.fit_transform(df_X)
X = pd.DataFrame(std_X, columns = df_X.columns)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
mlp = MLPClassifier()
    
parameters = {'hidden_layer_sizes': np.arange(1, 5),
                         'activation': ['logistic', 'tanh', 'relu'],
                         'solver': ['sgd', 'adam', 'lbfgs'],
                         'learning_rate': ['constant', 'adaptive'],
                         'learning_rate_init': np.arange(0.0001, 0.1)}
        
grid_search = GridSearchCV(estimator = mlp, param_grid = parameters, scoring = 'f1_weighted', cv = str_kf, verbose = 0)

grid_search.fit(X, y)

print('GridSearch CV best score: {:.4f}\n\n'.format(grid_search.best_score_))
print('Parameters that give the best results:', '\n\n', (grid_search.best_params_))
print('\n\nEstimator that was chosen by the search:', '\n\n', (grid_search.best_estimator_))
#print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))

best_mlp = MLPClassifier(**grid_search.best_params_)

actual_classes, predicted_classes, _ = cross_val_predict(best_mlp, str_kf, X.to_numpy(), y.to_numpy())

#print('actual: ', actual_classes)
#print('predicted: ', predicted_classes)

plot_confusion_matrix('detection', actual_classes, predicted_classes, [0, 1])


print('############################################# prediction #############################')

data = '/home/jihyeon1202/ML/data/eGeMAPS.csv'
df = pd.read_csv(data)
df = df.drop(['filename'], axis=1)
#df = df.drop(['eGFR'], axis=1)
#df = df.drop(['label_s'], axis=1)
df = df.drop(['label'], axis=1)

df_X = df.drop(['label_s'], axis=1)
y = df['label_s']

#X = std.fit_transform(X)
std_X = std.fit_transform(df_X)
X = pd.DataFrame(std_X, columns = df_X.columns)

mlp = MLPClassifier()
    
parameters = {'hidden_layer_sizes': np.arange(1, 5),
                         'activation': ['logistic', 'tanh', 'relu'],
                         'solver': ['sgd', 'adam', 'lbfgs'],
                         'learning_rate': ['constant', 'adaptive'],
                         'learning_rate_init': np.arange(0.0001, 0.1)}
        
grid_search = GridSearchCV(estimator = mlp, param_grid = parameters, scoring = 'f1_weighted', cv = str_kf, verbose = 0)

grid_search.fit(X, y)

print('GridSearch CV best score: {:.4f}\n\n'.format(grid_search.best_score_))
print('Parameters that give the best results:', '\n\n', (grid_search.best_params_))
print('\n\nEstimator that was chosen by the search:', '\n\n', (grid_search.best_estimator_))
#print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))

best_mlp = MLPClassifier(**grid_search.best_params_)

actual_classes, predicted_classes, _ = cross_val_predict(best_mlp, str_kf, X.to_numpy(), y.to_numpy())

plot_confusion_matrix('prediction', actual_classes, predicted_classes, [0, 3, 4])
