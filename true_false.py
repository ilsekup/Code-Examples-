import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import seaborn as sns
import xgboost as xgb
import pandas as pd

def true_false_map(predictions, y_test, method):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for i in range(len(predictions)):
        if predictions[i] == 1 and y_test[i] ==1:
            true_positive += 1
        elif predictions[i] == 1 and y_test[i] == 0:
            false_positive += 1
        elif predictions[i] == 0 and y_test[i] == 0:
            true_negative += 1
        elif predictions[i] == 0 and y_test[i] == 1:
            false_negative += 1

    results = np.array([[true_negative, true_positive], [false_negative, false_positive]])
    results = pd.DataFrame(results, index=['True', 'False'], columns=['Negative', 'Positive'])
    sns.heatmap(results, annot=True)
    plt.title(method)
    plt.show()
