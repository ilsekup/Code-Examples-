import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import xgboost as xgb
from true_false import true_false_map


# Load the data
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state=0)



#xgboost

xg_clf = xgb.XGBClassifier()
xg_clf.fit(X_train,y_train)
print("Test set accuracy with XGBoost: {:.3f}".format(xg_clf.score(X_test,y_test)))

predict_xgb = xg_clf.predict(X_test)

true_false_map(predict_xgb, y_test, 'XG-boost')


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)



#xgboost

xg_clf = xgb.XGBClassifier()
xg_clf.fit(X_train_scaled,y_train)
print("Test set accuracy with XGBoost scaled: {:.3f}".format(xg_clf.score(X_test_scaled,y_test)))

predict_xgb = xg_clf.predict(X_test_scaled)
