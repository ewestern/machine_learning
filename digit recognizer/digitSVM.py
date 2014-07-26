import pandas as pd
import numpy as np
from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier as RF
# from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler

def run_cv(X,y,clf_class,**kwargs):
    # Construct a kfolds object
    kf = KFold(len(y),n_folds=5,shuffle=True)
    y_pred = y.copy()
    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        # prediction = 
        # print "prediction %s" % prediction.dtype        
        y_pred[test_index] = clf.predict(X_test).astype(int)
    return y_pred



def accuracy(y_true,y_pred):
    # NumPy interpretes True and False as 1. and 0.
    return np.mean(y_true == y_pred)

if __name__ == '__main__':
  digits = pd.read_csv('train.csv')
  y = digits['label']
  digits = digits.drop(['label'], axis=1)
  scaler = StandardScaler()
  X = scaler.fit_transform(digits.as_matrix().astype(np.float))
  print "accuracy %.3f" % accuracy(y, run_cv(X, y, SVC))