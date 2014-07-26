# from __future__ import division
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler



# We don't need these columns




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
  churn_df = pd.read_csv('churn.csv')
  churn_result = churn_df['Churn?']
  y = np.where(churn_result == 'True.',1,0)
  to_drop = ['State','Area Code','Phone','Churn?']
  churn_feat_space = churn_df.drop(to_drop,axis=1)

  # 'yes'/'no' has to be converted to boolean values
  # NumPy converts these from boolean to 1. and 0. later
  yes_no_cols = ["Int'l Plan","VMail Plan"]
  churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'

  # Pull out features for future use
  features = churn_feat_space.columns
  scaler = StandardScaler()

  X = scaler.fit_transform(churn_feat_space.as_matrix().astype(np.float))
  print "Support vector machines:"
  print "%.3f" % accuracy(y, run_cv(X,y,SVC))
  print "Random forest:"
  print "%.3f" % accuracy(y, run_cv(X,y,RF))
  print "K-nearest-neighbors:"
  print "%.3f" % accuracy(y, run_cv(X,y,KNN))