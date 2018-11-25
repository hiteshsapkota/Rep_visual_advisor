#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 15:59:29 2018

@author: hiteshsapkota
"""

from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification

def predict(train_X, train_y, test_X):
    """Linear SVM is used"""
    clf = LinearSVC (random_state = 0, tol = 1e-14)
    clf.fit(train_X, train_y)
    test_y = clf.predict(test_X)
    return test_y

if __name__=='__main__':
    train_X, train_y = make_classification(n_features=4, random_state=0)
    test_X = [[0, 0, 0, 2], [0, 5, 3, 6]]
    predicted_values = predict(train_X, train_y, test_X)
    print(predicted_values.tolist())
    
