# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 17:57:09 2015

@author: Paco
"""

from utils import Utils
from evaluate import Evaluate
from metrics import Metrics
from gradient import Gradient
import numpy as np

# Load data
u = Utils()
train_facile = u.load_matrix('data/data_train_facile.mat')

#generate pairs
pairs_idx, pairs_label = u.generate_pairs(train_facile['label'], 1000, 0.1)
newX,newY = u.select_pairs_data(pairs_idx,train_facile['X'],train_facile['label'],c=700)
feat_idx = u._feat_idx

#test gradient
g = Gradient()
M_ini = g.generate_I(newX.shape[1])
M = g.sgd_metric_learning(newX, newY, 0.002, 50000, 0, M_ini)

# Calculate distance
m = Metrics()
X = u.select_features(train_facile['X'],feat_idx)
X -= X.mean(axis=0)
X /= X.std(axis=0)
X[np.isnan(X)] = 0.
dist = m.mahalanobis_dist(X, pairs_idx,M)
#dist[np.isnan(dist)] = 50.
## Evaluate model
e = Evaluate()
e.evaluation(pairs_label,dist)
## display results
e.display_roc()
e.easy_score()

# Evaluate test dataset and save it
test_facile = u.load_matrix('data/data_test_facile.mat')
#X2 = u.select_features(test_facile['X'],feat_idx)
#X2 -= X2.mean(axis=0)
#X2 /= X2.std(axis=0)
#X2[np.isnan(X2)] = 0.
#dist_test = m.mahalanobis_dist(X2, test_facile['pairs'],M)
#dist_test[np.isnan(dist_test)] = 1.
#u.save_test(dist_test)   
