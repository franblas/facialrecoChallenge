# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 17:57:09 2015

@author: Paco
"""

from utils import Utils
from evaluate import Evaluate
from metrics import Metrics

# Load data
u = Utils()
train_facile = u.load_matrix('data/data_train_facile.mat')

#generate pairs
pairs_idx, pairs_label = u.generate_pairs(train_facile['label'], 1000, 0.1)

# Calculate distance
m = Metrics()
dist = m.braycurtis_dist(train_facile['X'], pairs_idx)

# Evaluate model
e = Evaluate()
e.evaluation(pairs_label,dist)
# display results
e.display_roc()
e.easy_score()

# Evaluate test dataset and save it
test_facile = u.load_matrix('data/data_test_facile.mat')
dist_test = m.braycurtis_dist(test_facile['X'], test_facile['pairs'])
u.save_test(dist_test)   