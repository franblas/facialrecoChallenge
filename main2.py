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
train_hard = u.load_matrix('data/data_train_difficile.mat')

#generate pairs
pairs_idx, pairs_label = u.generate_pairs(train_hard['label'], 1000, 0.1)

# Calculate distance
m = Metrics()
dist = m.braycurtis_dist(train_hard['X'], pairs_idx)

# Evaluate model
e = Evaluate()
e.evaluation(pairs_label,dist)
# display results
e.display_roc()
e.hard_score()

# Evaluate test dataset and save it
test_hard = u.load_matrix('data/data_test_difficile.mat')
dist_test = m.braycurtis_dist(test_hard['X'], test_hard['pairs'])
u.save_test(dist_test,filetxt='soumission_dur.txt')   