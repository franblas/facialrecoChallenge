# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 17:45:25 2015

@author: Paco
"""

import matplotlib.pyplot as plt
import bisect
import numpy as np
from sklearn import metrics

class Evaluate(object):
    
    _fpr = None
    _tpr = None  

    def __init__(self): pass
    
    def evaluation(self,pairs_label,dist):
        fpr, tpr, thresholds = metrics.roc_curve(pairs_label, -dist)
        self._fpr = fpr
        self._tpr = tpr

    def display_roc(self):
        plt.clf()
        plt.plot(self._fpr, self._tpr, label='ROC curve')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()
        
    def easy_score(self):
        easy_score = 1.0 - self._tpr[bisect.bisect(self._fpr, 0.001) - 1]
        print 'Easy score : '+str(easy_score)
        return easy_score
        
    def hard_score(self):
        idx = (np.abs(self._fpr + self._tpr - 1.)).argmin()
        hard_score = (self._fpr[idx]+(1-self._tpr[idx]))/2
        print 'Hard score : '+str(hard_score)
        return hard_score
        