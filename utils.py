# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 17:42:56 2015

@author: Paco
"""

import numpy as np
from scipy import io

class Utils(object):
    
    _feat_idx = None    
    
    def __init__(self): pass

    def generate_pairs(self,label, n_pairs, positive_ratio, random_state=42):
        rng = np.random.RandomState(random_state)
        n_samples = label.shape[0]
        pairs_idx = np.zeros((n_pairs, 2), dtype=int)
        pairs_idx[:, 0] = rng.randint(0, n_samples, n_pairs)
        rand_vec = rng.rand(n_pairs)
        for i in range(n_pairs):
            if rand_vec[i] <= positive_ratio:
                idx_same = np.where(label == label[pairs_idx[i, 0]])[0]
                idx2 = rng.randint(idx_same.shape[0])
                pairs_idx[i, 1] = idx_same[idx2]
            else:
                idx_diff = np.where(label != label[pairs_idx[i, 0]])[0]
                idx2 = rng.randint(idx_diff.shape[0])
                pairs_idx[i, 1] = idx_diff[idx2]
        pairs_label = 2.0 * (label[pairs_idx[:, 0]] == label[pairs_idx[:, 1]]) - 1.0
        return pairs_idx, pairs_label
 
    def select_pairs_data(self,pairs_idx,X,y,c=150):
        temp = np.ndarray.flatten(pairs_idx)
        res = list()
        res2 = list()
        for t in temp:
            res.append(X[t])
            res2.append(y[t])
        yo = np.random.randint(0,X.shape[1]-1,c)
        self._feat_idx = yo
        tmp = np.array(res)
        res3 = list()
        for y in yo:
            res3.append(tmp[:,y])
        return np.array(res3).T,np.array(res2)    
   
    def reduce_features(self,X,c=150):
        yo = np.random.randint(0,X.shape[1]-1,c)
        res = list()
        for y in yo:
            res.append(X[:,y])
        return np.array(res).T
    
    def select_features(self,X,feat_idx):
        res = list()
        for f in feat_idx:
            res.append(X[:,f])
        return np.array(res).T    
   
    def save_test(self,dist_test,filetxt='soumission_facile.txt'):
        np.savetxt(filetxt, dist_test, fmt='%.5f')
        
    def load_matrix(self,path=''):
        return io.loadmat(path)
        
    '''
    ---------------------------------------------------
     Save / Load python numpy arrays (.bin)
    ---------------------------------------------------
    '''
    def saveBinaryMatrix(self,X,filename=''):
        f = file(filename,'wb')
        np.save(f,X)
        f.close()
 
    def loadBinaryMatrix(self,filename=''):
         f = file(filename,'rb')
         X = np.load(f)
         f.close()
         return X    
    