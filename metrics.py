# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 17:56:31 2015

@author: Paco
"""

import numpy as np

class Metrics(object):
    
    _batch_size = 10000    
    
    def __init__(self): pass
    
    '''
    http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.spatial.distance.braycurtis.html#scipy.spatial.distance.braycurtis
    Score = 0.214275245493
    '''
    def braycurtis_dist(self,X,pairs):
        n_pairs = pairs.shape[0]
        dist = np.ones((n_pairs,), dtype=np.dtype("float32"))
        for a in range(0, n_pairs, self._batch_size):
            b = min(a + self._batch_size, n_pairs)
            #dist[a:b] = np.sqrt(np.sum((X[pairs[a:b, 0], :] - X[pairs[a:b, 1], :]) ** 2, axis=1))
            up = np.sum(np.abs(X[pairs[a:b, 0], :] - X[pairs[a:b, 1], :]),axis=1)
            down = np.sum(np.abs(X[pairs[a:b, 0], :] + X[pairs[a:b, 1], :]),axis=1)
            dist[a:b] = up / down
        return dist
    
    '''
    http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.spatial.distance.cosine.html
    Score = 0.357907078998    
    '''
    def cosin_dist(self,X, pairs):    
        n_pairs = pairs.shape[0]
        dist = np.ones((n_pairs,), dtype=np.dtype("float32"))
        for a in range(0, n_pairs, self._batch_size):
            b = min(a + self._batch_size, n_pairs)
            #dist[a:b] = cosine(X[pairs[a:b, 0], :],X[pairs[a:b, 1], :])
            up = np.sum(X[pairs[a:b, 0]]*X[pairs[a:b, 1]],axis=1)
            d1 = np.linalg.norm(X[pairs[a:b, 0]],axis=1)
            d2 = np.linalg.norm(X[pairs[a:b, 1]],axis=1)
            dist[a:b] = np.array(1.0 - (up / (d1*d2)))
        return dist    
    
    def euc_dist(self,X, pairs):
        """Compute an array of Euclidean distances between points indexed by pairs
    
        To make it memory-efficient, we compute the array in several batches.
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Data matrix
        pairs : array, shape (n_pairs, 2)
            Pair indices
        batch_size : int
            Batch size (the smaller, the slower but less memory intensive)
            
        Output
        ------
        dist : array, shape (n_pairs,)
            The array of distances
        """
        n_pairs = pairs.shape[0]
        dist = np.ones((n_pairs,), dtype=np.dtype("float32"))
        for a in range(0, n_pairs, self._batch_size):
            b = min(a + self._batch_size, n_pairs)
            dist[a:b] = np.sqrt(np.sum((X[pairs[a:b, 0], :] - X[pairs[a:b, 1], :]) ** 2, axis=1))
        return dist
        
        