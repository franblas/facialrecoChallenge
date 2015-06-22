# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 21:24:42 2015

@author: Paco
"""

import numpy as np
from progressbar import *

class Gradient(object):
    
    def __init__(self): pass

    def generate_I(self,shape):
        return np.eye(shape)

    def psd_proj(self,M):
        """ projection de la matrice M sur le cone des matrices semi-definies
        positives"""
        eigenval, eigenvec = np.linalg.eigh(M)
        ind_pos = eigenval > 1e-10
        M = np.dot(eigenvec[:, ind_pos] * eigenval[ind_pos][np.newaxis, :],
                   eigenvec[:, ind_pos].T)
        return M

    def hinge_loss_pairs(self,X, pairs_idx, y_pairs, M):
        #Calcul du hinge loss sur les paires
        diff = X[pairs_idx[:, 0], :] - X[pairs_idx[:, 1], :]
        return np.maximum(0.0, 1.0 + y_pairs.T * (np.sum(
                                     np.dot(M, diff.T) * diff.T, axis=0) - 2.0))

    def sgd_metric_learning(self,X, y, gamma, n_iter, n_eval, M_ini, random_state=42):
        """Stochastic gradient algorithm for metric learning
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data
        y : array, shape (n_samples,)
            The targets.
        gamma : float | callable
            The step size. Can be a constant float or a function
            that allows to have a variable step size
        n_iter : int
            The number of iterations
        n_eval : int
            The number of pairs to evaluate the objective function
        M_ini : array, shape (n_features,n_features)
            The initial value of M
        random_state : int
            Random seed to make the algorithm deterministic
        """
        rng = np.random.RandomState(random_state)
        n_samples = X.shape[0]
        # tirer n_eval paires aleatoirement
        #pairs_idx = rng.randint(0, n_samples, (n_eval, 2))
        # calcul du label des paires
        #y_pairs = 2.0 * (y[pairs_idx[:, 0]] == y[pairs_idx[:, 1]]) - 1.0
        M = M_ini.copy()
        #pobj = np.zeros(n_iter)
        
        if not callable(gamma):
            gamma_func = lambda t: gamma
        else:
            gamma_func = gamma
    
        widgets = ['', Percentage(), ' ', Bar(marker='*',left='[',right=']'),' ', ETA()]        
        pbar = ProgressBar(widgets=widgets,maxval=n_iter)
        for t in pbar(range(n_iter)):
            #pobj[t] = np.mean(hinge_loss_pairs(X, pairs_idx, y_pairs, M))
            idx = rng.randint(0, n_samples, 2)
            diff = X[idx[0], :] - X[idx[1], :]
            y_idx = 2.0 * (y[idx[0]] == y[idx[1]]) - 1.0
            gradient = (y_idx * np.outer(diff, diff) *
                        ((1.0 + y_idx * (np.dot(diff, np.dot(M, diff.T)) - 2.0)) > 0))
            M -= gamma_func(t) * gradient
            M = self.psd_proj(M)
        return M    
        #return M, pobj
    
'''
n_features = X.shape[1]
M_ini = np.eye(n_features)
M, pobj = sgd_metric_learning(X, y, 0.002, 10000, 1000, M_ini)
'''    
    
    