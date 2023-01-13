#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 21:59:50 2022

@author: tomasnotenson
"""
import numpy as np
from skrmt.ensemble import GaussianEnsemble
from matplotlib import pyplot as plt
# sampling a GUE (beta=1) matrix of size 3x3
# gue = GaussianEnsemble(beta=2, n=2)
# print(gue.matrix)

#%% Chequeo IPR \sim Dim/2

def IPR(state):
    '''
    
    Compute inverse participation ratio defined as 
    
    
    \\xi = (\\sum_i p_i)^2/(\\sum_i p_i^2)
    

    Parameters
    ----------
    state : array_like
        Array with complex amplitudes.

    Returns
    -------
    res : float
        Inverse participation ratio.

    '''
    
    probs = np.abs(state)**2
    res = np.sum(probs)**2/np.sum(probs**2)
    return res

n = 20**2
beta = 2 # GUE ensemble

gue = GaussianEnsemble(beta=beta, n=n)
matriz = gue.matrix
e, evec = np.linalg.eig(matriz)

print(evec.shape)
IPRs = np.zeros((len(e)))
for ei in range(len(e)):
    state = evec[ei]
    IPRs[ei] = IPR(state)
    
index = np.arange(len(e))
plt.plot(index, IPRs, '-.')
plt.hlines(n/2, np.min(index), np.max(index), label='dim/2')
plt.xlabel('n')
plt.ylabel('IPR')
plt.grid()
# perfecto!
#%%

