#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 17:51:01 2021

@author: usuario
"""
import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from math import factorial
import time
from tqdm import tqdm # customisable progressbar decorator for iterators
from numba import jit

# define usefull one body operators 
sx=sigmax()
sy=sigmay()
sz=sigmaz()
si=qeye(2)
s0=(si-sz)/2

def can_bas(N,i):
    e = np.zeros(N)
    e[i] = 1.0
    return e

def U_parity(dim, U, basis):
    dim_par = basis.shape[1]
#    print(basis.shape)
    U_sub = np.zeros((dim_par,dim_par), dtype=np.complex_)
    U = U.data.toarray()
    for row in range(dim_par):
        for column in range(dim_par):
#            print(basis[:,row].conj().T.shape)
#            print(basis[:,column].shape)
#            print(U.shape)
            U_sub[row,column] = basis[:,row].conj().T@U@basis[:,column]
    return U_sub

#%% separate symmetries (2nd option)
N = 2    

dim = 2**N
e_basis = [Qobj(can_bas(dim,i)) for i in range(dim)]
par_basis_ones = np.zeros((dim,dim), dtype=np.complex_)
for i in range(dim):
    e_basis[i].dims = [[2 for i in range(N)], [1]]
    par_basis_ones[i] = (1/2*(e_basis[i] + e_basis[i].permute(np.arange(0,N)[::-1]))).data.toarray()[:,0]
    norma = np.linalg.norm(par_basis_ones[i])
    if norma != 0:
        par_basis_ones[i] = par_basis_ones[i]/norma
    par = par_basis_ones[i].T@par_basis_ones[i]
#    print(par)
par_basis_ones = np.unique(par_basis_ones, axis=1)
print(par_basis_ones[:,::-1])

dim_par = par_basis_ones.shape[1]

n = dim_par

ZZ = sum([tensor([qeye(2) if j!=k and j!=(k+1) else sz for j in range(n)]) for k in range(n-1)],tensor([qeye(2) if j!=(n-1) and j!=0 else sx for j in range(n)]))
Z=sum([tensor([qeye(2) if j!=k else sz for j in range(n)]) for k in range(n)])
X=sum([tensor([qeye(2) if j!=k else sx for j in range(n)]) for k in range(n)])

J = 1
B = 1

H = J*0.5*ZZ + B*(X+Z)


