#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 14:14:12 2021

@author: tomasnotenson
"""
import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from math import factorial
import time
from tqdm import tqdm # customisable progressbar decorator for iterators

# define usefull one body operators 
sx=sigmax()
sy=sigmay()
sz=sigmaz()
s0=(qeye(2)-sz)/2
si=qeye(2)

# construct the H
def XXZ(N, J, mu, lamb, h = 0, BC ='open', impurity=0):

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sx
        sx_list.append(tensor(op_list))

        op_list[n] = sy
        sy_list.append(tensor(op_list))

        op_list[n] = sz
        sz_list.append(tensor(op_list))

    # construct the hamiltonian
    H = 0

    # energy splitting terms
    for n in range(N):
        H += h * sz_list[n]
    H += impurity * sz_list[-1]

    # interaction terms
    for n in range(N-1):
        H += J * sx_list[n] * sx_list[n+1]
        H += J * sy_list[n] * sy_list[n+1]
        H += mu * J * sz_list[n] * sz_list[n+1]
        
    for n in range(N-2):
        H += lamb * J * sx_list[n] * sx_list[n+2]
        H += lamb * J * sy_list[n] * sy_list[n+2]
        H += lamb * mu * J * sz_list[n] * sz_list[n+2]
        
    if BC == 'periodic':
        H +=  J * sx_list[0] * sx_list[-1]
        H +=  J * sy_list[0] * sy_list[-1]
        H +=  mu * J * sz_list[0] * sz_list[-1]
        
        H +=  lamb * J * sx_list[0] * sx_list[-1]
        H +=  lamb * J * sy_list[0] * sy_list[-1]
        H +=  lamb * mu * J * sz_list[0] * sz_list[-1]
    
    return H

# construct tensor product of sz operators of each site
def Sz(N):
    sz_list = []
    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)
        op_list[n] = 0.5*sz
        sz_list.append(tensor(op_list))
    return sz_list

# construct permute operator
def pij(l,i,j):
    geye=[si for k in range(l)]
    
    H=0*tensor(geye)
    g=geye.copy(); g[i]=sx;g[j]=sx; H+= tensor(g)
    g=geye.copy(); g[i]=sy;g[j]=sy; H+= tensor(g)
    g=geye.copy(); g[i]=sz;g[j]=sz; H+= tensor(g)
    
    H+=tensor(geye); return H/2

# construct parity operator
def parity(l):
    geye=[si for k in range(l)]
    
    P=tensor(geye)
    for m in range(l//2):
        P=pij(l,m,l-m-1)*P
    return P

# search index of computational basis with Sz = M
def sz_subspace_diag(N, exc):
    s=sum([Sz(N)[x] for x in range(N)])
    s_diag = s.diag()
    index_M=[]
    M= 1/2 * exc - 1/2 * (N - exc)
    for x in range(len(s_diag)):
        if s_diag[x]==M:
            index_M.append(x)    
    ind_1=[];
    for i in range(2**N):
        if i in index_M:
            ind_1.append(i)
    return ind_1 

# calculate r chaos indicator
def r_chaometer(ener,normed=False):
    ra = np.zeros(len(ener)-2)
    #center = int(0.1*len(ener))
    #delter = int(0.05*len(ener))
    for ti in range(len(ener)-2):
        ra[ti] = (ener[ti+2]-ener[ti+1])/(ener[ti+1]-ener[ti])
        print(f'ra_0: {ra[ti]}')
        ra[ti] = min(ra[ti],1.0/ra[ti])
        print(f'ra_1: {ra[ti]}')
    ra = np.mean(ra)
    if normed == True:
        ra = (ra-0.3863) / (0.5307-0.3863)
    return ra

# reduce to a subspace with Sz = M and parity = 1
def chbasis_bloque(H, ind_1, pari=1):
    # define number of qubits
    N = len(H.dims[0])
    # define parity operator
    P = parity(N)
    
    # reduce H and parity to Sz = M subspace
    
    P_red = P.extract_states(ind_1)
    H_red = H.extract_states(ind_1)
    
    # diagonalize the reduced H
    time_0 = time.time()
    ene_red,aut_red=H_red.eigenstates()
    time_1 = time.time()
    print(f'H_sub diagonalization :{time_1 - time_0} seg')
    
    # search index with parity 1 in the energy basis
    ind_par = []
    for i in range(len(ind_1)):
        e = H_red.matrix_element(aut_red[i].dag(), aut_red[i]).real
        print(f'\nEnergy of {i} eigenstate: {e}')
        n = len(ind_1)
        par = int(P_red.matrix_element(aut_red[i].dag(), aut_red[i]).real)
        print(f'Parity of {i} eigenstate: {par}')
        if par == pari:
            ind_par.append(i)
    
    # construct the base change matrix
    C = [aut_red[i].data.todense() for i in range(len(aut_red))]
    C = np.array(C, ndmin=2)[:,:,0]
    # change from computational to energy basis
    H_sub = H_red.transform(C)
    P_sub = P_red.transform(C)
    
    # reduce to a subspace with Sz = M and parity = 1
    H_ssub = H_sub.extract_states(ind_par)
    P_ssub = P_sub.extract_states(ind_par)
               
    # check
    
    print('\n',ind_par) # energy basis index with parity 1
    print('\n',H_ssub) # check that H is diagonal
    print('\n',P_ssub) # # check that parity is equal (prop) to identity
    
    return H_ssub, ind_par


#%%
# define H's parameters
N = 12
J = 1
mu = 0.5
lamb = 0.5

# define number of excitations
exc = int(N/4)
H = XXZ(N, J, mu, lamb)

# search index of computational basis with Sz = M
time_0 = time.time()
ind_1=sz_subspace_diag(N,exc)
time_1 = time.time()
print(f'sz_subspace_diag :{time_1 - time_0} seg')

# reduce to a subspace with Sz = M and parity = 1
H_sub, ind_par = chbasis_bloque(H, ind_1)
time_2 = time.time()
print(f'\nchbasis_bloque :{time_2 - time_1} seg')
# print(f'\nIs H_sub diagonal?: {not bool(np.count_nonzero(H_sub - np.diag(np.diagonal(H_sub))))}')
# print(f'H_sub diagonal: {H_sub.diag()}')
#%%
sx_list = []
sy_list = []
sz_list = []

for n in range(N):
    op_list = []
    for m in range(N):
        op_list.append(si)

    op_list[n] = sx
    sx_list.append(tensor(op_list))

    op_list[n] = sy
    sy_list.append(tensor(op_list))

    op_list[n] = sz
    sz_list.append(tensor(op_list))
    

def Evolution2p_KI_Tinf(H, time_lim, N, J, hx, hz, A, B):
    
    start_time = time.time() 
    
    # define arrays for data storage
    Cs = np.zeros((time_lim), dtype=np.complex_)#[]
        
    # define time evolution operator
    U = H.expm()
    
    # compute OTOC, O1 and O2 for each time
    for i in tqdm(range(time_lim), desc='Evolution loop'):
        
        if i==0:
            B_t = B
        else:
            # Evolution
             B_t = B_t.transform(U.dag())# U*B_t*U.dag()
        
        # compute 2-point correlator
        dim = 2**N
        C_t = (B_t*A).tr()
        C_t = C_t/dim

        
        # store data
        Cs[i] = np.abs(C_t)

        
    print(f"\nTOTAL --- {time.time() - start_time} seconds ---" )
    flag = '2p_KI_with_Tinf_state'
    return [Cs, flag]



    
    
