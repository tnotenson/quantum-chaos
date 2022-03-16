#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:49:45 2022

@author: tomasnotenson
"""

import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from math import factorial
from time import time
from tqdm import tqdm # customisable progressbar decorator for iterators
from numba import jit
# importing "cmath" for complex number operations
from cmath import phase
plt.rcParams.update({
"text.usetex": True,
"font.family": "sans-serif",
"font.sans-serif": ["Helvetica"], "font.size": 30})

# define usefull one body operators 
sx=sigmax()
sy=sigmay()
sz=sigmaz()
si=qeye(2)

def TFIM(N, hx, hz, J):

    # construct the spin (super)operators for each site
    s_ops = [sigmax(), sigmay(), sigmaz()]
    
    sx_list = []
    sy_list = []
    sz_list = []
    
    s_lists = [sx_list, sy_list, sz_list]
    
    for n in range(N):
        op_list = [si for m in range(N)]

        for s_op, s_list in zip(s_ops, s_lists):
            op_list[n] = s_op
            s_list.append(tensor(op_list))

    # construct the hamiltonian
    H = 0

    # energy splitting terms      
    for n in range(N):
        H += hx[n] * sx_list[n]
        H += hz[n] * sz_list[n]
    # interaction terms
    #PBC
    for n in range(N):
        H += J[n] * sz_list[n] * sz_list[(n+1)%N]
    #OBC
    # for n in range(N-1):
    #     H += J[n] * sz_list[n] * sz_list[n+1]
    
    return H

def fH0(N, hx, hz, sx_list, sz_list):
    
    # construct the hamiltonian
    H = 0

    # energy splitting terms
    for n in range(N):
        H += hz[n] * sz_list[n]
        
    for n in range(N):
        H += hx[n] * sx_list[n]
   
    return H

def fH1(N, J, sz_list):
            
    # construct the hamiltonian
    H = 0
    
    # interaction terms
    #PBC
    for n in range(N):
        print('NN',n,n+1)
        H += J[n] * sz_list[n] * sz_list[(n+1)%N]
    #OBC
    # for n in range(N-1):
    #     H += J[n] * sz_list[n] * sz_list[n+1]
    return H

# Probar "propagator" de qutip
def U_from_H_numpy(H):
    e, evec = H.eigenstates()
    
    C = np.column_stack([vec.data.toarray() for vec in evec])
    Cinv = np.matrix(C).H
    
    U_diag = np.diag(np.exp(-1j*e))
    
    U = C@U_diag@Cinv
    
    return U

def fU(N, J, hx, hz):
    
    # construct the spin (super)operators for each site
    s_ops = [sigmax(), sigmay(), sigmaz()]
    
    sx_list = []
    sy_list = []
    sz_list = []
    
    s_lists = [sx_list, sy_list, sz_list]

    for n in range(N):
        op_list = [si for m in range(N)]

        for s_op, s_list in zip(s_ops, s_lists):
            op_list[n] = s_op
            s_list.append(tensor(op_list))
    
    # # define the hamiltonians
    H0 = fH0(N, hx, hz, sx_list, sz_list)
    H1 = fH1(N, J, sz_list)
    
    U0 = U_from_H_numpy(H0)
    U1 = U_from_H_numpy(H1)
    
    U = U1@U0
    
    return Qobj(U)

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

# construct tensor product of sz operators of each site
def Sj(N, j='z'):
        s_list = []
        for n in range(N):
            op_list = []
            for m in range(N):
                op_list.append(si)
            if j == 'z':
                op_list[n] = sz
            elif j == 'x':
                op_list[n] = sx
            elif j == 'y':
                op_list[n] = sy
            s_list.append(tensor(op_list))
        return sum(s_list)

def sigmai_j(N,i,j='z'):

    op_list = [si for m in range(N)]

    if j == 'z':
        op_list[i] = sz
    elif j == 'x':
        op_list[i] = sx
    elif j == 'y':
        op_list[i] = sy
    return tensor(op_list)

@jit(nopython=True, parallel=True, fastmath = True)
def Evolucion_numpy(B_t, U, Udag):
    res = Udag@B_t@U#U@B_t@Udag#
    return res

@jit(nopython=True, parallel=True, fastmath = True)
def twopC_numpy_Tinf(A, B_t):
    C = np.trace(B_t@A)
    return C

def C2p_time(time_lim, A, B, U, Udag, Cs):
    # compute OTOC, O1 and O2 for each time
    for i in tqdm(range(time_lim), desc='Evolution loop'):
        
        if i==0:
            B_t = B
        else:
            # Evolution
            B_t = Evolucion_numpy(B_t, U, Udag)
            
        # dim = A.shape[0]
        
        # compute 2-point correlator with qutip
        C_t = twopC_numpy_Tinf(A, B_t)# - A_average(A, dim)
        # C_t = C_t/dim/N
        
        # print(C_t)
        Cs[i] = C_t#np.abs(C_t)
    return Cs

def Evolution2p_H_KI_Tinf(H, time_lim, N, A, B):
    
    start_time = time() 
    
    # define arrays for data storage
    Cs = np.zeros((time_lim), dtype=np.complex_)#, dtype=np.complex_)#[]
        
    # define time evolution operator
    U = (-1j*H).expm().data.toarray()
    U = np.matrix(U, dtype=np.complex_)
    Udag = U.H
    # print(U)
    
    Cs = C2p_time(time_lim, A, B, U, Udag, Cs)
        
    print(f"\nTOTAL --- {time() - start_time} seconds ---" )
    flag = '2p_H_KI_with_Tinf_state'
    return [Cs, flag]

def Evolution2p_U_KI_Tinf(U, time_lim, A, B):
    
    start_time = time() 
    
    # define arrays for data storage
    Cs = np.zeros((time_lim), dtype=np.complex_)#[]
    
    # define dagger floquet operator
    Udag = U.H

    Cs = C2p_time(time_lim, A, B, U, Udag, Cs)
    
    print(f"\nTOTAL --- {time() - start_time} seconds ---" )
    flag = '2p_U_KI_with_Tinf_state'
    return [Cs, flag]

def TFIM_2pC_chaos_parameter(N, J, x, z, time_lim, evolution='U'):
        
    # define parameters of Heisenberg chain with inclined field 
    
    hx = x*np.ones(N)#np.sin(theta)*B*np.ones(N)
    hz = z*np.ones(N)#np.cos(theta)*B*np.ones(N)
    Jz = J*np.ones(N)
    
    
    start_time = time()
    # let's try it
    if evolution=='H':
        # ######## H evolution ##########
        H = TFIM(N, hx, hz, Jz)
    elif evolution=='U':
        ######## U evolution ##########
        U = fU(N, Jz, hx, hz)
        # U = np.matrix(U.data.toarray(), dtype=np.complex_)
        
    A = Sj(N, j='x')#/N
    B = A
    # A = A.data.toarray()
    # i = int(N/2)
    # A = sigmai_j(N,i,j='x')
    # j = i#+1
    # B = sigmai_j(N,j,j='z')
    
    # op = 'X'
    
    # operators = '_A'+op+'_B'+op
    
    print(f"\n Create Floquet operator --- {time() - start_time} seconds ---" )
    
    # separate symmetries
    sym_time = time()
    
    P = parity(N)
    ep, epvec = P.eigenstates()
    
    print(ep)
    
    n_mone, n_one = np.unique(ep, return_counts = True)[1]
    
    print('tamaños de subespacios de par.', n_mone,'+',n_one,'=', n_mone+n_one)
    C = np.column_stack([vec.data.toarray() for vec in epvec])
    Cinv = np.linalg.inv(C)
    
    if evolution=='H':
        ######## H evolution ##########
        
        H_par = H.transform(Cinv)
        # print(H_par)
        
    elif evolution=='U':
        # ######## U evolution ##########
        U_par = U.transform(Cinv)
        # # hinton(U_par)
  
    
    A_par = A.transform(Cinv)
    B_par = B.transform(Cinv)
    # hinton(A_par)
    
    
        
    if evolution=='H':
       
        # # ######## H evolution ##########
        H_mone = H_par.extract_states(np.arange(n_mone))
        H_one = H_par.extract_states(np.arange(n_mone,n_one+n_mone))
        # print(H_sub)
    
        
    elif evolution=='U':
        # ######## U evolution ##########
        U_mone = U_par.extract_states(np.arange(n_mone)).data.toarray()
        U_one = U_par.extract_states(np.arange(n_mone, n_mone+n_one)).data.toarray()
        U_mone = np.matrix(U_mone, dtype=np.complex_)
        U_one = np.matrix(U_one, dtype=np.complex_)
        
 
    A_mone = A_par.extract_states(np.arange(n_mone)).data.toarray()
    A_one = A_par.extract_states(np.arange(n_mone, n_mone+n_one)).data.toarray()
    
    B_mone = B_par.extract_states(np.arange(n_mone)).data.toarray()
    B_one = B_par.extract_states(np.arange(n_mone, n_mone+n_one)).data.toarray()
    # A_sub = np.matrix(A_sub, dtype=np.complex_)
    print(f"\n Separate parity eigenstates --- {time() - sym_time} seconds ---" )
    
    # r_normed_H = diagH_r(H_sub)
    # r_normed_U = diagU_r(U_sub)
    
    evol_time = time()
    #
    
    if evolution=='H':
                
        Cs_mone, _ = Evolution2p_H_KI_Tinf(H_mone, time_lim, A_mone, B_mone)
        Cs_one, _ = Evolution2p_H_KI_Tinf(H_one, time_lim, A_one, B_one)
        
    elif evolution=='U':
            
        Cs_mone, _ = Evolution2p_U_KI_Tinf(U_mone, time_lim, A_mone, B_mone)
        Cs_one, _ = Evolution2p_U_KI_Tinf(U_one, time_lim, A_one, B_one)
        # np.savez('2pC_'+flag+f'_time_lim{time_lim}_J{J:.2f}_hx{x:.2f}_hz{z:.2f}_basis_size{N}'+operators+'.npz', Cs=Cs)
    
    print(f"\n Evolution 2pC --- {time() - evol_time} seconds ---" )
    print(f"\n TOTAL --- {time() - start_time} seconds ---" )
    return [Cs_mone, Cs_one]#, r_normed_U# [Cs] #

#%% Calcular los correladores para un valor de theta y de B
# define parameters of Heisenberg chain with inclined field 
N = 3
J = 1

time_lim = 51
times = np.arange(0,time_lim)
points = 50

# tomo los valores de Prosen
hx = 1.4
hz = 1.4

alpha = 2*J
beta = 2*hx

DL = ((max(np.abs(np.cos(alpha)),np.abs(np.cos(beta)))-(np.cos(beta))**2))/(np.sin(beta)**2)
print(DL)

B = np.sqrt(hx**2 + hz**2)
if hz == 0:
    theta = np.pi/2
else:
    theta = np.arctan(hx/hz)
print('B=',B,'\ntheta=',theta)
#%%
Cs_mone, Cs_one = TFIM_2pC_chaos_parameter(N, J, hx, hz, time_lim)
# Cs = TFIM_2pC_chaos_parameter(N, B, J, theta, time_lim)[0]

Cs = (Cs_mone + Cs_one)/2**N/N

# Cs_numpy = Cs
# flag = 'Evol_alreves_2p_KI_with_Tinf_state_'
# # opi = str(int(N/2))
# # opj = str(int(N/2))
# opA = 'X'#+opi
# opB = opA#'Z_'+opj
# paridad = 'par'
# operators = '_A'+opA+'_B'+opB
# BC = 'PBC'
# flag = '2p_KI_with_Tinf_state_'
# np.savez('2pC_'+flag+f'_time_lim{time_lim}_J{J:.2f}_hx{hx:.2f}_hz{hz:.2f}_basis_size{N}'+operators+'.npz', Cs=Cs)
#%% Fidelidad diagonalización numpy qutip
# from scipy.linalg import eig

# hx = hx*np.ones(N)#np.sin(theta)*B*np.ones(N)
# hz = hz*np.ones(N)#np.cos(theta)*B*np.ones(N)
# Jz = J*np.ones(N)

# H = TFIM(N, hx, hz, Jz)

# e, evec = H.eigenstates()
# en, envec = eig(H.full())
# #%%
# indx = np.real(en).argsort()#[::-1]
# en_sort = en[indx]
# envec_sort = np.real(envec[:,indx])

# for i in range(len(e)):
    
#     print('i=',i)    
    
#     Qenvec = Qobj(envec_sort[:,i])
#     Qenvec.dims = evec[i].dims
    
#     # print(H.matrix_element(evec[i].dag(), evec[i]), H.matrix_element(Qenvec.dag(),Qenvec))
#     # print(e[i], en_sort[i])
    
    
#     norm = evec[i].norm()
#     # nnorm = Qenvec.norm()
#     # print('Normas antes',norm,nnorm)
    
#     nnorm = Qenvec.norm()
#     Qenvec = Qenvec/nnorm
#     print('Normas después',norm,nnorm)
    
#     F = fidelity(evec[i], Qenvec)
#     print('Fidelidad',F)
# #%%

# H_diag = np.diag(en)
# # print(H_diag)

# C = np.column_stack([vec for vec in envec])
# Cinv = np.matrix(C).H

# H = C@H_diag@Cinv

# hinton(Qobj(H))