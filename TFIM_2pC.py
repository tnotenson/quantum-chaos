#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 14:26:35 2021

@author: tomasnotenson
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


def TFIM(N, hx, hz, J):

    # construct the spin (super)operators for each site
    s_ops = [sigmax(), sigmay(), sigmaz()]
    
    sx_list = []
    sy_list = []
    sz_list = []
    
    s_lists = [sx_list, sy_list, sz_list]
    
    for n in range(N):
        op_list = [qeye(2) for m in range(N)]

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
    for n in range(N-1):
        H += - J[n] * 0.5 * sz_list[n] * sz_list[n+1]
    
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
    for n in range(N):
        H += J[n] * 0.5 * sz_list[n] * sz_list[(n+1)%N]
    
    return H

def fU(N, J, hx, hz):
    
    # construct the spin (super)operators for each site
    s_ops = [sigmax(), sigmay(), sigmaz()]
    
    sx_list = []
    sy_list = []
    sz_list = []
    
    s_lists = [sx_list, sy_list, sz_list]

    for n in range(N):
        op_list = [qeye(2) for m in range(N)]

        for s_op, s_list in zip(s_ops, s_lists):
            op_list[n] = s_op
            s_list.append(tensor(op_list))
    
    # # define the hamiltonians
    H0 = fH0(N, hx, hz, sx_list, sz_list)
    H1 = fH1(N, J, sz_list)
#    
###    # define the floquet operator
#    e1, evec1 = H1.eigenstates()
#    e0, evec0 = H0.eigenstates()
#   
#    C1 = np.column_stack([vec.data.toarray() for vec in evec1])
#    C1inv = np.matrix(C1).H
#    
#    C0 = np.column_stack([vec.data.toarray() for vec in evec0])
#    C0inv = np.matrix(C0).H
#    
#    U1_diag = np.diag(np.exp(-1j*e1))
#    U0_diag = np.diag(np.exp(-1j*e0))
##    
###    print(C1inv.shape)
###    print(C1.shape)
###    print(U1_diag)
##    
#    U1 = C1@U1_diag@C1inv
#    U0 = C0@U0_diag@C0inv
###    
#    U = U1@U0
    
    U = (-1j*H1).expm()*(-1j*H0).expm()
    
    return U

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

def can_bas(N,i):
    e = np.zeros(N)
    e[i] = 1.0
    return e

@jit(nopython=True, parallel=True, fastmath = True)
def Evolucion_numpy(B_t, U, Udag):
    res = Udag@B_t@U
    return res

@jit(nopython=True, parallel=True, fastmath = True)
def twopC_numpy_Tinf(A, B_t):
    C = np.trace(B_t@A)
    return C

@jit(nopython=True, parallel=True, fastmath = True)
def A_average(A, dims):
    res = np.trace(A)**2/dims
    return res

@jit(nopython=True, parallel=True, fastmath = True)
def C_t_commutator_numpy_Tinf(A, B_t):
    com = B_t@A - A@B_t
    C_t = np.trace(com.H@com)/N
    return C_t
#%% define operators
# define parameters of Heisenberg chain with inclined field 
N = 8
B = 1.4
J = 1
hx = B*np.ones(N)
hz = B*np.ones(N)
Jz = J*np.ones(N)

j = J
x = np.mean(hx)
z = np.mean(hz)

start_time = time.time()
# let's try it

# H = TFIM(N, hx, hz, Jz)
U = fU(N, Jz, hx, hz)

A = Sj(N, j='x')#/N
print(f"\n Create Floquet operator --- {time.time() - start_time} seconds ---" )
#%% separate symmetries (1st option)
start_time = time.time()

P = parity(N)
ep, epvec = P.eigenstates()

n_mone, n_one = np.unique(ep, return_counts = True)[1]

C = np.column_stack([vec.data.toarray() for vec in epvec])
Cinv = np.linalg.inv(C)

# H_par = H.transform(Cinv)
#print(H_par)
U_par = U.transform(Cinv)
#print(U_par)
A_par = A.transform(Cinv)

# H_sub = H_par.extract_states(np.arange(n_mone:n_one+1))
U_sub = U_par.extract_states(np.arange(n_mone,n_one+1)).data.toarray()
A_sub = A_par.extract_states(np.arange(n_mone,n_one+1)).data.toarray()
#print(H_sub)
U_sub = np.matrix(U_sub)
#print(U_sub)
print(f"\n Separate parity eigenstates --- {time.time() - start_time} seconds ---" )
#%% separate symmetries (2nd option)
start_time = time.time()
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
#print(par_basis_ones[:,::-1])
bas = par_basis_ones
A_red = bas.conj().T@A.data.toarray()@bas
U_red = bas.conj().T@U.data.toarray()@bas
#print(U_red)
print(f"\n Separate parity eigenstates --- {time.time() - start_time} seconds ---" )
#%% 
#def U_parity(dim, U, basis):
#    dim_par = basis.shape[1]
##    print(basis.shape)
#    U_sub = np.zeros((dim_par,dim_par), dtype=np.complex_)
#    U = U.data.toarray()
#    for row in range(dim_par):
#        for column in range(dim_par):
##            print(basis[:,row].conj().T.shape)
##            print(basis[:,column].shape)
##            print(U.shape)
#            U_sub[row,column] = basis[:,row].conj().T@U@basis[:,column]
#    return U_sub
#
#U_sub = U_parity(dim, U, par_basis_ones)
#%%
def Evolution2p_H_KI_Tinf(H, time_lim, N, A, B):
    
    start_time = time.time() 
    
    # define arrays for data storage
    Cs = np.zeros((time_lim))#, dtype=np.complex_)#[]
        
    # define time evolution operator
    U = (-1j*H).expm()
    
    # print(U)
    
    # compute OTOC, O1 and O2 for each time
    for i in tqdm(range(time_lim), desc='Evolution loop'):
        
        if i==0:
            B_t = B
        else:
            # Evolution
            B_t = B_t.transform(U.dag())# U*B_t*U.dag()
        
        # compute 2-point correlator

        dim = A.shape[0]
        C_t = (B_t*A).tr() - A.tr()**2/dim
        C_t = C_t/dim


        print(C_t)
        # store data
        Cs[i] = np.abs(C_t)

        
    print(f"\nTOTAL --- {time.time() - start_time} seconds ---" )
    flag = '2p_H_KI_with_Tinf_state'
    return [Cs, flag]

def Evolution2p_U_KI_Tinf(U, time_lim, N, A, B):
    
    start_time = time.time() 
    
    # define arrays for data storage
    Cs = np.zeros((time_lim), dtype=np.complex_)#[]
    
    # define floquet operator
#    U = fU(N, J, hx, hz, theta)
    Udag = U.H


    # compute OTOC, O1 and O2 for each time
    for i in tqdm(range(time_lim), desc='Evolution loop'):
        
        if i==0:
            B_t = B
        else:

            # qutip evolution
            # B_t = B_t.transform(U.dag())
            # numpy evolution
            B_t = Evolucion_numpy(B_t, U, Udag)
            
        dim = A_sub.shape[0]
        # compute 2-point correlator with qutip
        # C_t = (B_t*A).tr() - A.tr()/dim
        # C_t = C_t/dim
        
        # compute 2-point correlator with qutip
        C_t = twopC_numpy_Tinf(A, B_t) - A_average(A, dim)
        C_t = C_t/dim
        
        print(C_t)

        
        # store data
        Cs[i] = np.abs(C_t)

        
    print(f"\nTOTAL --- {time.time() - start_time} seconds ---" )
    flag = '2p_U_KI_with_Tinf_state'
    return [Cs, flag]

time_lim = 50

#Cs, flag = Evolution2p_U_KI_Tinf(U_red, time_lim, N, A_red, A_red)
Cs, flag = Evolution2p_U_KI_Tinf(U_sub, time_lim, N, A_sub, A_sub)
# Cs, flag = Evolution2p_H_KI_Tinf(H_sub, time_lim, N, A_sub, A_sub)

#%% plot 2-point correlator results

times = np.arange(0,time_lim)

# # define colormap
# # colors = plt.cm.jet(np.linspace(0,1,len(Ks)))

# # create the figure
plt.figure(figsize=(12,8), dpi=100)

# # all plots in the same figure
plt.title(f'2-point correlator N = {N}')
    
# # compute the Ehrenfest's time
# # tE = np.log(N)/ly1[i]
    
# plot log10(C(t))


yaxis = np.log10(Cs/N)# - np.log10(Cs[0])
plt.plot(times, yaxis,':^r' , label=r'$T = \infty$');

# # plot vertical lines in the corresponding Ehrenfest's time for each K
# # plt.vlines(tE, np.min(np.log10(OTOC_Ks[0]*factor)+b), np.max(np.log10(OTOC_Ks[0]*factor)+b), lw=0.5, ls='dashed', alpha=0.8, color=colors[i])
    
plt.xlabel('Time');
plt.ylabel(r'$log \left(C(t) \right)$');
# plt.xlim((0,10))
# plt.ylim((-4,0.1))
plt.grid();
plt.legend(loc='best');
plt.show()
plt.savefig('2pC_'+flag+f'_time_lim{time_lim}_J{j:.2f}_hz{x:.2f}_hz{z:.2f}_basis_size{N}_AZ_BZ.png', dpi=300)#_Gauss_sigma{sigma}
