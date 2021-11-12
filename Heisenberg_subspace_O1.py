#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 16:15:16 2021

@author: usuario
"""
import matplotlib.pyplot as plt
from qutip import *
import numpy as np
import time
from tqdm import tqdm # customisable progressbar decorator for iterators

# define usefull one body operators 
sx=sigmax()
sy=sigmay()
sz=sigmaz()
s0=(qeye(2)-sz)/2
si=qeye(2)

# define hamiltonians 

def Heisenberg_inclined_field(N, hx, hz, theta, Jx, Jy, Jz):

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
        H += hz[n] * np.cos(theta) * sz_list[n]
        
    for n in range(N):
        H += hx[n] * np.sin(theta) * sx_list[n]

    # interaction terms
    for n in range(N-1):
        H += - Jx[n] * sx_list[n] * sx_list[n+1]
        H += - Jy[n] * sy_list[n] * sy_list[n+1]
        H += - Jz[n] * sz_list[n] * sz_list[n+1]
    
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

# reduce to a subspace with Sz = M
def chbasis_bloque(A, B, H, ind_1):
    # define number of qubits
    N = len(H.dims[0])
    
    s=sum([Sz(N)[x] for x in range(N)])
    
    # reduce H and other operators to Sz = M subspace
    A_red = A.extract_states(ind_1)
    B_red = B.extract_states(ind_1)
    H_red = H.extract_states(ind_1)
    s_red = s.extract_states(ind_1)
    
    # diagonalize the reduced H
    time_0 = time.time()
    ene_red,aut_red=H_red.eigenstates()
    time_1 = time.time()
    print(f'H_sub diagonalization :{time_1 - time_0} seg')
    
    # check
    for i in range(len(ind_1)):
        e = ene_red[i]
        print(f'\nEnergy of {i} eigenstate: {e}')
        mag = int(s_red.matrix_element(aut_red[i].dag(), aut_red[i]).real)
        print(f'Magnetization of {i} eigenstate: {mag}')
        
    return H_red, A_red, B_red, s_red


# define parameters of Heisenberg chain with inclined field 
N = 8
B = 2 
J = 2
hx = 0*np.ones(N)
hz = B**np.ones(N) #np.random.uniform(-B,B) #random field in z 
theta = 0 # angle of the field
Jx = J*np.ones(N)
Jy = J*np.ones(N)
Jz = J*np.ones(N)

# let's try it
H = Heisenberg_inclined_field(N, hx, hz, theta, Jx, Jy, Jz)
#e, ev = H.eigenstates()

# define number of excitations
exc = int(N/4)

# search index of computational basis with Sz = M
time_0 = time.time()
ind_1=sz_subspace_diag(N,exc)
time_1 = time.time()
print(f'sz_subspace_diag :{time_1 - time_0} seg')
time_2 = time.time()
print(f'\nchbasis_bloque :{time_2 - time_1} seg')


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

A = sum(Sz(N))
B = sum(Sz(N))

H_red, A_red, B_red, s_red = chbasis_bloque(A, B, H, ind_1)

#%%
def Evolution2p_KI_Tinf(H, time_lim, N, A, B):
    
    start_time = time.time() 
    
    # define arrays for data storage
    Cs = np.zeros((time_lim))#, dtype=np.complex_)#[]
        
    # define time evolution operator
    U = H.expm()
    
    # print(U)
    
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

        print(C_t)
        # store data
        Cs[i] = np.abs(C_t)

        
    print(f"\nTOTAL --- {time.time() - start_time} seconds ---" )
    flag = '2p_KI_with_Tinf_state'
    return [Cs, flag]


time_lim = 30

Cs, flag = Evolution2p_KI_Tinf(H_red, time_lim, N, A_red, B_red)

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


yaxis = np.log10(Cs)
plt.plot(times, yaxis,':^r' , label=r'$T = \infty$');

# # plot vertical lines in the corresponding Ehrenfest's time for each K
# # plt.vlines(tE, np.min(np.log10(OTOC_Ks[0]*factor)+b), np.max(np.log10(OTOC_Ks[0]*factor)+b), lw=0.5, ls='dashed', alpha=0.8, color=colors[i])
    
plt.xlabel('Time');
plt.ylabel(r'$log \left(C(t) \right)$');
# plt.xlim((0,10))
# plt.ylim((0,1e5))
plt.grid();
plt.legend(loc='best');
plt.show()
# plt.savefig('2pC_'+flag+f'_J{j}_hz{x}_hz{z}_basis_size{N}_AZ_BZ.png', dpi=300)#_Gauss_sigma{sigma}