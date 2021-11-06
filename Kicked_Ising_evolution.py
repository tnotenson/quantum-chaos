#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 11:53:09 2021

@author: tomasnotenson
"""
from qutip import *
import numpy as np
import time
from tqdm import tqdm
from matplotlib import pyplot as plt

si = qeye(2)
sx = sigmax()
sy = sigmay()
sz = sigmaz()

def fH0(N, hx, hz):

    # construct the hamiltonian
    H = 0

    # energy splitting terms
    for n in range(N):
        H += hz[n] * sz_list[n]
        
    for n in range(N):
        H += hx[n] * sx_list[n]
   
    return H

def fH1(N, J):
    
    # construct the hamiltonian
    H = 0
    
    # interaction terms
    for n in range(N):
        H += J[n] * sz_list[n] * sz_list[(n+1)%N]
    
    return H

def fU(N, J, hx, hz):
    
    # define the hamiltonians
    H0 = fH0(N, hx, hz)
    H1 = fH1(N, J)
    
    # define the floquet operator
    U = (-1j*H1).expm()*(-1j*H0).expm()
    
    return U

def Evolution_KI_Tinf(time_lim, N, J, hx, hz, A, B):
    
    start_time = time.time() 
    
    # define arrays for data storage
    OTOCs = np.zeros((time_lim))#[]
    O1s = np.zeros((time_lim))#[]
    O2s = np.zeros((time_lim))#[]
        
    # define floquet operator
    U = fU(N, J, hx, hz)
    
    # compute OTOC, O1 and O2 for each time
    for i in tqdm(range(time_lim), desc='Evolution loop'):
        
        if i==0:
            B_t = B
        else:
            # FFT for efficient evolution
            # diagonal evolution
             B_t = B_t.transform(U.dag())# U*B_t*U.dag()
        
        
        O1 = (B_t*A*B_t*A).tr()
        O2 = (B_t**2*A**2).tr()
                    
        dim = 2**N
        C_t = -2*( O1 - O2 )/dim
        
        OTOCs[i] = np.abs(C_t)#OTOC.append(np.abs(C_t.data.toarray()))
        O1s[i] = np.abs(O1)#O1s.append(np.abs(O1.data.toarray()))
        O2s[i] = np.abs(O2)#O2s.append(np.abs(O2.data.toarray()))

        
    print(f"\nTOTAL --- {time.time() - start_time} seconds ---" )
    flag = 'KI_with_Tinf_state'
    return [OTOCs, O1s, O2s, flag]
#%%
N = 8
J = 1*np.ones(N)
hx = 0.5*np.ones(N)
hz = 0.5*np.ones(N)

sx_list = []
sy_list = []
sz_list = []

for n in range(N):
    op_list = [si for m in range(N)]

    op_list[n] = sx
    sx_list.append(tensor(op_list))

    op_list[n] = sy
    sy_list.append(tensor(op_list))

    op_list[n] = sz
    sz_list.append(tensor(op_list))

A = sum(sz_list)
B = sum(sz_list)

time_lim = 30

OTOCs, O1s, O2s, flag = Evolution_KI_Tinf(time_lim, N, J, hx, hz, A, B)
#%% plot results

times = np.arange(0,time_lim)

# define colormap
# colors = plt.cm.jet(np.linspace(0,1,len(Ks)))

# create the figure
plt.figure(figsize=(12,8), dpi=100)

# all plots in the same figure
plt.title(f'OTOC N = {N} A = Z, B = Z')
    
# compute the Ehrenfest's time
# tE = np.log(N)/ly1[i]
    
# plot log10(C(t))


yaxis = np.log10(OTOCs)
plt.plot(times, yaxis,':^r' , label=r'T = \infty');

# plot vertical lines in the corresponding Ehrenfest's time for each K
# plt.vlines(tE, np.min(np.log10(OTOC_Ks[0]*factor)+b), np.max(np.log10(OTOC_Ks[0]*factor)+b), lw=0.5, ls='dashed', alpha=0.8, color=colors[i])
    
plt.xlabel('Time');
plt.ylabel(r'$log \left(C(t) \right)$');
# plt.xlim((0,10))
# plt.ylim((-12,0))
plt.grid();
plt.legend(loc='best');
plt.show()
plt.savefig('OTOC_'+flag+f'_J{np.mean(J)}_basis_size{N}_AZ_BZ.png', dpi=300)#_Gauss_sigma{sigma}

# create the figure
plt.figure(figsize=(12,8), dpi=100)

# all plots in the same figure
plt.title(f'O1 N = {N} A = Z, B = Z')
    
# compute the Ehrenfest's time
# tE = np.log(N)/ly1[i]
    
# plot log10(C(t))


yaxis = np.log10(O1s)
plt.plot(times, yaxis,':^r' , label=r'T = \infty');

# plot vertical lines in the corresponding Ehrenfest's time for each K
# plt.vlines(tE, np.min(np.log10(OTOC_Ks[0]*factor)+b), np.max(np.log10(OTOC_Ks[0]*factor)+b), lw=0.5, ls='dashed', alpha=0.8, color=colors[i])
    
plt.xlabel('Time');
plt.ylabel(r'$log \left(O_1(t) \right)$');
# plt.xlim((0,10))
# plt.ylim((-12,0))
plt.grid();
plt.legend(loc='best');
plt.show()
plt.savefig('O1_'+flag+f'_J{np.mean(J)}_basis_size{N}_AZ_BZ.png', dpi=300)#_Gauss_sigma{sigma}

# create the figure
plt.figure(figsize=(12,8), dpi=100)

# all plots in the same figure
plt.title(f'O2 N = {N} A = Z, B = Z')
    
# compute the Ehrenfest's time
# tE = np.log(N)/ly1[i]
    
# plot log10(C(t))


yaxis = np.log10(O2s)
plt.plot(times, yaxis,':^r' , label=r'T = \infty');

# plot vertical lines in the corresponding Ehrenfest's time for each K
# plt.vlines(tE, np.min(np.log10(OTOC_Ks[0]*factor)+b), np.max(np.log10(OTOC_Ks[0]*factor)+b), lw=0.5, ls='dashed', alpha=0.8, color=colors[i])
    
plt.xlabel('Time');
plt.ylabel(r'$log \left(O_2(t) \right)$');
# plt.xlim((0,10))
# plt.ylim((-12,0))
plt.grid();
plt.legend(loc='best');
plt.show()
plt.savefig('O2_'+flag+f'_J{np.mean(J)}_basis_size{N}_AZ_BZ.png', dpi=300)#_Gauss_sigma{sigma}