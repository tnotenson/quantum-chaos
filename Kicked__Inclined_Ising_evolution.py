#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:54:17 2021

@author: usuario
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

def fH0(N, hx, hz, theta):

    # construct the hamiltonian
    H = 0

    # energy splitting terms
    for n in range(N):
        H += hz[n] * np.cos(theta) * sz_list[n]
        
    for n in range(N):
        H += hx[n] * np.sin(theta) * sx_list[n]
   
    return H

def fH1(N, J):
    
    # construct the hamiltonian
    H = 0
    
    # interaction terms
    for n in range(N):
        H += J[n] * sz_list[n] * sz_list[(n+1)%N]
    
    return H

def fU(N, J, hx, hz, theta):
    
    # define the hamiltonians
    H0 = fH0(N, hx, hz, theta)
    H1 = fH1(N, J)
    
    # define the floquet operator
    U = (-1j*H1).expm()*(-1j*H0).expm()
    
    return U

def Evolution4p_KII_Tinf(time_lim, N, J, hx, hz, theta, A, B):
    
    start_time = time.time() 
    
    # define arrays for data storage
    OTOCs = np.zeros((time_lim), dtype=np.complex_)#[]
    O1s = np.zeros((time_lim), dtype=np.complex_)#[]
    O2s = np.zeros((time_lim), dtype=np.complex_)#[]
        
    # define floquet operator
    U = fU(N, J, hx, hz, theta)
    
    # compute OTOC, O1 and O2 for each time
    for i in tqdm(range(time_lim), desc='Evolution loop'):
        
        if i==0:
            B_t = B
        else:
            # Evolution
             B_t = B_t.transform(U.dag())# U*B_t*U.dag()
        
        # compute O1 and O2 
        O1 = (B_t*A*B_t*A).tr()
        O2 = (B_t**2*A**2).tr()
        
        # compute OTOC
        dim = 2**N
        C_t = -2*( O1 - O2 )/dim
        
        # store data
        OTOCs[i] = np.abs(C_t)
        O1s[i] = np.abs(O1)
        O2s[i] = np.abs(O2)

        
    print(f"\nTOTAL --- {time.time() - start_time} seconds ---" )
    flag = '4p_KII_with_Tinf_state'
    return [OTOCs, O1s, O2s, flag]

def Evolution2p_KII_Tinf(time_lim, N, J, hx, hz, theta, A, B):
    
    start_time = time.time() 
    
    # define arrays for data storage
    Cs = np.zeros((time_lim), dtype=np.complex_)#[]
        
    # define floquet operator
    U = fU(N, J, hx, hz, theta)
    
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
    flag = '2p_KII_with_Tinf_state'
    return [Cs, flag]
#%%
N = 6
J = 1*np.ones(N)
hx = 1*np.ones(N)
hz = 1*np.ones(N)

j = np.mean(J)
x = np.mean(hx)
z = np.mean(hz)

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

time_lim = 100

thetas = np.pi/2*np.arange(0,1.2,0.2)
times = np.arange(0,time_lim)

# define colormap
colors = plt.cm.jet(np.linspace(0,1,len(thetas)))

plt.figure()
for i,theta in enumerate(thetas):
    OTOCs, O1s, O2s, flag = Evolution4p_KII_Tinf(time_lim, N, J, hx, hz, theta, A, B)
    np.savez(flag+f'_J{j}_hz{x}_hz{z}_basis_size{N}_AZ_BZ.npz', OTOCs=OTOCs, O1s=O1s, O2s=O2s)
#%% level statistics
# importing "cmath" for complex number operations
import cmath

# Calculates r parameter in the 10% center of the energy "ener" spectrum. If plotadjusted = True, returns the magnitude adjusted to Poisson = 0 or WD = 1

def r_chaometer(ener,normed=False):
    ra = np.zeros(len(ener)-2)
    #center = int(0.1*len(ener))
    #delter = int(0.05*len(ener))
    for ti in range(len(ener)-2):
        ra[ti] = (ener[ti+2]-ener[ti+1])/(ener[ti+1]-ener[ti])
        ra[ti] = min(ra[ti],1.0/ra[ti])
    ra = np.mean(ra)
    if normed == True:
        ra = (ra-0.3863) / (0.5307-0.3863)
    return ra

theta = np.pi/2

# define floquet operator
U = fU(N, J, hx, hz, theta)

r = 2
r_lim = 4**r
U = U[:r_lim,:r_lim]

evalues, evectors = np.linalg.eig(U)#.eigenstates()

ephases = np.zeros((len(evalues)))
for i in range(len(evalues)):
    ephases[i] = cmath.phase(evalues[i])
#%%
# extract real part using numpy array
x = evalues.real
# extract imaginary part using numpy array
y = evalues.imag
  
# plot the complex numbers
plt.plot(x, y, 'g*')
plt.ylabel('Imaginary')
plt.xlabel('Real')
plt.show()
#%% 
sorted_phases = np.sort(ephases)
#print(sorted_phases[:10])
level_s = np.diff(sorted_phases)
#print(level_s[:10])
plt.hist(level_s, normed=True)
plt.xlabel('eigenphases')
plt.savefig('espaciado_Emi.png', dpi=300)
