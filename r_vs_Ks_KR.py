#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 21:09:21 2022

@author: tomasnotenson
"""

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from qutip import *
from time import time 
from tqdm import tqdm # customisable progressbar decorator for iterators
from cmath import phase
from scipy.stats import skew
import time 
plt.rcParams.update({
"text.usetex": True,
"font.family": "sans-serif",
"font.sans-serif": ["Helvetica"], "font.size": 24})

def normalize(array):
    return (array - min(array))/(max(array)-min(array))

@jit(nopython=True, parallel=True, fastmath = True)#, cache=True)#(cache=True)#(nopython=True)
def FF(N, x1 = 0, p1 = 0):
    FF = np.zeros((N,N),dtype=np.complex_)
    for i in range(N):
        for j in range(N):
            FF[i,j]=np.exp(-1j*2*np.pi*(i + x1)*(j + p1)/N)*np.sqrt(1/N)
    return FF

@jit#(nopython=True, parallel=True, fastmath = True)#, cache=True)#(nopython=True)
def UU(K, N, op = 'P', x1 = 0, p1 = 0):
    UU = np.zeros((N,N),dtype=np.complex_)
    MM = np.zeros((N,N),dtype=np.complex_)
    F = FF(N)
    Keff = K/(4*np.pi**2)
    
    if op == 'X':
        
        for i in range(N):
                for j in range(N):
                    UU[i,j]=F[i,j]*np.exp(-1j*2*np.pi*N*Keff*np.cos(2*np.pi*(i + x1)/N))#/N
                    MM[i,j]=np.conjugate(F[j,i])*np.exp(-1j*np.pi*(i + p1)**2/N)
    elif op == 'P'  :
        for i in range(N):
                for j in range(N):
                    UU[i,j]=np.exp(-1j*2*np.pi*N*Keff*np.cos(2*np.pi*(i + x1)/N))*np.conjugate(F[j,i])
                    MM[i,j]=np.exp(-1j*np.pi*(i + p1)**2/N)*F[i,j]
    
    U = MM@UU
    U = np.matrix(U)
    return U
@jit
# Calcultes r parameter in the 10% center of the energy "ener" spectrum. If plotadjusted = True, returns the magnitude adjusted to Poisson = 0 or WD = 1
def r_chaometer(ener,plotadjusted):
    ra = np.zeros(len(ener)-2)
    #center = int(0.1*len(ener))
    #delter = int(0.05*len(ener))
    for ti in range(len(ener)-2):
        if (ener[ti+1]-ener[ti])!=0:
            ra[ti] = (ener[ti+2]-ener[ti+1])/(ener[ti+1]-ener[ti])
        else:
            ra[ti] = (ener[ti+2]-ener[ti+1])/np.finfo(float).eps
        if ra[ti]!=0:
            ra[ti] = min(ra[ti],1.0/ra[ti])
    ra = np.mean(ra)
    if plotadjusted == True:
        ra = (ra -0.3863) / (-0.3863+0.5307)
    return ra

def diagU_r(U_sub):
    ener = np.linalg.eigvals(U_sub)
    # ener = np.sort(ener)
    fases = [phase(en) for en in ener]
    fases = np.sort(fases)
    # histo_level_spacing(fases)
    r_normed = r_chaometer(fases, plotadjusted=True) 
    return r_normed

def fr(Ks,N, op='P'):
    
    start_time = time.time() 
    
    r_Ks = np.zeros((len(Ks)))
    for j, K in tqdm(enumerate(Ks), desc='Primary loop'):
        
        # Distinct evolution for each operator X or P
        U = UU(K, N, op)
        r_normed = diagU_r(U)
        r_Ks[j] = r_normed
    print(f"\nTOTAL --- {time.time() - start_time} seconds ---" )
    return r_Ks

def normalize(x):
    return (x-min(x))/(max(x)-min(x))
#%% defino parámetros
N = 3000
Kpaso = .25
Ks = np.arange(0,20,Kpaso)
#%% simulo
r_Ks = fr(Ks,N)

np.savez(f'r_vs_Ks_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_basis_size{N}.npz',Ks=Ks,r_Ks=r_Ks)
#%% cargo 
# Kpaso1 = .5
# K1 = np.arange(2,20,Kpaso1)
# archives = np.load(f'r_vs_Ks_Kmin{min(K1)}_Kmax{max(K1):.1f}_Kpaso{Kpaso1}_basis_size{N}.npz')
# r_1 = archives['r_Ks']
# Kpaso2 = .05
# K2 = np.arange(0,2,Kpaso2)
# archives = np.load(f'r_vs_Ks_Kmin{min(K2)}_Kmax{max(K2):.2f}_Kpaso{Kpaso2}_basis_size{N}.npz')
# r_2 = archives['r_Ks']

# Ks = np.concatenate((K2,K1))
# r_Ks = np.concatenate((r_2,r_1))

#%% grafico



x = Ks
y = normalize(r_Ks*(-0.3863+0.5307) +0.3863)#normalize

plt.figure(figsize=(16,8))
plt.title(f'N={N}')
plt.plot(x, y, '.-', ms=10, lw=1.5,  color='blue')
plt.ylabel(r'$r$')
plt.xlabel(r'$K$')
# plt.xticks(times)
# plt.yscale('log')
plt.ylim(-0.1,1.1)
# plt.xlim(-0.01,15)
plt.grid()
# plt.legend(loc = 'best')
plt.show()
# file = flag+f'_K{Ks[k]:.1f}_basis_size{N}_time_lim{time_lim}.png'
# plt.savefig(file, dpi=80)
