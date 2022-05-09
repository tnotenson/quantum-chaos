#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 14:07:47 2022

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
plt.rcParams.update({
"text.usetex": True,
"font.family": "sans-serif",
"font.sans-serif": ["Helvetica"], "font.size": 24})

def normalize(array):
    return (array - min(array))/(max(array)-min(array))
#%%
N = 2**10#11
hbar = 1/(2*np.pi*N)
# q0 = 0
# p0 = 2/10
sigma = 1#np.power(hbar,1/2)

tipo_de_base = 'gaussiana'#'integrable_wK0'#
Kpaso = .5
Ks = np.arange(0,10.1,Kpaso)#

archives = np.load(f'r_vs_K_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_N{N}.npz')
rs = archives['rs']

operators = 'AX_BP'
time_lim = 21
archives = np.load(f'pendientes_4pC_FFT_with_Tinf_state_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_basis_size{N}_time_lim{time_lim}'+operators+'.npz')
pendientes_O1 = archives['pendientes_O1']
archives = np.load(f'IPR_mean_skew_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_N{N}_basis_'+tipo_de_base+'.npz')
IPR_means = archives['IPR_means']

r_normed = normalize(rs)
y_mean = normalize(IPR_means)
pendientes_y = normalize(pendientes_O1)


plt.figure(figsize=(16,8))
plt.plot(Ks, y_mean, 'r.-', label='mean')
plt.plot(Ks, pendientes_y, 'b.-', label=r'$\alpha_{O1}$')
plt.plot(Ks, r_normed, 'k.-', label='r')
# plt.vlines(0, 0, 1, ls='dashed', alpha=0.5)
plt.ylabel(r'IPR / r / $\alpha_{O1}$')
plt.xlabel('K')
# plt.xlim(4,10)
plt.grid(True)
plt.legend(loc='best')
plt.savefig(f'Comparacion_IPR_r_decO1_vs_Ks_N{N}_basis_'+tipo_de_base+'.png', dpi=80)
#%%

Ns = 2**np.arange(7,11)

operators = 'AX_BP'
time_lim = 21

O1_plot = []

for N in Ns:
        
    archives = np.load(f'pendientes_4pC_FFT_with_Tinf_state_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_basis_size{N}_time_lim{time_lim}'+operators+'.npz')
    pendientes_O1 = archives['pendientes_O1']
    O1_plot.append(pendientes_O1)
    
plt.figure(figsize=(16,8))
for i in range(4):
    pendientes_O1 = O1_plot[i]
    pendientes_y = normalize(pendientes_O1)
    plt.plot(Ks, pendientes_y, '.-', label=f'N={Ns[i]}')
    
# plt.vlines(0, 0, 1, ls='dashed', alpha=0.5)
plt.ylabel(r'$\alpha_{O1}$')
plt.xlabel('K')
# plt.xlim(4,10)
plt.grid(True)
plt.legend(loc='best')
plt.savefig(f'Comparacion_decO1_vs_Ks_Ns{Ns}.png', dpi=80)

#%% 

IPR_plot = []

for N in Ns:
    archives = np.load(f'IPR_mean_skew_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_N{N}_basis_'+tipo_de_base+'.npz')
    IPR_means = archives['IPR_means']
    IPR_plot.append(IPR_means)

plt.figure(figsize=(16,8))
for i in range(4):
    IPR_means = IPR_plot[i]
    y_mean = normalize(IPR_means)
    plt.plot(Ks, y_mean, '.-', label=f'N={Ns[i]}')

# plt.vlines(0, 0, 1, ls='dashed', alpha=0.5)
plt.ylabel('IPR')
plt.xlabel('K')
# plt.xlim(4,10)
plt.grid(True)
plt.legend(loc='best')
plt.savefig(f'Comparacion_IPR_vs_Ks_Ns{Ns}_'+tipo_de_base+'.png', dpi=80)
