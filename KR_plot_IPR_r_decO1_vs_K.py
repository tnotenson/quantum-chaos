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
N = 2**7
Nstates = int(2*N)
hbar = 1/(2*np.pi*N)
# q0 = 0
# p0 = 2/10
sigma = 1#np.power(hbar,1/2)

tipo_de_base = 'gaussiana'#'integrable_wK0'#
Kpaso = .05
Ks = np.arange(0,10.1,Kpaso)#

texto = np.loadtxt('ipr_dek_std.n200.txt')
K_nacho, PR_nacho = texto[:,0], texto[:,1]
K_nacho  *= 4*np.pi**2
IPR_nacho = 1/PR_nacho

archives = np.load(f'r_vs_K_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_N{N}.npz')
rs = archives['rs']

operators = 'AX_BP'
time_lim = 21
archives = np.load(f'pendientes_4pC_FFT_with_Tinf_state_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_basis_size2048_time_lim{time_lim}'+operators+'.npz')
pendientes_O1 = archives['pendientes_O1']

t_sat = 500
time_lim = int(6e3)
archives = np.load(f'IPR_O1_from_t_sat{t_sat}_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_time_lim{time_lim}_basis_size{N}'+operators+'.npz')
IPRs = archives['IPRs']

N = 200#2**6
Nstates = int(2*N)#N/2)
Kpaso = 1
Ks = np.arange(0,10.1,Kpaso)#

archives = np.load(f'IPR_Husimi_vs_Ks_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_N{N}_coherent_basis_grid{Nstates}x{Nstates}_numpy.npz')
IPR_means = archives['IPR_means']

y_nacho = normalize(IPR_nacho)
y_fourier = normalize(IPRs)
r_normed = normalize(rs)
y_mean = normalize(IPR_means)
pendientes_y = normalize(pendientes_O1)

N = 2**7
Kpaso = 0.05
Ks = np.arange(0,10.1,Kpaso)#

plt.figure(figsize=(16,8))
plt.title(f'N={N}')
plt.plot(K_nacho, y_nacho, 'c.-', label=r'$\langle IPR \rangle_{coherentes}^{Nacho}$')
plt.plot(Ks, y_fourier, 'g.-', label='IPR O1')
plt.plot(Ks, r_normed, 'k.-', label='r')
plt.plot(Ks, pendientes_y, 'b.-', label=r'$\alpha_{O1}$')

Kpaso = 1
Ks = np.arange(0,10.1,Kpaso)#

plt.plot(Ks, y_mean, 'r.-', label=r'$\langle IPR \rangle_{coherentes}$')

# plt.vlines(0, 0, 1, ls='dashed', alpha=0.5)
plt.ylabel(r'IPR / r / $\alpha_{O1}$')
plt.xlabel('K')
# plt.xlim(4,10)
plt.grid(True)
plt.legend(loc='best')
plt.savefig(f'Comparacion_IPR_Nacho_r_decO1_vs_Ks_N{N}_basis_'+tipo_de_base+'.png', dpi=80)
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
Ns = 2**np.arange(4,9)

IPR_plot = []

for N in Ns:
    Nstates= np.int32(N/2)  #% numero de estados coherentes de la base
    archives = np.load(f'IPR_vs_Ks_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_N{N}_coherent_basis_grid{Nstates}x{Nstates}.npz')
    IPR_means = archives['IPR_means']
    IPR_plot.append(IPR_means)

plt.figure(figsize=(16,8))
for i in range(len(Ns)):
    IPR_means = IPR_plot[i]
    y_mean = normalize(IPR_means)
    plt.plot(Ks, y_mean, '.-', label=f'N={Ns[i]}')

# plt.vlines(0, 0, 1, ls='dashed', alpha=0.5)
plt.ylabel('IPR')
plt.xlabel('K')
# plt.xlim(4,10)
plt.grid(True)
plt.legend(loc='best')
plt.savefig(f'Comparacion_IPR_vs_Ks_Ns{Ns}_coherent_basis.png', dpi=80)
