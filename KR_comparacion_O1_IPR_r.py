#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 20:38:10 2022

@author: tomasnotenson
"""


import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from qutip import *
import time
from tqdm import tqdm # customisable progressbar decorator for iterators
from cmath import phase
from scipy.optimize import curve_fit
plt.rcParams.update({
"text.usetex": True,
"font.family": "sans-serif",
"font.sans-serif": ["Helvetica"], "font.size": 24})

def normalize(array):
    return (array - min(array))/(max(array)-min(array))

def func(x,m,b):
    return m*x+b

N = 1000
tipo_de_base = 'gaussiana'#'integrable'
Kpaso = .5
Ks = np.arange(1,10.1,Kpaso)#
time_lim = 21


archives = np.load(f'4pC_FFT_with_Tinf_state_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_basis_size{N}_time_lim{time_lim}AX_BP.npz')

rs = archives['r_Ks']
r_normed = normalize(rs)

O1 = archives['O1']

archives = np.load(f'IPR_mean_skew_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_N{N}_basis_integrable_wK1.npz')

IPR_means = archives['IPR_means']
y_mean = normalize(IPR_means)

# IPR_skews = archives['IPR_skews']
# y_skew = normalize(IPR_skews)

pendientes = np.zeros((len(Ks)))
sigma_pendientes = np.zeros((len(Ks)))

Kfrom = 0
tmin = 0
tmax = 16
plott = 'O1'

print(O1.shape)

for k, K in enumerate(Ks):
    
    O1s = O1[k]
    
    x = np.arange(tmin,tmax)
    
    if plott == 'Var':
        Var = (O1s - Cs**2)#/dimension#/N
        y_Var = np.log10(Var)
        y = y_Var[tmin:tmax]
    elif plott == 'O1':
        y_O1 = np.log10(O1s)
        y = y_O1[tmin:tmax]
    elif plott == 'C2':
        y_Cs = np.log10(Cs**2)
        y = y_Cs[tmin:tmax]
        
    par, covar = curve_fit(func, x, y)
    sigma_m, sigma_b = np.sqrt(np.diag(covar))
    m, b = par
    
    pendientes[k] = np.abs(m)
    sigma_pendientes[k] = np.abs(sigma_m)

y_pend = normalize(pendientes)
y_pend_sigma = (sigma_pendientes - min(pendientes))/(max(pendientes)-min(pendientes))
#%% plot

plt.figure(figsize=(16,8))
plt.title(f'A=X, B=P. N={N}')
plt.plot(Ks, y_mean, 'r.-', label=r'$\langle IPR \rangle$')
# plt.plot(Ks, y_skew, 'b.-', label='skew')
plt.plot(Ks, r_normed, 'k.-', label='r')
plt.plot(Ks, y_pend, 'b.-', label=r'$\alpha_{O1}$')
plt.errorbar(Ks, y_pend, y_pend_sigma, fmt='b', ms=.1)
# plt.vlines(0, 0, 1, ls='dashed', alpha=0.5)
plt.ylabel('IPR')
plt.xlabel('K')
# plt.xlim(4,10)
plt.grid(True)
plt.legend(loc='best')
plt.savefig(f'IPR_pend_r_vs_Ks_N{N}_basis_'+tipo_de_base+'_wK1.png', dpi=80)
