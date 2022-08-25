#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 20:38:28 2022

@author: tomasnotenson
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from qutip import *
from time import time
from tqdm import tqdm # customisable progressbar decorator for iterators
from sklearn.linear_model import LinearRegression #Regresión Lineal con scikit-learn

# parameters
N = 5000
Kpaso = 20/100
Ks = np.arange(0,20,Kpaso)
# some filename definitions
opA = 'X'
opB = 'P'
operatorss = 'A'+opA+'_B'+opB
time_lim = int(2e1+1) # number of kicks
phi = []#np.identity(N)#-sustate
modif = ''#'_sust_evec0'
state = '_Tinf_state'
flag = '4pC_FFT_with'+state
#%% open file for writing

f = open("pendientes.txt", "a")
#%% load file .npz
file = flag+f'_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_basis_size{N}_time_lim{time_lim}'+operatorss+modif+'.npz'#+'_evolucion_al_reves' _centro{n0}
archives = np.load(file)
O1_Ks = archives['O1']

k = 15

# create plot
plt.figure(figsize=(16,8))
plt.title(f'A={opA}, B={opB}. N={N}')

# variables for plot
O1s = O1_Ks[k]
y = np.abs(O1s)/N#np.log10(O1s)

times = np.arange(0,time_lim)
x = times

# fit limits
linf=5
# lsup=9
# variables for fit
xt = x[linf:lsup].reshape(-1,1)
yt = np.log(y[linf:lsup])
# fit
regresion_lineal = LinearRegression() # creamos una instancia de LinearRegression
regresion_lineal.fit(xt, yt) 

# estimated parameters
m = regresion_lineal.coef_
b = regresion_lineal.intercept_

# # fit limits
# linf=lsup-1
lsup=12
# variables for fit
xt = x[linf:lsup].reshape(-1,1)
yt = np.log(y[linf:lsup])
# fit
regresion_lineal = LinearRegression() # creamos una instancia de LinearRegression
regresion_lineal.fit(xt, yt) 

# estimated parameters
m2 = regresion_lineal.coef_
b2 = regresion_lineal.intercept_

# # # fit limits
# linf=lsup-1
# lsup=11
# # variables for fit
# xt = x[linf:lsup].reshape(-1,1)
# yt = np.log(y[linf:lsup])
# # fit
# regresion_lineal = LinearRegression() # creamos una instancia de LinearRegression
# regresion_lineal.fit(xt, yt) 

# # estimated parameters
# m3 = regresion_lineal.coef_
# b3 = regresion_lineal.intercept_

# # # fit limits
# linf=11
# lsup=16
# # variables for fit
# xt = x[linf:lsup].reshape(-1,1)
# yt = np.log(y[linf:lsup])
# # fit
# regresion_lineal = LinearRegression() # creamos una instancia de LinearRegression
# regresion_lineal.fit(xt, yt) 

# # estimated parameters
# m4 = regresion_lineal.coef_
# b4 = regresion_lineal.intercept_


f.write(f'{Ks[k]}, {np.exp(m[0]/2)}, {np.exp(m2[0]/2)}')#', {np.exp(m3[0]/2)}')#', {np.exp(m4[0]/2)}')
f.write('\n')

print(f'K={Ks[k]}, eval = {np.exp(m[0]/2):.4f},  {np.exp(m2[0]/2):.4f}')

plt.plot(x, y, '.-', ms=10, lw=1.5,  color='blue', label=f'K={Ks[k]:.2f}')
plt.plot(x, np.exp(m*x+b), '-r', lw=1.5, label=f'{np.exp(m[0]/2):.3f}**(2*t) ajuste', alpha=0.6)
plt.plot(x, np.exp(m2*x+b2), '-g', lw=1.5, label=f'{np.exp(m2[0]/2):.3f}**(2*t) ajuste', alpha=0.6)
# plt.plot(x, np.exp(m3*x+b3), '-c', lw=1.5, label=f'{np.exp(m3[0]/2):.3f}**(2*t) ajuste', alpha=0.6)
# plt.plot(x, np.exp(m4*x+b4), '-k', lw=1.5, label=f'{np.exp(m4[0]/2):.3f}**(2*t) ajuste', alpha=0.6)
plt.ylabel(r'$O_1$')
plt.xlabel(r'$t$')
plt.xticks(times)
plt.yscale('log')
plt.ylim(1e-6,1)
# plt.xlim(-0.01,15)
plt.grid()
plt.legend(loc = 'best')

file = flag+f'_K{Ks[k]:.1f}_basis_size{N}_time_lim{time_lim}'+operatorss+modif+'.png'
plt.savefig(file, dpi=80)