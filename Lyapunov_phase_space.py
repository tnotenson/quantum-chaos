#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 15:51:24 2022

@author: tomasnotenson
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from qutip import *
# from math import factorial
from time import time
from tqdm import tqdm # customisable progressbar decorator for iterators

# Analytic
def lystandar(q,k):
    # pf = np.mod(p+k*np.sin(2.*np.pi*q)/(2.*np.pi),1.)
    # qf = np.mod(q+pf,1.)
    a=1
    b=k*np.cos(2*np.pi*q)
    c=1
    d=1+k*np.cos(2*np.pi*q)
    m=[[a,b],[c,d]]
#    print(a*d-b*c)
    # print(m)
    l1=np.linalg.eig(m)
    return l1[0]
#%%
# # number of initial conditions
# nci=int(1e3)
# # limits of K array
# k0=0.1
# k1=20.
# # # points
# npp=100
# # K array
# kk = np.linspace(k0, k1, npp+1)
# # kk = Ks
# # Lyapunovs
# ly1 = []

# for count10, k in tqdm(enumerate(kk), desc='K loop'):
#     # print(f'Iteration {count10}',f'K = {k}')
#     # initial conditions
#     ci=np.random.rand(nci,2)
#     # counts of l1.imag() == 0
#     nc=0
#     # initialize lyapunovs
#     lyap=0.
#     lyap2=0.
#     for count0 in range(nci):
#         q0, p0 = ci[count0]
#         # eigenvalues of standar map's matrix
#         l1, l2 = lystandar(q0,k)
#         # check imaginary part of eigenvalues
#         if(np.imag(l1) == 0):
#             # take absolute value
#             l1=np.abs(l1)
#             l2=np.abs(l2)
#             # select greatest eigenvalue
#             if(l1 >l2):
#                 # actualize lyapunovs
#                 lyap=lyap+np.log(l1)
#             else:
#                 # actualize lyapunovs
#                 lyap=lyap+np.log(l2)
#             nc+=1
#     # save values
#     ly1.append(lyap/nci)
# #%%
# plt.plot(kk,ly1)
# plt.xlabel(r'$K$')
# plt.ylabel(r'$\lambda_L$')
# plt.grid()

#%% estudio el lyapunov en el espacio de fases
N = 100
k = 5
# sólo varío q porque el lyapunov no depende de p
qs = np.arange(0,N)/N; #ps = np.arange(0,N)/N

nci = 100

lyq = np.zeros(N)

for nqi, q in tqdm(enumerate(qs), desc='q loop'):
    # initial conditions
    ci=np.random.rand(nci,1)/N
    ci+=q 
    # print(ci)
    # counts of l1.imag() == 0
    nc=0
    # initialize lyapunovs
    lyap=np.zeros(nci)
    
    for count0 in range(nci):
        q0 = ci[count0][0]
        # print(q0)
        # print(ci[count0])
        # eigenvalues of standar map's matrix
        l1, l2 = lystandar(q0,k)
        # check imaginary part of eigenvalues
        if(np.imag(l1) == 0):
            assert np.abs(l1*l2-1) < 1e-3, 'No conserva el área'
            # take absolute value
            l1=np.abs(l1)
            l2=np.abs(l2)
            # select greatest eigenvalue
            if(l1 >l2):
                # actualize lyapunovs
                lyap[count0]=np.log(l1)
            else:
                # actualize lyapunovs
                lyap[count0]=np.log(l2)
            nc+=1
    # save values
    lyq[nqi] = np.mean(lyap)
        
#%%
plt.title(f'Standard Map. N={N}, K={k}, nci={nci}')
plt.plot(qs, lyq)
plt.xlabel(r'$q$')
plt.ylabel(r'$\lambda_L$')
plt.grid(True)
plt.show()
plt.savefig(f'Standard_Map_N={N}_K={k}_nci={nci}.png', dpi=80)
#%% 
plt.figure()
plt.title(f'Standard Map. N={N}, K={k}, nci={nci}')
sns.histplot(lyq,stat='density')
# plt.ylabel(r'$q$')
plt.xlabel(r'$\lambda_L$')
plt.grid(True)
plt.savefig(f'heatmap_Standard_Map_N={N}_K={k}_nci={nci}.png', dpi=80)
#%% lyaponuv vs (q,k)
N = 100
# q array
qs = np.arange(0,N)/N; #ps = np.arange(0,N)/N
k0=0.1
k1=20.
# # points
npp=100
# K array
kk = np.linspace(k0, k1, npp)
# kk = Ks
# number of initial condition
nci = 100
lyqk = np.zeros((N, len(kk)))


for count10, k in tqdm(enumerate(kk), desc='K loop'):
    for nqi, q in enumerate(qs):
        # initial conditions
        ci=np.random.rand(nci,1)/N
        ci+=q 
        # print(ci)
        # counts of l1.imag() == 0
        nc=0
        # initialize lyapunovs
        lyap=np.zeros(nci)
        
        for count0 in range(nci):
            q0 = ci[count0][0]
            # print(q0)
            # print(ci[count0])
            # eigenvalues of standar map's matrix
            l1, l2 = lystandar(q0,k)
            # check imaginary part of eigenvalues
            if(np.imag(l1) == 0):
                assert np.abs(l1*l2-1) < 1e-3, 'No conserva el área'
                # take absolute value
                l1=np.abs(l1)
                l2=np.abs(l2)
                # select greatest eigenvalue
                if(l1 >l2):
                    # actualize lyapunovs
                    lyap[count0]=np.log(l1)
                else:
                    # actualize lyapunovs
                    lyap[count0]=np.log(l2)
                nc+=1
        # save values
        lyqk[nqi, count10] = np.mean(lyap)

#%% lyapunov vs K
N = 100
# q array
qs = np.arange(0,N)/N; #ps = np.arange(0,N)/N
k0=0
k1=20.1
# # points
# npp=100
# K array
Kpaso = 0.2
kk = np.arange(k0, k1, Kpaso)
# kk = Ks
# number of initial condition
nci = 100
lyqk = np.zeros((len(kk)))


for count10, k in tqdm(enumerate(kk), desc='K loop'):
    # initial conditions
    ci=np.random.rand(nci,1)/N
    ci+=q 
    # print(ci)
    # counts of l1.imag() == 0
    nc=0
    # initialize lyapunovs
    lyap=np.zeros(nci)
    
    for count0 in range(nci):
        q0 = ci[count0][0]
        # print(q0)
        # print(ci[count0])
        # eigenvalues of standar map's matrix
        l1, l2 = lystandar(q0,k)
        # check imaginary part of eigenvalues
        if(np.imag(l1) == 0):
            assert np.abs(l1*l2-1) < 1e-3, 'No conserva el área'
            # take absolute value
            l1=np.abs(l1)
            l2=np.abs(l2)
            # select greatest eigenvalue
            if(l1 >l2):
                # actualize lyapunovs
                lyap[count0]=np.log(l1)
            else:
                # actualize lyapunovs
                lyap[count0]=np.log(l2)
            nc+=1
    # save values
    lyqk[count10] = np.mean(lyap)

#%% plot 3D
sns.set_style('whitegrid')

X1, Y1 = np.meshgrid(qs, kk)

Z1 = lyqk

ax = plt.axes(projection='3d')   
# axes.plot_surface(X1, Y1, Z1)
ax.plot_wireframe(X1, Y1, Z1)
ax.set_xlabel(r'$q$')
ax.set_ylabel(r'$K$')
ax.set_zlabel(r'$\lambda_L$')
plt.show() 