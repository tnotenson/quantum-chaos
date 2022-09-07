#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 19:18:37 2022

Leading Ruelle resonances of chaotic maps

authors: Blum and Agam

@author: Tomas Notenson
"""
# import libraries and define rutines
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib as mpl
from time import time
from scipy.sparse.linalg import eigs
import seaborn as sb

dpi = 2*np.pi

def finite_delta(x,s,cant=3):
    d = 0
    for npr in range(cant):
        n = npr - cant//2
        # print(n)
        d+= 1/np.sqrt(np.pi*s)*np.exp(-(x-n)**2/s)
    return d

# dom = np.linspace(0,1,100)
# plt.plot(dom,finite_delta(dom,s=0.01))

def mat_elem_U(N,xp,yp,x,y,K,mapa='estandar',*args,**kwargs): 
    U = finite_delta((yp-(y+K/dpi*np.sin(dpi*x))%1)%1,s)*finite_delta((xp-(x+y+K/dpi*np.sin(dpi*x))%1)%1,s)
    return U

def qp_from_j(j,N):
    paso = 1/N
    qj = (j%N)*paso
    pj = ((j//N)%N)*paso
    return qj, pj

def eigenvec_j_to_qp(eigenvector, mapa='normal'):
    N = len(eigenvector)
    Nx = int(np.sqrt(N))
    eig_res = np.zeros((Nx,Nx), dtype=np.complex_)
    for j in range(N):
        q,p = qp_from_j(j, Nx)
        # print(int(q),int(p))
        eig_res[int(Nx*q),int(Nx*p)] = eigenvector[j]
    return eig_res
#%% define parameters and compute resonances
N = 30
K = 19.74
nvec = 20

ss = np.arange(1,6)*1e-3

es = np.zeros((len(ss),nvec), dtype=np.complex_)

for num,s in tqdm(enumerate(ss),desc='loop s'):
        
    U = np.zeros((N**2,N**2), dtype=np.complex_)
    for i in tqdm(range(N**2), desc='loop 1 celdas'):
        for j in range(N**2):    
            x,y = qp_from_j(i,N)
            xp,yp = qp_from_j(j,N)
            U[i,j] = mat_elem_U(N,xp,yp,x,y,K,mapa='estandar',s=s)
    # U = U
    t0 = time()
    # e, evec = np.linalg.eig(U)
    e, evec = eigs(U,k=nvec)
    print(f'Diagonalization: {time()-t0} s')
    eabs = np.abs(e)
    evec=evec[:,eabs.argsort()[::-1]]
    e = e[eabs.argsort()][::-1]/eabs[0]
    es[num,:] = e
    
    # print(np.abs(e[:nevec]))
#%% plot eigenvalues and unit circle in complex plane
r = 1
theta = np.linspace(0, 2*np.pi, 100)

x = r*np.cos(theta)
y = r*np.sin(theta)

plt.figure(figsize=(10,6))
plt.plot(x,y,color='b', lw=1)
plt.ylabel(r'$\Im(\lambda)$')
plt.xlabel(r'$\Re(\lambda)$')

markers = ['o','v','^','>','<','8','s','p','*']

fit = np.linspace(0,1,len(e))
cmap = mpl.cm.get_cmap('viridis')
color_gradients = cmap(fit)  

for num in range(len(e)): 
    e = es[:,num]
    x = np.real(e)
    y = np.imag(e)
    plt.plot(x,y,'.', color=color_gradients[num], marker=markers[num%len(markers)] ,ms=7, alpha=0.65)
#%% plot eigenvectors


# for i in range(evec.shape[1]):#ies:#
i = 5
hus = np.abs(eigenvec_j_to_qp(evec[:,i]))#**2
plt.figure()
plt.title(f'Standard Map. N={N} K={K}, i={i}, eval={np.abs(e[i]):.3f}')
sb.heatmap(hus)
plt.tight_layout()
# plt.savefig(f'autoestado_n{i}'+flag+'.png', dpi=80)
# plt.close()