#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 17:23:46 2022

@author: tomasnotenson
"""

import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from time import time
from tqdm import tqdm # customisable progressbar decorator for iterators
from numba import jit
# importing "cmath" for complex number operations
from cmath import phase
from random import random
import seaborn as sns
from scipy.stats import norm, multivariate_normal
from scipy.sparse.linalg import eigs
import scipy.sparse as ss

plt.rcParams.update({
"text.usetex": True,
"font.family": "sans-serif",
"font.sans-serif": ["Helvetica"], "font.size": 16})

dpi = 2*np.pi

@jit
def standard_map(q,p,K):
    pf = (p + K/(dpi)*np.sin(dpi*q))%1
    qf = (q + pf)%1
    return qf, pf

@jit
def standard_map_absorption(q,p,K,a=2):
    pf = (p + K*np.sin(q+p/2))*(a*K)-a*K/2
    qf = (q + (p+pf)/2)%dpi
    
    # assert (-a*K/2<=pf and pf<=a*K/2), 'no está en el intervalo permitido [-aK/2,aK/2]'
    
    return qf, pf

@jit
def standard_map_dissipation(q,p,K,eta=0.3):
    pf = (p + K*np.sin(q))%(8*np.pi)-4*np.pi
    qf = (q + pf)%(dpi)
    
    assert (-4*np.pi<=pf and pf<=4*np.pi), 'no está en el intervalo [-4pi,4pi]'
    
    return qf, pf

@jit
def Harper_map(q,p,K):
    pf = (p-K*np.sin(dpi*q))%1
    qf = (q+K*np.sin(dpi*pf))%1
    return qf, pf    

@jit
# probar con K = 0.01, 0.02
def perturbed_cat_map(q,p,K):
    pf = (p + q - dpi*K*np.sin(dpi*q))%1
    qf = (q + pf + dpi*K*np.sin(dpi*pf))%1
    return qf, pf

@jit
def CI(qj,pj,paso):
    rq = np.random.uniform()*paso
    rp = np.random.uniform()*paso
    
    q = qj+rq
    p = pj+rp
    
    # print('CI', q,p)
    return q,p

@jit
def cae_en_celda(qf,pf,qi,pi,paso):
    cond1 = (0<(qf-qi) and (qf-qi)<paso)
    cond2 = (0<(pf-pi) and (pf-pi)<paso)
    
    cond = (cond1 and cond2)
    # if cond:
    #     print('cond1', qf, pf, 'dif', (qf-qi), (pf-pi))
    return cond

@jit
def CI_and_evolution(qj,pj,paso,K,mapa='normal',ruido=10):
    # tomo una CI dentro de la celda j
    q, p = CI(qj, pj, paso)
    # q, p = gaussian_kernel(q, p, paso, ruido)
    
    # evoluciono un paso 
    qf, pf, nqi, npi = evolution(q, p, K, mapa)
    
    return qf, pf, nqi, npi

@jit
def gaussian_kernel(q,p,paso,ruido,modulo=1):
    # rv = norm(0, ruido*paso)
    qf = (q + np.random.normal(0,paso/ruido))%modulo
    pf = (p + np.random.normal(0,paso/ruido))%modulo
    
    return qf, pf

@jit
def evolution(q,p,K,mapa='normal'):
    # evoluciono un paso 
    if mapa=='Harper':
        qf, pf = Harper_map(q,p,K)
        
        nqi = qf//paso
        npi = pf//paso
        
    if mapa=='cat':
        qf, pf = perturbed_cat_map(q,p,K)
        
        nqi = qf//paso
        npi = pf//paso
        
    if mapa=='normal':
        qf, pf = standard_map(q,p,K)
        
        nqi = qf//paso
        npi = pf//paso
        
    elif mapa=='absortion':
        a=2
        qf, pf = standard_map_absorption(q,p,K,a)
        
        if not(-a*K/2<=pf and pf<=a*K/2):
            q, p = CI(q, p, paso)
            qf, pf = evolution(q,p,K,mapa='absortion')
        
        nqi = qf/dpi//paso
        npi = (pf+a*K/2)/(a*K)//paso
                
    elif mapa=='dissipation':
        eta=0.3
        qf, pf = standard_map_dissipation(q,p,K,eta)
        
        nqi = qf/dpi//paso
        npi = (pf+4*np.pi)/(8*np.pi)//paso
        
    return qf, pf, nqi, npi

@jit
def n_from_qp(qf,pf,paso,mapa='normal'):
    
    
    if (mapa=='normal' or mapa=='cat' or mapa=='Harper'):
                
        # print(qf, pf, paso)
        nqi = qf//paso
        npi = pf//paso
        
    elif mapa=='absortion':
        a=2
        
        nqi = qf/dpi//paso
        npi = (pf+a*K/2)/(a*K)//paso
                
    elif mapa=='dissipation':
        
        nqi = qf/dpi//paso
        npi = (pf+4*np.pi)/(8*np.pi)//paso
        
    return nqi, npi

@jit
def qp_from_j(j,Nx,paso,mapa='normal'):
    if (mapa=='normal' or mapa=='cat' or mapa=='Harper'):
        qj = (j%Nx)*paso
        pj = ((j//Nx)%Nx)*paso
    elif mapa=='absortion':
        a = 2
        qj = (j%Nx)*paso*dpi
        pj = ((j//Nx)%Nx)*paso*(a*K)-a*K/2
    elif mapa=='dissipation':
        # eta = 0.3
        qj = (j%Nx)*paso*dpi
        pj = ((j//Nx)%Nx)*paso*(8*np.pi)-4*np.pi
    return qj, pj

@jit
def Ulam(N,Nx,paso,Nc,K,mapa,ruido=10,modulo=1):
    
    S = np.zeros((N, N), dtype=np.float64)
    
    for j in tqdm(range(N), 'Ulam approximation'):
        # celda j fija
        # límites inf. de la celda j fija
        qj, pj = qp_from_j(j, Nx, paso, mapa)
        # repito Nc veces
        for nj in range(Nc):
            # tomo una CI random dentro de la celda j 
            # y luego evoluciono un paso
            qf, pf, nqi, npi = CI_and_evolution(qj,pj,paso,K,mapa,ruido)
            qf, pf = gaussian_kernel(qf, pf, paso, ruido,modulo)
            nqi, npi = n_from_qp(qf, pf, paso,mapa)
            # print('llega')
            i = int(nqi+npi*Nx)
            
            # print(qf, pf, nqi, npi, i)
            
            S[i,j] += 1
        S[:,j] /= Nc
        assert  (np.sum(S[:,j]) - 1.0) < 1e-3, 'ojo con sum_i Sij'
    return S

@jit
def Ulam_one_trayectory(N,Nx,paso,Nc,K,mapa,symmetry=True):
    Nmitad = N#//2
    
    # S = np.zeros((Nmitad,Nmitad))
    S = np.zeros((Nmitad, Nmitad), dtype=np.float64)
    # inicializo en una condición inicial (cercana al atractor)
    # celda j (en lo posible cerca del atractor)
    qj = 0.1/(2*np.pi); pj = 0.1/(2*np.pi)
    # tomo una CI random en la celda j
    q, p = CI(qj,pj,paso)

    qf, pf = q, p
    nqi, npi = n_from_qp(qf, pf, paso, mapa)
    # evoluciono hasta t=100
    # for nj in range(100):
    #     # evoluciono un tiempo
    #     qf, pf, nqi, npi = evolution(qf, pf, K, mapa)   
    normaj = np.zeros((Nmitad))
    # cont=0
    for nj in tqdm(range(Nc), desc='loop evolution'):
        
        # número de celda
        j = int(nqi+npi*Nx)
        
        assert j < Nx**2, f'number of cell j={j} out of bounds. Con nqi={nqi} y npi={npi}.'
        
        # evoluciono un tiempo
        qf, pf, nqi, npi = evolution(qf,pf,K,mapa)
        qf, pf = gaussian_kernel(qf, pf, paso,ruido,modulo=1)
        
        nqi, npi = n_from_qp(qf, pf, paso, mapa)
        
        # número de celda
        i = int(nqi+npi*Nx)
        
        assert i < Nx**2, f'number of cell i={i} out of bounds. Con nqi={nqi} y npi={npi}.'
        
        
        # if pf>1/2:
        #     cont+=1
        
        # print(qf, pf, nqi, npi, i)
        
        S[i,j] += 1
        normaj[j] += 1
    for j in range(Nmitad):
        if normaj[j] != 0:
            S[:,j] /= normaj[j]
            # print(np.sum(S[:,j]))
        assert  (np.sum(S[:,j]) - 1.0) < 1e-3, f'ojo con sum_i Sij = {np.sum(S[:,j])} para j={j}'
    # print(cont/Nc)
    return S

@jit
def eigenvec_j_to_qp(eigenvector, mapa='normal'):
    N = len(eigenvector)
    Nx = int(np.sqrt(N))
    paso = 1
    eig_res = np.zeros((Nx,Nx))
    for j in range(N):
        q,p = qp_from_j(j, Nx, paso, mapa)
        # print(int(q),int(p))
        eig_res[int(q),int(p)] = eigenvector[j]
    return eig_res
#%%
Ns = np.arange(128,130,2) #np.arange(124,128,2)#np.concatenate((np.arange(20,71,1),np.arange(120,131,2)))#2**np.arange(5,8)
es = [1e10]#np.logspace(2,4,1) 
resonancias = np.zeros((len(Ns),len(es)))
        
mapa = 'normal'#'absortion'#'dissipation'#'normal'#'cat'#'Harper'#
method = 'Ulam'#'one_trayectory'#
eta = 0.3
a = 2
cx = 1

K = 12#0.971635406

Nc = int(1e3)
#%%

for ni in tqdm(range(len(Ns)), desc='loop ni'):
    for ri in tqdm(range(len(es)), desc='loop ri'):
                
        Neff = Ns[ni]
        Nx = int(cx*Neff)
        N = Nx**2
        ruido = es[ri]
        
        print(f'N={Neff}d',f'e={ruido}')
        
        paso = 1/Nx
                
        if (mapa=='normal' or mapa=='absortion' or mapa=='cat' or mapa=='Harper'):
            if method=='one_trayectory':
                S = Ulam_one_trayectory(N, Nx, paso, Nc, K, mapa)    
            elif method=='Ulam':
                S = Ulam(N, Nx, paso, Nc, K, mapa, ruido)
            
        elif mapa=='dissipation':
            S = Ulam_one_trayectory(N, Nx, paso, Nc, K, mapa)
            
        # hinton(S)
        # mitad = int(S.shape[0]/2)
        # diagonalize operator
        t0=time()
        e, evec = eigs(S,k=k)#np.linalg.eig(S)
        t1=time()
        print(f'\nDiagonalización: {t1-t0} seg')
        flag = f'Ulam_approximation_method{method}_mapa{mapa}_Sij_eigenvals_N{Neff}_ruido{ruido}_grilla{cx}N_K{K}_Nc{Nc}'
        np.savez(flag+'.npz', e=e, evec=evec[:,:k])
#%%
flag = f'Ulam_approximation_method{method}_mapa{mapa}_Sij_eigenvals_N{Neff}_ruido{ruido}_grilla{cx}N_K{K}_Nc{Nc}'
archives = np.load(flag+'.npz')
e = archives['e']
evec = archives['evec']
ies = [1,9]
for i in range(evec.shape[1]):#ies:#
    # i = 1
    vecqp = np.abs(eigenvec_j_to_qp(evec[:,i]))**2
    plt.figure()
    plt.title(f'Standard Map. N={Ns[ni]}, e={es[ri]:.0e}, K={K}')
    sns.heatmap(vecqp)
    plt.savefig(f'autoestado_n{i}'+flag+'.png', dpi=80)