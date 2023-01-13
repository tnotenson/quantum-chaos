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
from scipy.linalg import eig
import scipy.sparse as ss
from scipy.sparse.linalg import eigs

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
    qf = (q + np.random.normal(0,ruido))%modulo
    pf = (p + np.random.normal(0,ruido))%modulo
    
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
    eig_res = np.zeros((Nx,Nx), dtype=np.complex_)
    for j in range(N):
        q,p = qp_from_j(j, Nx, paso, mapa)
        # print(int(q),int(p))
        eig_res[int(q),int(p)] = eigenvector[j]
    return eig_res
#%%
Ns = [90]#np.arange(81,100,2) #np.arange(124,128,2)#np.concatenate((np.arange(20,71,1),np.arange(120,131,2)))#2**np.arange(5,8)
es = [0.00390625]#1/2**np.arange(1,3,2)*110 # abs
resonancias = np.zeros((len(Ns),len(es)))
        
mapa = 'normal'#'absortion'#'dissipation'#'normal'#'cat'#'Harper'#
method = 'Ulam'#'one_trayectory'#
eta = 0.3
a = 2
cx = 1

K = 17#0.971635406

Nc = int(1e3)#int(2.688e7)#int(1e8)#
#%%
k = 45
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
        e, evec = eigs(S, k=k)
        eabs = np.abs(e)
        evec=evec[:,eabs.argsort()[::-1]]
        e = e[eabs.argsort()][::-1]
        t1=time()
        print(f'\nDiagonalización: {t1-t0} seg')
        flag = f'Ulam_approximation_method{method}_mapa{mapa}_Sij_eigenvals_N{Neff}_ruido_abs{ruido}_grilla{cx}N_K{K}_Nc{Nc}'
        np.savez(flag+'.npz', e=e, evec=evec[:,:k])
        del S; del e; del evec;
#%% cargo autoestado ni 
import os
import imageio
ni = 0

Neff = Ns[ni]
ruido = es[0]
# flag = f'Ulam_approximation_method{method}_mapa{mapa}_Sij_eigenvals_N{Neff}_ruido{ruido}_grilla{cx}N_K{K}_Nc{Nc}'
flag = 'Ulam_approximation_methodUlam_mapanormal_Sij_eigenvals_N110_ruido0.00390625_grilla1N_K7_Nc1000'
archives = np.load(flag+'.npz')
e = archives['e']
evec = archives['evec']
# e = np.abs(e)
# evec=evec[:,e.argsort()[::-1]]
# e = np.sort(e)[::-1]
# ies = [1,9]
# guardo los autoestados 
# ni = 0
ri = 0
for i in range(evec.shape[1]):#ies:#
    # i = 1
    hus = np.abs(eigenvec_j_to_qp(evec[:,i]))#**2
    plt.figure()
    plt.title(f'Standard Map. N={Ns[ni]}, e={es[ri]:.0e}, K={K}, i={i}, eval={np.abs(e[i]):.3f}')
    sns.heatmap(hus)
    plt.tight_layout()
    plt.savefig(f'autoestado_n{i}'+flag+'.png', dpi=80)
    plt.close()
# hago gif

# Build GIF
with imageio.get_writer('gif_autoestados_'+flag+'.gif', mode='I', fps=1) as writer:
    for i in range(evec.shape[1]):
        filename = f'autoestado_n{i}'+flag+'.png'
        image = imageio.imread(filename)
        writer.append_data(image)
#%%
def IPR(state, tol = 0.0001):
    if (np.linalg.norm(state)-1) > tol:
        print(np.linalg.norm(state))
    # state = state.full()
    pi = np.abs(state) # calculo probabilidades
    IPR = np.sum(pi**2)/np.sum(pi)**2 # calculo el IPR
    return IPR # devuelvo el IPR

IPRs=np.zeros((evec.shape[1]))
for i in range(evec.shape[1]):
    hus = np.abs(eigenvec_j_to_qp(evec[:,i]))**2
    tol=1e-3
    hus /= np.sum(np.abs(hus))
    aux = IPR(np.abs(hus), tol)
    IPRs[i] = aux

plt.figure(figsize=(10,6))
plt.title(f'Criterio IPR. N={Ns[ni]}, e={es[ri]:.0e}, K={K}')
plt.plot(np.arange(evec.shape[1]), IPRs, 'r.')
plt.vlines(np.arange(evec.shape[1]), 0, IPRs, color='red', alpha=0.8)
plt.xticks(np.arange(evec.shape[1]), rotation=45, fontsize=8)
plt.ylabel('IPR')
plt.xlabel('autoestado')
# plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig(f'IPR_evec'+flag+'.png', dpi=80)
#%% IPR vs N
# aislo autoestado asociado a la resonancia para cada N, chequeando que su 
# autovalor se mantiene cte aprox
# calculo el IPR y lo guardo
IPR_lower = 5e-4
# IPR_treshold = 1e-3
IPR_N = np.zeros(len(Ns)); evals_N = np.zeros(len(Ns)); IPR_localized = np.zeros(len(Ns))
indexes = np.zeros(len(Ns))
index = 1
for j in range(len(Ns)):
    Neff = Ns[j]
    ruido = es[0]
    flag = f'Ulam_approximation_method{method}_mapa{mapa}_Sij_eigenvals_N{Neff}_ruido{ruido}_grilla{cx}N_K{K}_Nc{Nc}'
    archives = np.load(flag+'.npz')
    e = archives['e']
    evec = archives['evec']
    
    IPRs=np.zeros((evec.shape[1]))
    for vi in range(evec.shape[1]):
        hus = np.abs(eigenvec_j_to_qp(evec[:,vi]))**2
        tol=1e-3
        hus /= np.sum(np.abs(hus))
        aux = IPR(np.abs(hus), tol)
        IPRs[vi] = aux
    
    cont=0
    i=1
    
    IPR_localized[j] = IPRs[index]
    while (cont==0 and i<evec.shape[1]):

        # print(f'IPR evec{i} = ', IPRs[i])
        if ((IPRs[i]-min(IPRs[1:]))<IPR_lower):# and (IPRs[i]<IPR_treshold):
            cont=1
        else:
            i+=1
    print(i, np.abs(e[i]))
    indexes[j] = i
    evals_N[j] = np.abs(e[i])
    IPR_N[j] = IPR(evec[:,i])
#%% criterio del overlap

for ni in range(len(Ns)):
    
    Neff = Ns[ni]
    ruido = es[0]
    flag = f'Ulam_approximation_method{method}_mapa{mapa}_Sij_eigenvals_N{Neff}_ruido{ruido}_grilla{cx}N_K{K}_Nc{Nc}'
    archives = np.load(flag+'.npz')
    e = archives['e']
    evec = archives['evec']
    
    e1 = np.abs(evec[:,1])
    overlaps=np.zeros((evec.shape[1]))
    for i in range(evec.shape[1]):
        ei = np.abs(evec[:,i])
        overlap = np.vdot(e1,ei)**2
        # print(overlap)
        overlaps[i] = overlap
    
    plt.figure(figsize=(10,6))
    plt.title(f'Criterio del overlap. N={Ns[ni]}, e={es[ri]:.0e}, K={K}')
    plt.plot(np.arange(evec.shape[1]), overlaps, 'r.')
    plt.vlines(np.arange(evec.shape[1]), 0, overlaps, color='red', alpha=0.8)
    plt.xticks(np.arange(evec.shape[1]), rotation=45, fontsize=8)
    plt.ylabel(r'overlap $|\langle 1 | i \rangle|^2$')
    plt.xlabel(r'autoestado $i$')
    # plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f'overlaps_evec'+flag+'.png', dpi=80)
#%% overlap vs N
# aislo autoestado asociado a la resonancia para cada N, chequeando que su 
# autovalor se mantiene cte aprox
# calculo el IPR y lo guardo
overlap_lim = 0.35
# IPR_treshold = 1e-3
overlap_N = np.zeros(len(Ns)); evals_N = np.zeros(len(Ns)); indexes = np.zeros(len(Ns))
index = 1
for j in range(len(Ns)):
    Neff = Ns[j]
    ruido = es[0]
    flag = f'Ulam_approximation_method{method}_mapa{mapa}_Sij_eigenvals_N{Neff}_ruido{ruido}_grilla{cx}N_K{K}_Nc{Nc}'
    archives = np.load(flag+'.npz')
    e = archives['e']
    evec = archives['evec']
    
    e1 = np.abs(evec[:,1])
    overlaps=np.zeros((evec.shape[1]))
    for i in range(evec.shape[1]):
        ei = np.abs(evec[:,i])
        overlap = np.vdot(e1,ei)**2
        # print(overlap)
        overlaps[i] = overlap
    
    cont=0
    i=1
    
    while (cont==0 and i<evec.shape[1]):

        # print(f'IPR evec{i} = ', IPRs[i])
        if (overlaps[i]<overlap_lim):# and (IPRs[i]<IPR_treshold):
            cont=1
        else:
            i+=1
    print(Ns[j], i, np.abs(e[i]))
    indexes[j] = i
    evals_N[j] = np.abs(e[i])
    overlap_N[j] = overlaps[i]
#%%
# indexes = np.array([7,7,11,12,11,11,11,11,11,11])
for j, i in enumerate(indexes):
    Neff = Ns[j]
    ruido = es[0]
    flag = f'Ulam_approximation_method{method}_mapa{mapa}_Sij_eigenvals_N{Neff}_ruido{ruido}_grilla{cx}N_K{K}_Nc{Nc}'
    archives = np.load(flag+'.npz')
    e = archives['e']
    evec = archives['evec']
    evals_N[j] = np.abs(e[i])
    IPR_N[j] = IPR(evec[:,i])
#%%
plt.figure()
plt.title(f'Standard Map. IPR vs N. e={es[ri]:.0e}, K={K}')
plt.plot(Ns, IPR_N, 'r.')
# plt.plot(Ns[3], IPR(evec[:,11]), 'r.')
# plt.vlines(np.arange(evec.shape[1]), 0, IPRs, color='red', alpha=0.8)
# plt.xticks(np.arange(evec.shape[1]), rotation=45)
plt.ylabel('IPR')
plt.xlabel(r'$N$')
# plt.grid(True)
# plt.ylim(0,0.005)
plt.tight_layout()
plt.show()
plt.savefig('IPR_N'+flag+'.png', dpi=80)
#%% IPR localized
plt.figure()
plt.title(f'Standard Map. Localized estates. e={es[ri]:.0e}, K={K}')
plt.plot(Ns, IPR_localized, 'r.')
plt.hlines(IPR_localized[0], min(Ns), max(Ns))
# plt.vlines(np.arange(evec.shape[1]), 0, IPRs, color='red', alpha=0.8)
# plt.xticks(np.arange(evec.shape[1]), rotation=45)
plt.ylabel('IPR')
plt.xlabel(r'$N$')
# plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('Localized_evecs_IPR_N'+flag+'.png', dpi=80)
#%%
std_N = 1/(2*np.sqrt(Nc))

conv = 2
# from scipy.stats import std
plt.figure()
plt.title(f'Standard Map. Autovalores. e={es[ri]:.0e}, K={K}')
plt.plot(Ns, evals_N, 'r.')
# plt.hlines(np.mean(evals_N[conv:]), min(Ns), max(Ns))
# plt.fill_between(Ns, np.mean(evals_N[conv:])-std_N, np.mean(evals_N[conv:])+std_N, alpha=0.4)
# plt.vlines(np.arange(evec.shape[1]), 0, IPRs, color='red', alpha=0.8)
# plt.xticks(np.arange(evec.shape[1]), rotation=45)
plt.ylabel(r'$\lambda$')
plt.xlabel(r'$N$')
# plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('eval_IPR_N'+flag+'.png', dpi=80)
#%% 2d histogram eigenvalues many Ns

theta = np.linspace(0, 2*np.pi, 100)
   
r= 1

x = r*np.cos(theta)
y = r*np.sin(theta)

dom = np.linspace(-1,1,100)
std_N = 1/(2*np.sqrt(Nc))

plt.figure(figsize=(10,6))
plt.plot(x,y,color='b', lw=1)
plt.fill_betweenx(dom, np.mean(evals_N[conv:])-std_N,np.mean(evals_N[conv:])+std_N, alpha=0.8)
plt.ylabel(r'$\Im(\lambda)$')
plt.xlabel(r'$\Re(\lambda)$')
# plt.xlim(0.20,1.1)
# plt.ylim(-0.25,0.25)
r= 0.5559

x = r*np.cos(theta)
y = r*np.sin(theta)
plt.plot(x,y,color='g', lw=1, alpha=0.7)

k = 45

markers = ['o','v','^','>','<','8','s','p','*']

e_todos = np.zeros((k,len(Ns)))

for j in range(len(Ns)):
    Neff = Ns[j]
    ruido = es[0]
    flag = f'Ulam_approximation_method{method}_mapa{mapa}_Sij_eigenvals_N{Neff}_ruido{ruido}_grilla{cx}N_K{K}_Nc{Nc}'
    archives = np.load(flag+'.npz')
    e = archives['e']
    
    e_todos[:,j] = np.abs(e)
    
    evec = archives['evec']
    
    x = np.abs(np.real(e[int(indexes[j])]))
    y = np.imag(e[int(indexes[j])])
    
    print(x,y)
    
    plt.plot(x,y,'r.', ms=15, label='NL evec', alpha=0.5)
    
    x = np.real(e)
    
    y = np.imag(e)
    plt.plot(x,y,'k.', marker=markers[j%len(markers)], ms=5, label='Ns[j]', alpha=0.45)
    
    # print(np.abs(e[9]),np.abs(e[11]))
# plt.legend(loc='best')    
plt.tight_layout()
plt.savefig(f'autovalores_1er_evec_NL_K{K}_Nmin{min(Ns)}_Nmax{max(Ns)}_Npaso{int(np.diff(Ns).mean())}_k{k}.png')
#%% compraramos autovalores de ambos métodos de Ulam
metodos = ['Ulam', 'one_trayectory']
labels = ['común','1trayectory']
Ncs = [int(1e4), int(2.688e7)]
ni=0
Neff = Ns[ni]
ruido = es[ri]

array = np.zeros((k,len(metodos)))

theta = np.linspace(0, 2*np.pi, 100)
   
r= 1

x = r*np.cos(theta)
y = r*np.sin(theta)

plt.plot(x,y,color='b', lw=1)
plt.ylabel(r'$\Im(\lambda)$')
plt.xlabel(r'$\Re(\lambda)$')
# plt.xlim(0.20,1.1)
# plt.ylim(-0.25,0.25)
plt.tight_layout()

r = 0.69
x = r*np.cos(theta)
y = r*np.sin(theta)

plt.plot(x,y,color='g', lw=1, alpha=0.7)

for j, (method, Nc) in enumerate(zip(metodos,Ncs)):
    flag = f'Ulam_approximation_method{method}_mapa{mapa}_Sij_eigenvals_N{Neff}_ruido{ruido}_grilla{cx}N_K{K}_Nc{Nc}'
    archives = np.load(flag+'.npz')
    e = archives['e']
    x = np.real(e)
    
    y = np.imag(e)
    plt.plot(x,y,'.', marker=markers[j%len(markers)], label=labels[j], ms=5, alpha=0.45)
    
    array[:,j] = np.abs(e)
plt.legend(loc='upper right', framealpha=0.4)#, alpha=0.3)
plt.savefig(f'Comparacion_metodos_Ulam_K{K}_N{Neff}_e{ruido}.png', dpi=80)
#%%
# plt.plot(np.abs(array[:,0]),np.abs(array[:,1]), 'r.',ms=4,alpha=0.8)
#%%    
