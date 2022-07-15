#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 16:26:03 2022

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

plt.rcParams.update({
"text.usetex": True,
"font.family": "sans-serif",
"font.sans-serif": ["Helvetica"], "font.size": 16})

dpi = 2*np.pi

def standard_map(q,p,K):
    pf = (p + K/(dpi)*np.sin(dpi*q))%1
    qf = (q + pf)%1
    return qf, pf

def standard_map_absorption(q,p,K,a=2):
    pf = (p + K*np.sin(q+p/2))*(a*K)-a*K/2
    qf = (q + (p+pf)/2)%dpi
    
    # assert (-a*K/2<=pf and pf<=a*K/2), 'no está en el intervalo permitido [-aK/2,aK/2]'
    
    return qf, pf

def standard_map_dissipation(q,p,K,eta=0.3):
    pf = (p + K*np.sin(q))%(8*np.pi)-4*np.pi
    qf = (q + pf)%(dpi)
    
    assert (-4*np.pi<=pf and pf<=4*np.pi), 'no está en el intervalo [-4pi,4pi]'
    
    return qf, pf


# probar con K = 0.01, 0.02
def perturbed_cat_map(q,p,K):
    pf = (p + q - dpi*K*np.sin(dpi*q))%1
    qf = (q + pf + dpi*K*np.sin(dpi*pf))%1
    return qf, pf

def CI(qj,pj,paso):
    rq = random()*paso
    rp = random()*paso
    
    q = qj+rq
    p = pj+rp
    
    # print('CI', q,p)
    return q,p

def cae_en_celda(qf,pf,qi,pi,paso):
    cond1 = (0<(qf-qi) and (qf-qi)<paso)
    cond2 = (0<(pf-pi) and (pf-pi)<paso)
    
    cond = (cond1 and cond2)
    # if cond:
    #     print('cond1', qf, pf, 'dif', (qf-qi), (pf-pi))
    return cond

def CI_and_evolution(qj,pj,paso,K,mapa='normal'):
    # tomo una CI dentro de la celda j
    q, p = CI(qj, pj, paso)
    
    # evoluciono un paso 
    qf, pf, nqi, npi = evolution(q, p, K, mapa)
    
    return qf, pf, nqi, npi

def evolution(q,p,K,mapa='normal'):
    # evoluciono un paso 
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

def qp_from_j(j,Nx,paso,mapa='normal'):
    if mapa=='cat':
        qj = (j%Nx)*paso
        pj = ((j//Nx)%Nx)*paso
    elif mapa=='normal':
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

def Ulam(N,Nx,paso,Nc,K,mapa):
    
    S = np.zeros((N,N))
    
    for j in tqdm(range(N), desc='j loop'):
        # celda j fija
        # límites inf. de la celda j fija
        qj, pj = qp_from_j(j, Nx, paso, mapa)
        # repito Nc veces
        for nj in range(Nc):
            # tomo una CI random dentro de la celda j 
            # y luego evoluciono un paso
            qf, pf, nqi, npi = CI_and_evolution(qj,pj,paso,K,mapa)
            
            i = int(nqi+npi*Nx)
            
            # print(qf, pf, nqi, npi, i)
            
            S[i,j] += 1
        S[:,j] /= Nc
        assert  (np.sum(S[:,j]) - 1.0) < 1e-3, 'ojo con sum_i Sij'
    return S

def Ulam_one_trayectory(N,Nx,paso,Nc,K,mapa):
    
    S = np.zeros((N,N))
    
    # inicializo en una condición inicial (cercana al atractor)
    # celda j (en lo posible cerca del atractor)
    qj = np.pi
    pj = 0
    # tomo una CI random en la celda j
    q, p = CI(qj,pj,paso)
    
    qf, pf = q, p
    
    # evoluciono hasta t=100
    for nj in range(100):
        # evoluciono un tiempo
        qf, pf, nqi, npi = evolution(qf, pf, K, mapa)   
        
    for nj in range(Nc):
        
        # número de celda
        j = int(nqi+npi*Nx)
        
        assert j < Nx**2, f'number of cell j={j} out of bounds. Con nqi={nqi} y npi={npi}.'
        
        # evoluciono un tiempo
        qf, pf, nqi, npi = evolution(qf,pf,K,mapa)
        
        # número de celda
        i = int(nqi+npi*Nx)
        
        assert i < Nx**2, f'number of cell i={i} out of bounds. Con nqi={nqi} y npi={npi}.'
        
        # print(qf, pf, nqi, npi, i)
        
        S[i,j] += 1
    S[:,j] /= Nc
    assert  (np.sum(S[:,j]) - 1.0) < 1e-3, 'ojo con sum_i Sij'
    return S
        
        
#%%
K = 7

Neff = 140#2**6
cx = 1
Nx = int(cx*Neff)
N = Nx**2

mapa = 'normal'#'absortion'#'dissipation'#'cat'#
eta = 0.3
a = 2

paso = 1/Nx

Nc = int(1e4)

if (mapa=='normal' or mapa=='absortion' or mapa=='cat'):
    S = Ulam(N, Nx, paso, Nc, K, mapa)
elif mapa=='dissipation':
    S = Ulam_one_trayectory(N, Nx, paso, Nc, K, mapa)
    
# diagonalize operator
e, evec = np.linalg.eig(S)
flag = f'Ulam_approximation_mapa{mapa}_Sij_eigenvals_N{Neff}_grilla{cx}N_K{K}_Nc{Nc}'
np.savez(flag+'.npz', e)
#%% plot eigenvalues
plt.figure()
plt.plot(e.real, e.imag, 'r*', ms=2, alpha=0.7, label='Tomi')
plt.xlabel(r'$\Re(\lambda)$')
plt.ylabel(r'$\Im(\lambda)$')
# plt.xlim(0.49,1.1)
# plt.ylim(-0.1,0.1)
plt.grid(True)

# x = w.real
# # extract imaginary part using numpy
# y = w.imag
  
# ############################################################################
# # fig1 = plt.figure(4)
# plt.scatter(x, y, s=5, alpha=0.8, label='Diego')

theta = np.linspace(0, 2*np.pi, 100)
   
r= 1

x = r*np.cos(theta)
y = r*np.sin(theta)
   
plt.plot(x,y,color='b', lw=1)
# plt.legend(loc='best')
# plt.savefig(f'Comparacion_zoom_Ulam_Diego_Neff{Neff}_Nx{Nx}_K{K}.png', dpi=100)
#%% plot eigenvalues for each N
Ns = 2**np.arange(5,8)
Ncs = [int(1e3), int(1e4), int(1e5)]
Nc = Ncs[1]

markers = ['o','*','s','^']

plt.figure(figsize=(10,10))

r= 1
theta = np.linspace(0, 2*np.pi, 100)

x = r*np.cos(theta)
y = r*np.sin(theta)
   
plt.plot(x,y,color='b', lw=1)

for i, Neff in enumerate(Ns):
    flag = f'Ulam_approximation_mapa{mapa}_Sij_eigenvals_N{Neff}_grilla{cx}N_K{K}_Nc{Nc}'
    archives = np.load(flag+'.npz')
    e = archives['arr_0']
    print(f'N={Neff} ', e.real[:10])
    plt.plot(e.real, e.imag, markers[i], ms=10, alpha=0.6, label=f'N={Neff}')
plt.xlabel(r'$\Re(\lambda)$')
plt.ylabel(r'$\Im(\lambda)$')
# plt.xlim(0.5,1)
# plt.ylim(-0.025,0.025)
plt.xlim(-1.1,1.1)
plt.ylim(-1.1,1.1)
plt.grid(True)
plt.legend(loc='best')
plt.savefig(flag+'.png', dpi=300)
#%%
# Copyright Dominique Delande
# Provided "as is" without any warranty
#
# Computes several "trajectories" of the standard map
# Initial conditions are generated randomly
# Prints the points in the (I,theta) plane (modulo 2*pi)
# Consecutive trajectories are separated by an empty line
# Output is printed in the "several_trajectories.dat" file
# Only points in a restricted area are actually printed. This makes
# it possible to zoom in a limited region of phase space

# from math import *
# import time
# import socket
# import random
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# from tqdm import tqdm

# dpi=2.0*pi
# # The following 6 lines for the primary plot
# # Execution time is insignificant
# K=0.97
# # Ks = np.linspace(0.5, 8.5, 6)#np.array([0.5, 3, 6, 8, 10, 12])#[0.25, 0.5, 2, 3, 6, 10]
# number_of_trajectories=100
# number_of_points=1000
# theta_min_for_plot=0.
# theta_max_for_plot=1.#dpi
# I_min_for_plot=0.
# I_max_for_plot=1.#dpi

# # The following 6 lines good for a secondary island
# # Execution time is insignificant
# # K=1.0
# # number_of_trajectories=200
# # number_of_points=1000
# # theta_min_for_plot=1.7
# # theta_max_for_plot=4.6
# # I_min_for_plot=2.5
# # I_max_for_plot=3.8

# # The following 6 lines good for a ternary island
# # Execution time: 28seconds
# # K=1.0
# # number_of_trajectories=1000
# # number_of_points=10000
# # theta_min_for_plot=4.05
# # theta_max_for_plot=4.35
# # I_min_for_plot=3.6
# # I_max_for_plot=3.75
# N = number_of_points*number_of_trajectories

# data_Ks = np.zeros((N, 2, len(Ks)))

        
# thetas, Is = [[], []]
# Keff = K/(2*np.pi)
# for i_traj in range(number_of_trajectories):
#   theta=random.uniform(0,1)
#   I=random.uniform(0,1)
#   for i in range(number_of_points):
#     if ((theta>=theta_min_for_plot)&(theta<=theta_max_for_plot)&(I>=I_min_for_plot)&(I<=I_max_for_plot)):
#       # print (theta,I)
#       thetas.append(theta)
#       Is.append(I)
#     I=(I+Keff*sin(dpi*theta))%1
#     theta=(theta+I)%1
#   # print(' ')
# data_Ks[:,0,k] = thetas
# data_Ks[:,1,k] = Is
