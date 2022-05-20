#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 18:58:38 2022

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

def expand_in_basis(state, base, formato='qutip'):
    if formato=='qutip':
        # print(type(state), type(base[0])) # chequeo los types
        assert type(state) == type(base[0]), 'r u sure this is a Qobj?' # aviso cuando no son Qobj
        # prueba = (base[0].dag()*state).full()[0][0]
        # print(prueba, prueba.shape)
        coefs = np.array([(base[i].dag()*state).full()[0][0] for i in range(len(base))], dtype=np.complex_) # coeficientes del estado a expandir en la base elegida
        norm = np.linalg.norm(coefs) # norma del estado en esa base
        print(coefs.shape)
        res = Qobj(coefs)
        return res
    elif formato=='numpy':
        coefs = np.array([np.sum(np.conj(base[i])*state) for i in range(len(base))], dtype=np.complex_) # coeficientes del estado a expandir en la base elegida
        print(coefs.shape)
        # print('norma expansion', norm) # chequeo las normas
        return coefs#/norm # devuelvo el estado expandido en la base

def normalize(array):# normalizo valores entre 0 y 1
    return (array - min(array))/(max(array)-min(array)) # devuelvo la variable normalizada

def coherent_state_Augusto(N, q0, p0, lim_suma=4, formato='qutip'):#, cuasiq=0
    state = np.zeros((N), dtype=np.complex_) # creo el estado como array de ceros 
    # cte = np.power(2/N,4) # cte de norm. 1
    # coefq = np.exp(np.pi/(2*N)*(q0**2+p0**2)) # cte de norm. 2
    for q in range(N): #calculo el coeficiente para cada q
        coefsnu = np.zeros((2*lim_suma+1), dtype=np.complex_) # creo un array de ceros
        for i,nu in enumerate(np.arange(-lim_suma,lim_suma+1)):
            coefnu = np.exp(-np.pi/N*(nu*N-(q0-q))**2-1j*2*np.pi/N*p0*(nu*N-q+q0/2)) 
            coefsnu[i] = coefnu # lo voy llenando con los términos de la sumatoria
        coef = np.sum(coefsnu) # hago la sumatoria
        state[q] = coef # agrego el coeficiente para el q fijo
    nrm = np.linalg.norm(state) # calculo la norma
    # print('norm state', nrm) # chequeo la norma del estado
    if formato=='qutip':
        return Qobj(state/nrm) # devuelvo el estado normalizado como Qobj
    elif formato=='numpy':
        return state/nrm # devuelvo el estado normalizado como numpy darray

def base_gaussiana(N, sigma=1, formato='qutip'):
    # hbar = 1/(2*np.pi*N)
    # paso = N/2#np.sqrt(N)
    total = int(N/2)#round(N/paso)#round(np.sqrt(N))#
    # print(total)
    # print('total',total)
    qs = np.linspace(0,N-1,total)#[i*paso for i in range(total)]
    # print(np.array(qs))
    ps = np.linspace(0,N-1,total)#[i*paso for i in range(total)]
    base = []
    cont=0
    for i,q in enumerate(qs):
        for j,p in enumerate(ps):
            state = coherent_state_Augusto(N,q,p,formato=formato)
            # print(f'norma base gaussiana {q},{p}', np.linalg.norm(state))
            if np.linalg.norm(state)==0:
                print('q0,p0',q,p)
                cont+=1
            base.append(state)
    print('cant nulos', cont)
    return base
#%%

N = 2**3 # exponente par
sigma = 1

formato = 'qutip'
# formato = 'numpy'

start_time = time()
base = base_gaussiana(N,sigma,formato=formato)
total = len(base)
print('Duration: ', time()-start_time)

identi = qeye(N)

# Calculo la ide
if formato == 'qutip':
    id_cs = np.sum(np.array([base[i]*base[i].dag() for i in range(len(base))]), axis=0)
    
elif formato == 'numpy':
    id_cs = np.sum(np.array([np.outer(base[i],base[i].conj()) for i in range(total)]), axis=0)
    
# print(id_cs)

### base de estados random
np.random.seed(9)
evec = np.random.rand(N,N)

index1 = int(N/2) # elijo el índice del estado de referencia contra el cual bracketeo
index2 = index1

if formato == 'qutip':
    vec1 = Qobj(evec[index1]).unit()#evec[index1]#
    vec2 = Qobj(evec[index2]).unit()#evec[index2]#
elif formato == 'numpy':
    vec1 = evec[index1]#
    vec1 /= np.linalg.norm(vec1)
    vec2 = evec[index2]#
    vec2 /= np.linalg.norm(vec2)
    
coefs1 = expand_in_basis(vec1, base, formato=formato)
if formato == 'qutip':
    coefs1dag = coefs1.dag()#np.conj(coefs1)
elif formato == 'numpy':
    coefs1dag = np.conj(coefs1)
    
    
coefs2 = expand_in_basis(vec2, base, formato=formato)
if formato == 'qutip':
    coefs2dag = coefs2.dag()#np.conj(coefs2)
elif formato == 'numpy':
    coefs2dag = np.conj(coefs2)

if formato == 'qutip':
    overlap_id = vec1.dag() * id_cs * vec2
    overlap = vec1.dag() * vec2
    overlap_cs = coefs1dag * coefs2
elif formato == 'numpy':
    overlap_id = np.matrix(vec1) @ id_cs @ np.matrix(vec2).H
    overlap = np.matrix(vec1) @ np.matrix(vec2).H
    overlap_cs = np.sum(coefs1dag * coefs2)
print(overlap_id, overlap, overlap_cs)