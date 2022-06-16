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

@jit(nopython=True, parallel=True, fastmath = True)#, cache=True)#(cache=True)#(nopython=True)
def FF(N, x1 = 0, p1 = 0):
    FF = np.zeros((N,N),dtype=np.complex_)
    for i in range(N):
        for j in range(N):
            FF[i,j]=np.exp(-1j*2*np.pi*(i + x1)*(j + p1)/N)*np.sqrt(1/N)
    return FF

@jit#(nopython=True, parallel=True, fastmath = True)#, cache=True)#(nopython=True)
def UU(K, N, op = 'P', x1 = 0, p1 = 0):
    UU = np.zeros((N,N),dtype=np.complex_)
    MM = np.zeros((N,N),dtype=np.complex_)
    F = FF(N)
    Keff = K/(4*np.pi**2)
    
    if op == 'X':
        
        for i in range(N):
                for j in range(N):
                    UU[i,j]=F[i,j]*np.exp(-1j*2*np.pi*N*Keff*np.cos(2*np.pi*(i + x1)/N))#/N
                    MM[i,j]=np.conjugate(F[j,i])*np.exp(-1j*np.pi*(i + p1)**2/N)
    elif op == 'P'  :
        for i in range(N):
                for j in range(N):
                    UU[i,j]=np.exp(-1j*2*np.pi*N*Keff*np.cos(2*np.pi*(i + x1)/N))*np.conjugate(F[j,i])
                    MM[i,j]=np.exp(-1j*np.pi*(i + p1)**2/N)*F[i,j]
    
    U = MM@UU
    U = np.matrix(U)
    return U

def fV(N):
    V = 0
    for j in range(N):
        V += basis(N,(j+1)%N)*basis(N,j).dag()
    return V

@jit#(nopython=True, parallel=True, fastmath = True)
def fV_numpy(N):
    V = np.zeros((N,N), dtype=np.complex_)
    I = np.identity(N)
    for j in range(N):
        V += np.outer(I[:,(j+1)%N],I[j,:])
    V = np.matrix(V)
    return V #np.matrix(V)
# print(V(N))

def fU(N, qs):
    U = 0
    tau = np.exp(2j*np.pi/N)
    for j in range(N):
        U += basis(N,j)*basis(N,j).dag()*tau**(qs[j])
    return U

@jit#(nopython=True, parallel=True, fastmath = True)
def fU_numpy(N, qs):
    U = np.zeros((N,N), dtype=np.complex_)
    tau = np.exp(2j*np.pi/N)
    I = np.identity(N)
    for j in range(N):
        U += np.outer(I[:,j],I[j,:])*tau**(qs[j])
    U = np.matrix(U)
    return U #np.matrix(U)

# Calcultes r parameter in the 10% center of the energy "ener" spectrum. If plotadjusted = True, returns the magnitude adjusted to Poisson = 0 or WD = 1
def r_chaometer(ener,plotadjusted):
    ra = np.zeros(len(ener)-2)
    #center = int(0.1*len(ener))
    #delter = int(0.05*len(ener))
    for ti in range(len(ener)-2):
        ra[ti] = (ener[ti+2]-ener[ti+1])/(ener[ti+1]-ener[ti])
        ra[ti] = min(ra[ti],1.0/ra[ti])
    ra = np.mean(ra)
    if plotadjusted == True:
        ra = (ra -0.3863) / (-0.3863+0.5307)
    return ra

def diagU_r(U_sub):
    ener = np.linalg.eigvals(U_sub)
    # ener = np.sort(ener)
    fases = [phase(en) for en in ener]
    fases = np.sort(fases)
    # histo_level_spacing(fases)
    r_normed = r_chaometer(fases, plotadjusted=True) 
    return r_normed

# def expand_in_basis(state, base, formato='qutip'):
#     if formato=='qutip':
#         # print(type(state), type(base[0])) # chequeo los types
#         assert type(state) == type(base[0]), 'r u sure this is a Qobj?' # aviso cuando no son Qobj
#         # prueba = (base[0].dag()*state).full()[0][0]
#         # print(prueba, prueba.shape)
#         coefs = np.array([(base[i].dag()*state).full()[0][0] for i in range(len(base))], dtype=np.complex_) # coeficientes del estado a expandir en la base elegida
#         norm = np.linalg.norm(coefs) # norma del estado en esa base
#         print(coefs.shape)
#         res = Qobj(coefs)
#         return res
#     elif formato=='numpy':
#         coefs = np.array([np.sum(np.conj(base[i])*state) for i in range(len(base))], dtype=np.complex_) # coeficientes del estado a expandir en la base elegida
#         print(coefs.shape)
#         # print('norma expansion', norm) # chequeo las normas
#         return coefs#/norm # devuelvo el estado expandido en la base

def IPR(state):
    if np.linalg.norm(state) != 1:
        print(np.linalg.norm(state))
    # state = state.full()
    coef4 = np.abs(state)**4 # calculo cada término del denominador del IPR
    norma = np.abs(state)**2
    IPR = np.sum(norma)**2/np.sum(coef4) # calculo el IPR
    return IPR # devuelvo el IPR

def normalize(array):# normalizo valores entre 0 y 1
    return (array - min(array))/(max(array)-min(array)) # devuelvo la variable normalizada

def coherent_state_Augusto(N, q0, p0, lim_suma=4, formato='qutip'):#, cuasiq=0
    state = np.zeros((N), dtype=np.complex_) # creo el estado como array de ceros 
    # cte = np.power(2/N,4) # cte de norm. 1
    # coefq = np.exp(np.pi/(2*N)*(q0**2+p0**2)) # cte de norm. 2
    for q in range(N): #calculo el coeficiente para cada q
        coefsnu = np.zeros((2*lim_suma+1), dtype=np.complex_) # creo un array de ceros
        for i,nu in enumerate(np.arange(-lim_suma,lim_suma+1)):
            # print('q',q,', nu',nu)
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

def base_coherente_Augusto(N):
    Nstates = np.int32(N/2)
    base = np.zeros((N,Nstates**2), dtype=np.complex_)
    paso = N/Nstates
    for q in tqdm(range(Nstates), desc='q coherent basis'):
        for p in range(Nstates):
            # indice
            i=q+p*Nstates
            #centros

            q0=q*paso
            p0=p*paso
            # estado coherente
            # print(i, 'de ',Nstates**2)
            b = coherent_state_Augusto(N, q0, p0, lim_suma=4)
            base[:,i] = np.array(b).flatten()
    return base

def base_integrable(N, K=0):
    U = Qobj(UU(K, N))
    e, evec = U.eigenstates()
    return evec

# def base_gaussiana(N, sigma=1, formato='qutip'):
#     # hbar = 1/(2*np.pi*N)
#     # paso = N/2#np.sqrt(N)
#     total = int(N/2)#round(N/paso)#round(np.sqrt(N))#
#     # print(total)
#     # print('total',total)
#     qs = np.linspace(0,N-1,total)#[i*paso for i in range(total)]
#     # print(np.array(qs))
#     ps = np.linspace(0,N-1,total)#[i*paso for i in range(total)]
#     base = []
#     cont=0
#     for i,q in enumerate(qs):
#         for j,p in enumerate(ps):
#             state = coherent_state_Augusto(N,q,p,formato=formato)
#             # print(f'norma base gaussiana {q},{p}', np.linalg.norm(state))
#             if np.linalg.norm(state)==0:
#                 print('q0,p0',q,p)
#                 cont+=1
#             base.append(state)
#     print('cant nulos', cont)
#     return base
#%%

# N = 2**3 # exponente par
# sigma = 1

# formato = 'qutip'
# # formato = 'numpy'

# start_time = time()
# base = base_gaussiana(N,sigma,formato=formato)
# total = len(base)
# print('Duration: ', time()-start_time)

# identi = qeye(N)

# # Calculo la ide
# if formato == 'qutip':
#     id_cs = np.sum(np.array([base[i]*base[i].dag() for i in range(len(base))]), axis=0)
    
# elif formato == 'numpy':
#     id_cs = np.sum(np.array([np.outer(base[i],base[i].conj()) for i in range(total)]), axis=0)
    
# # print(id_cs)

# ### base de estados random
# np.random.seed(9)
# evec = np.random.rand(N,N)

# index1 = int(N/2) # elijo el índice del estado de referencia contra el cual bracketeo
# index2 = 2

# if formato == 'qutip':
#     vec1 = Qobj(evec[index1]).unit()#evec[index1]#
#     vec2 = Qobj(evec[index2]).unit()#evec[index2]#
# elif formato == 'numpy':
#     vec1 = evec[index1]#
#     vec1 /= np.linalg.norm(vec1)
#     vec2 = evec[index2]#
#     vec2 /= np.linalg.norm(vec2)
    
# coefs1 = expand_in_basis(vec1, base, formato=formato)
# if formato == 'qutip':
#     coefs1dag = coefs1.dag()#np.conj(coefs1)
# elif formato == 'numpy':
#     coefs1dag = np.conj(coefs1)
    
    
# coefs2 = expand_in_basis(vec2, base, formato=formato)
# if formato == 'qutip':
#     coefs2dag = coefs2.dag()#np.conj(coefs2)
# elif formato == 'numpy':
#     coefs2dag = np.conj(coefs2)

# if formato == 'qutip':
#     overlap_id = vec1.dag() * id_cs * vec2
#     overlap = vec1.dag() * vec2
#     overlap_cs = coefs1dag * coefs2
# elif formato == 'numpy':
#     overlap_id = np.matrix(vec1) @ id_cs @ np.matrix(vec2).H
#     overlap = np.matrix(vec1) @ np.matrix(vec2).H
#     overlap_cs = np.sum(coefs1dag * coefs2)
# print(overlap_id, overlap, overlap_cs)


#%%
nqubits = 7;
N = 200#2**nqubits#11
# hbar = 1/(2*np.pi*N)
Nstates= np.int32(N/2)  #% numero de estados coherentes de la base
paso = N/Nstates

# armo la base coherente
start_time = time()
base = base_coherente_Augusto(N)
# base = base_integrable(N)
print('Duration: ', time()-start_time)

# seed
np.random.seed(0)

# defino 2 estados complejos random
a=np.random.random(N)+np.random.random(N)*1j
norm=np.linalg.norm(a)
a = Qobj(a/norm)

c=np.random.random(N)+np.random.random(N)*1j
norm=np.linalg.norm(c)
c = Qobj(c/norm)

# calculo el overlap entre a y b (directamente)
over1 = a.overlap(c)

# defino la identidad
Iden = np.zeros([N,N], dtype=np.complex_)
Iden = Qobj(Iden)

est_a = np.zeros(Nstates**2, dtype=np.complex_)
est_c = np.zeros(Nstates**2, dtype=np.complex_)

for q in tqdm(range(Nstates), desc='q loop'):
    for p in range(Nstates):
        # indice
        i=q+p*Nstates
        #centros

        q0=q*paso
        p0=p*paso
        # estado coherente
        # print(i, 'de ',Nstates**2)
        b = Qobj(base[:,i])
        
        # expando los estados a y b en la base de estados coherentes
        # overlap entre el estado a y el coherente centrado en q0 y p0
        est_a[i] = c.overlap(a)
        est_c[i] = c.overlap(c)
        
        # sumo el proyector asociado a ese estado coherente
        ident1 = b.proj()
        Iden = Iden+ident1
        
# convierto arrays en Qobj
est_a = Qobj(est_a)
est_c = Qobj(est_c)

# calculo el overlap en la base de estados coherentes
over3 = est_a.overlap(est_c)

# calculo las normas y el overlap tomando los elementos de matriz de la...
# ... identidad calculada a partir de los proyectores de la base coherente
nor1=Iden.matrix_element(a,a)
nor2=Iden.matrix_element(c,c)
over=Iden.matrix_element(a,c)

# imprimo resultados
print(nor1)
print(nor2/nor1)
print(over/nor1)
print(over1)
print(over3/nor1)
#%%
Kpaso = .5
Ks = np.arange(0,10.1,Kpaso)#

norma = np.sqrt(nor1)
IPR_means = np.zeros((len(Ks)))
rs = np.zeros((len(Ks)))

    
for k,K in tqdm(enumerate(Ks), desc='K loop'):
    IPRs = np.zeros((N))    
    U = Qobj(UU(K, N))
    ev, evec = U.eigenstates()
    r = diagU_r(U)
    rs[k] = r
    
    for j,vec in tqdm(enumerate(evec), desc='vec loop'):
        est_vec = np.zeros(Nstates**2, dtype=np.complex_)
        for q in range(Nstates):
            for p in range(Nstates):
                # indice
                i=q+p*Nstates
                #centros

                q0=q*paso
                p0=p*paso
                # estado coherente
                # print(i, 'de ',Nstates**2)
                b = Qobj(base[:,i])
                
                # expando los estados a y b en la base de estados coherentes
                # overlap entre el estado a y el coherente centrado en q0 y p0
                est_vec[i] = b.overlap(vec) #Iden.matrix_element(b,vec)#
                
        aux = IPR(est_vec/norma)
        IPRs[j] = aux
       
    IPR_means[k] = np.mean(IPRs)
np.savez(f'IPR_vs_Ks_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_N{N}_coherent_basis_grid{Nstates}x{Nstates}.npz', IPR_means = IPR_means, rs = rs)#integrable_K0
    #%% base integrable
# nqubits = 5;
# N = 2**nqubits#11

# # armo la base coherente
# start_time = time()
# base = base_integrable(N)
# print('Duration: ', time()-start_time)

# Kpaso = .5
# Ks = np.arange(0,10.1,Kpaso)#

# IPR_means = np.zeros((len(Ks)))
# rs = np.zeros((len(Ks)))

# for k,K in tqdm(enumerate(Ks), desc='K loop'):
#     IPRs = np.zeros((N))    
#     U = Qobj(UU(K, N))
#     ev, evec = U.eigenstates()
#     r = diagU_r(U)
#     rs[k] = r
    
#     for j,vec in tqdm(enumerate(evec), desc='vec loop'):
#         est_vec = np.zeros(N, dtype=np.complex_)
#         for i in range(N):
            
#             b = base[i]
            
#             # expando los estados a y b en la base de estados coherentes
#             # overlap entre el estado a y el coherente centrado en q0 y p0
#             est_vec[i] = b.overlap(vec)

#         aux = IPR(est_vec)
#         IPRs[j] = aux
       
#     IPR_means[k] = np.mean(IPRs)
# np.savez(f'IPR_vs_Ks_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_N{N}_integrable_K0.npz', IPR_means = IPR_means)#

#%%
# IPR_integrable = normalize(IPR_means)

# archives = np.load(f'IPR_vs_Ks_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_N{N}_coherent_basis_grid{Nstates}x{Nstates}.npz')
# IPR_means = archives['IPR_means']

# r_normed = normalize(rs)
# IPR_normed = normalize(IPR_means)

# plt.figure(figsize=(16,8))
# plt.title(f'N={N}')
# plt.plot(Ks, IPR_normed, 'r.-', label='coherentes')
# plt.plot(Ks, IPR_integrable, 'b.-', label='integrable')
# plt.plot(Ks, r_normed, 'k.-', label='r')
# # plt.vlines(0, 0, 1, ls='dashed', alpha=0.5)
# plt.ylabel('IPR')
# plt.xlabel('K')
# plt.xlim(-0.05,10.05)
# plt.ylim(-0.001,1.001)
# plt.grid(True)
# plt.legend(loc='best')
# plt.savefig(f'IPR_vs_Ks_N{N}_integrable_and_coherent_basis.png', dpi=80)