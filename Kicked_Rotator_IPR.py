#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 20:11:16 2022

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

# @jit(nopython=True, parallel=True, fastmath = True)
# def Evolucion_numpy(B_t, U, Udag):
#     res = Udag@B_t@U
#     return res

# @jit(nopython=True, parallel=True, fastmath = True)
# def O1_numpy_Tinf(A, B_t):
#     O1 = np.trace(B_t@A@B_t@A)
#     return O1

# @jit(nopython=True, parallel=True, fastmath = True)
# def C2_numpy_Tinf(A, B_t):
#     C2 = np.trace(B_t@A)
#     return C2

# def C2_Evolution_FFT_numpy_Tinf(time_lim, N, Ks, A, B, op = 'X'):
    
#     start_time = time.time() 
    
#     C2_Ks = np.zeros((len(Ks), time_lim), dtype=np.complex_)#[] # OTOC for each Ks
    
#     for j, K in tqdm(enumerate(Ks), desc='Primary loop'):
        
#         C2 = np.zeros((time_lim), dtype=np.complex_)#[]
        
#         # Distinct evolution for each operator X or P
#         U = UU(K, N, op)
#         Udag = U.H
#         # Calculo el OTOC para cada tiempo pero para un K fijo
#         for i in tqdm(range(time_lim), desc='Secondary loop'):
            
#             if i==0:
#                 B_t = B
#             else:
#                 # FFT for efficient evolution
#                 # diagonal evolution
#                 B_t = Evolucion_numpy(B_t, U, Udag)# U*B_t*U.dag()
                               

#             C_t = C2_numpy_Tinf(A, B_t)
            
#             C2[i] = C_t#OTOC.append(np.abs(C_t.data.toarray()))

#         C2_Ks[j,:] = C2
        
#     print(f"\nTOTAL --- {time.time() - start_time} seconds ---" )
#     flag = '2pC_FFT_with_Tinf_state'
#     return [C2_Ks, flag]

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

# def O1_Evolution_FFT_numpy_Tinf(time_lim, N, Ks, A, B, op = 'X'):
    
#     start_time = time.time() 
    
#     O1_Ks = np.zeros((len(Ks), time_lim), dtype=np.complex_)#[] # OTOC for each Ks
#     r_Ks = np.zeros((len(Ks)))
#     for j, K in tqdm(enumerate(Ks), desc='Primary loop'):
        
#         O1 = np.zeros((time_lim), dtype=np.complex_)#[]
        
#         # Distinct evolution for each operator X or P
#         U = UU(K, N, op)
#         Udag = U.H
        
#         r_normed = diagU_r(U)
        
#         # Calculo el OTOC para cada tiempo pero para un K fijo
#         for i in tqdm(range(time_lim), desc='Secondary loop'):
            
#             if i==0:
#                 B_t = B
#             else:
#                 # FFT for efficient evolution
#                 # diagonal evolution
#                 B_t = Evolucion_numpy(B_t, U, Udag)# U*B_t*U.dag()
                               
#             C_t = O1_numpy_Tinf(A, B_t)
            
#             O1[i] = C_t#OTOC.append(np.abs(C_t.data.toarray()))

#         O1_Ks[j,:] = O1
#         r_Ks[j] = r_normed
#     print(f"\nTOTAL --- {time.time() - start_time} seconds ---" )
#     flag = '4pC_FFT_with_Tinf_state'
#     return [O1_Ks, r_Ks, flag]

# def r_Ks(N, K):
     
#     start_time = time()
#     # let's try it

#     ######## U evolution ##########
#     U = UU(K, N)
    
#     # separate symmetries
#     start_time = time()
        
#     # #########################################################################
#     r_normed = diagU_r(U)
    
#     print(f"\n r-chaometer K = {K} --- {time() - start_time} seconds ---" )
#     return r_normed

@jit#(nopython=True, parallel=True, fastmath = True) 
def gaussian_state_numpy(N, n0, sigma, ket=True, nrm=1):
    # heff = 1/(2*np.pi*N)
    ans = [np.exp(-(i-n0)**2/(2*sigma**2)) for i in range(0, N)] # heff**2*
    I = np.identity(N)
    psi0 = sum([ans[i]*np.matrix(I[:,i]).T for i in range(0,N)])
    psi0 = psi0/np.linalg.norm(psi0)
    psi0 = np.sqrt(nrm)*psi0
    if ket:
        return psi0
    else:
        state0 = np.outer(psi0,psi0)
        state0 = np.matrix(state0, dtype=np.complex_)#/np.trace(state0)
        return state0
    
# def gaussian_basis(N):
#     heff = 1/(2*np.pi*N)
#     sigma = np.sqrt(heff)/2
#     paso = np.sqrt(heff)
#     cant = int(N/paso)
#     basis = np.zeros((N,cant))
#     center = 0
#     for i in range(cant):
#         state = gaussian_state_numpy(N, (i+1)*paso, sigma, nrm=heff)
#         basis[:,i] = state.T
#     return basis

def base_integrable(N, K=1):
    U = Qobj(UU(K, N))
    ev, evec = U.eigenstates()
    
    # C = np.column_stack([vec.full() for vec in evec])
    return [ev, evec]
    
    
def IPR(state):
    # state = state.full()
    coef4 = np.abs(state)**4
    IPR = 1/np.sum(coef4)
    return IPR

def expand_in_basis(state, basis):
    assert type(state) == type(basis[0]), 'r u sure this a Qobj?'
    coefs = np.array([(state.dag()*basis[i]).full() for i in range(len(basis))])
    return coefs

def normalize(array):
    return (array - min(array))/(max(array)-min(array))

def F(N,r,x,y,lim_suma=2):
    coefsnu = np.array([np.exp(-np.pi*N/r*(nu-x)**2-1j*2*np.pi*N*y*nu) for nu in range(-lim_suma, +lim_suma)])
    return np.sum(coefsnu)

def coherent_state(N, q0, p0, sigma, cuasiq=0):
    hbar = 1/(2*np.pi*N)
    # cte = ((4*np.pi*hbar)/(sigma**2))**(1/4)
    cte = (2/(N*sigma**2))**(1/4)
    # z = 1/np.sqrt(2)*(q0/sigma - 1j*sigma*p0)
    # lim_suma = 2#int(N/4)
    # zconj = z.conjugate()
    # print(zconj)
    state = np.zeros((N), dtype=np.complex_)
    for q in range(N):
        # print(q, -1/hbar*(1/2*zconj**2+1/2*(q**2/sigma**2)-np.sqrt(2)*zconj*q/sigma))
        Q = q/N
        # P = p/N
        # print(Q, -1/hbar*(1/2*zconj**2+1/2*(Q**2/sigma**2)-np.sqrt(2)*zconj*Q/sigma))
        # coefq = np.exp(-1/hbar*(1/2*zconj**2+1/2*(Q**2/sigma**2)-np.sqrt(2)*zconj*Q/sigma))#*cte
        coefq = np.exp(1j/hbar*p0*(Q-q0/2))#1/(4*hbar)*(q0**2/sigma**2+p0**2*sigma**2)+
        # print('coefq',coefq)
        # coefsnu = np.array([cte*np.exp(-1/hbar*(1/2*((nu/N)**2/sigma**2)+np.sqrt(2)*zconj*(nu/N)/sigma-Q*(nu/N)/(sigma**2))+1j*2*np.pi*cuasiq*(nu/N)) for nu in range(-lim_suma, +lim_suma)])
        # print('sum(coefsnu)',np.sum(coefsnu))
        coef = coefq*F(N, sigma**2, -np.abs(q0-Q), -np.abs(p0-cuasiq/N))# Q-q0, -p0)#
        state[q] = coef
    return Qobj(state/np.linalg.norm(state))

def base_gaussiana(N, sigma):
    hbar = 1/(2*np.pi*N)
    paso = np.sqrt(hbar)
    total = round(1/paso)
    # print('total',total)
    qs = [i*paso for i in range(total)]
    # print(np.array(qs)*N)
    ps = [i*paso for i in range(total)]
    basis = []
    cont=0
    for i,q in enumerate(qs):
        for j,p in enumerate(ps):
            state = coherent_state(N,q,p,sigma)
            # print(np.linalg.norm(state))
            if np.linalg.norm(state)==0:
                print('q0,p0',q*N,p*N)
                cont+=1
            basis.append(state)
    print(cont)
    return basis
#%%
N = 1000
hbar = 1/(2*np.pi*N)
# q0 = 0
# p0 = 2/10
sigma = 1#np.power(hbar,1/4)
# state = coherent_state(N, q0, p0, sigma)
# print(state, np.linalg.norm(state))

# _, basis = base_integrable(N, K=1)
start_time = time()
basis = base_gaussiana(N,sigma)
print('Duration: ', time()-start_time)

# #%%
# index = 100#int(len(basis)/2)

# vec = basis[index].full()
# vec = vec/np.linalg.norm(vec)

# x = np.arange(N)
# y = np.abs(vec)**2
# plt.plot(x, y, '.-')
#%%
# K = 5
# U = Qobj(UU(K, N))
# ev, evec = U.eigenstates()
# vec = evec[0]
# coefs = expand_in_basis(vec, basis)
# print(np.sum(np.abs(coefs))**2)
#%%

tipo_de_base = 'gaussiana'#'integrable'
Kpaso = .5
Ks = np.arange(1,10.1,Kpaso)#

IPR_means = np.zeros((len(Ks)))
IPR_skews = np.zeros((len(Ks)))

for j, K in tqdm(enumerate(Ks), desc=f'K loop total{len(Ks)}'):
    IPRs = np.zeros((N))    
    U = Qobj(UU(K, N))
    ev, evec = U.eigenstates()
        
    for i, vec in tqdm(enumerate(evec), desc=f'eigenstates loop total {N}'):
        coefs = expand_in_basis(vec, basis)
        if tipo_de_base=='gaussiana':
            coefs /= np.sqrt(np.sum(np.abs(coefs))**2)
            print(np.sum(np.abs(coefs))**2)
        aux = IPR(coefs)
        IPRs[i] = aux
    
    IPR_means[j] = np.mean(IPRs)
    IPR_skews[j] = skew(IPRs)
    
    plt.figure(figsize=(16,8))
    plt.title(f'K={K}')
    plt.hist(IPRs)
    plt.xlabel('IPR')
    # plt.xlim(0,300)
    plt.grid(True)
    plt.savefig(f'IPR_distribution_K{K}_N{N}_basis_integrable_wK1.png', dpi=80)
#%%
np.savez(f'IPR_mean_skew_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_N{N}_basis_integrable_wK1.npz', IPR_means = IPR_means, IPR_skews = IPR_skews, Ks = Ks)

archives = np.load(f'r_vs_K_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_N{N}.npz')
rs = archives['rs']

r_normed = normalize(rs)
y_mean = normalize(IPR_means)
y_skew = normalize(IPR_skews)

plt.figure(figsize=(16,8))
plt.plot(Ks, y_mean, 'r.-', label='mean')
# plt.plot(Ks, y_skew, 'b.-', label='skew')
plt.plot(Ks, r_normed, 'k.-', label='r')
# plt.vlines(0, 0, 1, ls='dashed', alpha=0.5)
plt.ylabel('IPR')
plt.xlabel('K')
# plt.xlim(4,10)
plt.grid(True)
plt.legend(loc='best')
plt.savefig(f'IPR_vs_Ks_N{N}_basis_'+tipo_de_base+'_wK1.png', dpi=80)