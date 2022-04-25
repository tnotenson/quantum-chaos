#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:54:35 2022

@author: tomasnotenson
"""

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from qutip import *
from time import time 
from tqdm import tqdm # customisable progressbar decorator for iterators
from cmath import phase

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

def r_Ks(N, K):
     
    start_time = time()
    # let's try it

    ######## U evolution ##########
    U = UU(K, N)
    
    # separate symmetries
    start_time = time()
        
    # #########################################################################
    r_normed = diagU_r(U)
    
    print(f"\n r-chaometer K = {K} --- {time() - start_time} seconds ---" )
    return r_normed

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
    
def gaussian_basis(N):
    heff = 1/(2*np.pi*N)
    sigma = np.sqrt(heff)/2
    paso = np.sqrt(heff)
    cant = int(N/paso)
    basis = np.zeros((N,cant))
    center = 0
    for i in range(cant):
        state = gaussian_state_numpy(N, (i+1)*paso, sigma, nrm=np.sqrt(heff))
        basis[:,i] = state.T
    return basis

#%%

# Define basis size        
N = 1000#2**10 # Tiene que ser par

# Define K values for the simulation

Kpaso = 0.2

Ks = np.arange(0,14,Kpaso)#np.array([2, 5, 8, 12, 15, 17, 19.74])#Kspelado*(4*np.pi**2) # K values np.array([Kpelado])

rs = np.zeros((len(Ks)))

for i,K in tqdm(enumerate(Ks), desc='Ks loop'):
    # r_normed_H, r_normed_U = r_thetas(N, B, J, theta, x, z, dt)
    r_normed = r_Ks(N, K)
    rs[i] = r_normed


# Define position and momentum values in the torus
# qs = np.arange(0, N) #take qs in [0;N) with qs integer

# t0 = time.time()
# Define Schwinger operators
# Us = fU(N, qs)
# Vs = fV(N)

# # # Define momentum and position operatorsÇ
# P = (Vs - Vs.dag())/(2j)
# X = (Us - Us.dag())/(2j)

# t1 = time.time()
# print(f'Tiempo operadores: {t1-t0}')

# Select operators for the out-of-time correlators

# A = P.data.toarray()
# B = P.data.toarray()

# operators = 'AP_BP'



# Define pure state
# t2 = time.time()
# sigma = 1/(2*np.pi*N)/10000
# state0 = gaussian_state(int(N/2), 0, sigma)
# state0 = gaussian_state(int(N/2), 0, sigma, ket=True)

# numpy states
# n0 = 0
# state0_numpy = gaussian_state_numpy(int(N/2), n0, sigma)
# state0_numpy = gaussian_state_numpy(int(N/2), 0, sigma, ket=True)
# t3 = time.time()
# print(f'Tiempo estado: {t3-t2}')
# Define time array
# time_lim = int(3e1+1) # number of kicks

# ### compute OTOC with O1 and O2 in the "Heisenberg picture"
# O1_Ks, r_Ks, flag = O1_Evolution_FFT_numpy_Tinf(time_lim, N, Ks, A, B, op ='X')

# O1 = np.abs(O1_Ks)
# # define file name

# file = flag+f'_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_basis_size{N}_time_lim{time_lim}'+operators+'.npz'#+'_evolucion_al_reves' _centro{n0}
# np.savez(file, O1=O1, r_Ks=r_Ks)


#%%
xstr = r'$K$'

inicio = 1

r = rs[inicio:]

r_normed = (r-min(r))/(max(r) - min(r))
# r_U_normed = rs_U#(rs_U-min(rs_U))/(max(rs_U) - min(rs_U))

x = Ks[inicio:]
y = r_normed

plt.figure(figsize=(16,10))
plt.plot(x,y, '^-r', ms=0.8, lw=0.8, label='H')
# plt.plot(zs,r_U_normed, '^-b', ms=0.8, lw=0.8, label='U')
# plt.plot(thetas,r_H_normed, '^-r', ms=0.8, lw=0.8, label='H')
# plt.plot(thetas,r_U_normed, '^-b', ms=0.8, lw=0.8, label='U')
# plt.xlabel(r'$\theta$')
plt.xlabel(xstr)
plt.ylabel(r'$r$ (chaometer) ')
# plt.ylim(-0.1,1.1)
plt.grid(True)
plt.legend(loc='best')

plt.savefig(f'r_vs_K_comparacion_paper_Emi_D{N}.png', dpi=80)