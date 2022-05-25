#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 14:47:26 2022

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
import time 
plt.rcParams.update({
"text.usetex": True,
"font.family": "sans-serif",
"font.sans-serif": ["Helvetica"], "font.size": 24})

def normalize(array):
    return (array - min(array))/(max(array)-min(array))

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

@jit(nopython=True, parallel=True, fastmath = True)
def Evolucion_numpy(B_t, U, Udag):
    res = Udag@B_t@U
    return res

@jit(nopython=True, parallel=True, fastmath = True)
def O1_numpy_Tinf(A, B_t):
    O1 = np.trace(B_t@A@B_t@A)
    return O1

@jit(nopython=True, parallel=True, fastmath = True)
def O2_numpy_Tinf(A, B_t):
    O2 = np.trace(B_t@B_t@A@A)
    return O2

@jit(nopython=True, parallel=True, fastmath = True)
def C2_numpy_Tinf(A, B_t):
    C2 = np.trace(B_t@A)
    return C2

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

def O1_Evolution_FFT_numpy_Tinf(time_lim, N, Ks, A, B, op = 'X', r=True):
    
    start_time = time.time() 
    
    O1_Ks = np.zeros((len(Ks), time_lim), dtype=np.complex_)#[] # OTOC for each Ks
    if r:
        r_Ks = np.zeros((len(Ks)))
    for j, K in tqdm(enumerate(Ks), desc='Primary loop'):
        
        O1 = np.zeros((time_lim), dtype=np.complex_)#[]
        
        # Distinct evolution for each operator X or P
        U = UU(K, N, op)
        Udag = U.H
        if r:
            r_normed = diagU_r(U)
        
        # Calculo el OTOC para cada tiempo pero para un K fijo
        for i in tqdm(range(time_lim), desc='Secondary loop'):
            
            if i==0:
                B_t = B
            else:
                # FFT for efficient evolution
                # diagonal evolution
                B_t = Evolucion_numpy(B_t, U, Udag)# U*B_t*U.dag()
                               
            C_t = O1_numpy_Tinf(A, B_t)
            
            O1[i] = C_t#OTOC.append(np.abs(C_t.data.toarray()))

        O1_Ks[j,:] = O1
        if r:
            r_Ks[j] = r_normed
    print(f"\nTOTAL --- {time.time() - start_time} seconds ---" )
    flag = '4pC_FFT_with_Tinf_state'
    if r:
        return [O1_Ks, r_Ks, flag]
    return [O1_Ks, flag]
#%%
# Define basis size        
N = 2**7 # Tiene que ser par

# Define position and momentum values in the torus
qs = np.arange(0, N) #take qs in [0;N) with qs integer

t0 = time.time()
# Define Schwinger operators
Us = fU(N, qs)
Vs = fV(N)

# Define Schwinger operators
# Us_numpy = fU_numpy(N, qs)
# Vs_numpy = fV_numpy(N)

# # # Define momentum and position operatorsÇ
P = (Vs - Vs.dag())/(2j)
X = (Us - Us.dag())/(2j)

# P_numpy = (Vs_numpy - Vs_numpy.H)/2j
# X_numpy = (Us_numpy - Us_numpy.H)/2j
t1 = time.time()
print(f'Tiempo operadores: {t1-t0}')

# Select operators for the out-of-time correlators
# A = P
# B = X

A = X.data.toarray()
B = P.data.toarray()

opA = 'X'
opB = 'P'

operators = 'A'+opA+'_B'+opB

# Define K values for the simulation

# Kspelado = np.array([0.5])#(np.arange(0.3, 0.51, 0.2))#

Kpaso = .05
Ks = np.arange(0,10.1,Kpaso)##np.array([2, 5, 8, 12, 15, 17, 19.74])#Kspelado*(4*np.pi**2) # K values np.array([Kpelado])

# Define pure state
t2 = time.time()
# sigma = 1/(2*np.pi*N)/10000
# state0 = gaussian_state(int(N/2), 0, sigma)
# state0 = gaussian_state(int(N/2), 0, sigma, ket=True)

# numpy states
# n0 = 0
# state0_numpy = gaussian_state_numpy(int(N/2), n0, sigma)
# state0_numpy = gaussian_state_numpy(int(N/2), 0, sigma, ket=True)
t3 = time.time()
print(f'Tiempo estado: {t3-t2}')
# Define time array
time_lim = int(6e3) # number of kicks

# ### compute OTOC with O1 and O2 in the "Heisenberg picture"
O1_Ks, flag = O1_Evolution_FFT_numpy_Tinf(time_lim, N, Ks, A, B, op ='X', r=False)

O1 = np.abs(O1_Ks)/N
# # define file name
#%%
index = -3

y1 = O1[index]

times = np.arange(time_lim)

plt.figure(figsize=(16,8))
plt.title(f'O1 a tiempos largos. A={opA}, B={opB}')
plt.plot(times, y1, '-b', lw=1.5, label=f'K={Ks[index]}')
# plt.plot(Kx, r_normed, '-r', lw=1.5, label='r')
# plt.plot(Ks, 1-r_one, '-g', lw=1.5, label='1-r paridad +1')
plt.xlabel(r'$t$')
plt.ylabel(r'$O_1(t)$')
# plt.ylim(0,1)
# plt.xlim(-0.2,max(times)+0.2)
plt.grid(True)
plt.legend(loc = 'best')
plt.savefig(f'O1_vs_time_K{Ks[index]}_time_lim{time_lim}_basis_size{N}'+operators+'.png', dpi=80)
#%%
from scipy.fft import fft, ifft, fftfreq

t_sat = 500
M = len(y1[t_sat:])
O1_fourier = fft(y1[t_sat:])
xf = fftfreq(M, t_sat)[:M//2]
yf = 2.0/M * np.abs(O1_fourier[0:M//2])


plt.figure(figsize=(16,8))
plt.title(f'O1 a tiempos largos. A={opA}, B={opB}')
plt.plot(xf, yf, '-b', lw=1.5, label=f'K={Ks[index]}')
# plt.plot(Kx, r_normed, '-r', lw=1.5, label='r')
# plt.plot(Ks, 1-r_one, '-g', lw=1.5, label='1-r paridad +1')
plt.xlabel(r'$\omega$')
plt.ylabel(r'$O_1(\omega)$')
# plt.ylim(0,1)
# plt.xlim(-0.2,max(times)+0.2)
plt.grid(True)
plt.legend(loc = 'best')
plt.savefig(f'O1_vs_freq_from_t_sat{t_sat}_K{Ks[index]}_time_lim{time_lim}_basis_size{N}'+operators+'.png', dpi=80)

#%%

def IPR_normed(vec):
    PR = np.sum(np.abs(vec)**2)**2
    nrm2 = np.sum(np.abs(vec))**2
    IPR = nrm2/PR
    return IPR

IPRs = np.zeros((len(Ks)))

for k in range(len(Ks)):
    y1 = O1[k]
    O1_fourier = fft(y1[t_sat:])
    xf = fftfreq(M, t_sat)[:M//2]
    yf = 2.0/M * np.abs(O1_fourier[0:M//2])
    IPRs[k] = IPR_normed(yf)

np.savez(f'IPR_O1_from_t_sat{t_sat}_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_time_lim{time_lim}_basis_size{N}'+operators+'.npz',IPRs=IPRs)

y_IPR = normalize(IPRs)

plt.figure(figsize=(16,8))
plt.title(f'IPR O1. A={opA}, B={opB}')
plt.plot(Ks, y_IPR, '-b', lw=1.5)
# plt.plot(Kx, r_normed, '-r', lw=1.5, label='r')
# plt.plot(Ks, 1-r_one, '-g', lw=1.5, label='1-r paridad +1')
plt.xlabel(r'$K$')
plt.ylabel('IPR')
# plt.ylim(0,1)
# plt.xlim(-0.2,max(times)+0.2)
plt.grid(True)
# plt.legend(loc = 'best')
plt.savefig(f'IPR_O1_from_t_sat{t_sat}_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_time_lim{time_lim}_basis_size{N}'+operators+'.png', dpi=80)

