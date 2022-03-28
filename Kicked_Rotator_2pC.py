#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:09:09 2022

@author: tomasnotenson
"""

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from qutip import *
import time
from tqdm import tqdm # customisable progressbar decorator for iterators

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
def C2_numpy_Tinf(A, B_t):
    C2 = np.trace(B_t@A)
    return C2

def C2_Evolution_FFT_numpy_Tinf(time_lim, N, Ks, A, B, op = 'X'):
    
    start_time = time.time() 
    
    C2_Ks = np.zeros((len(Ks), time_lim), dtype=np.complex_)#[] # OTOC for each Ks
    
    for j, K in tqdm(enumerate(Ks), desc='Primary loop'):
        
        C2 = np.zeros((time_lim), dtype=np.complex_)#[]
        
        # Distinct evolution for each operator X or P
        U = UU(K, N, op)
        Udag = U.H
        # Calculo el OTOC para cada tiempo pero para un K fijo
        for i in tqdm(range(time_lim), desc='Secondary loop'):
            
            if i==0:
                B_t = B
            else:
                # FFT for efficient evolution
                # diagonal evolution
                B_t = Evolucion_numpy(B_t, U, Udag)# U*B_t*U.dag()
                               

            C_t = C2_numpy_Tinf(A, B_t)
            
            C2[i] = C_t#OTOC.append(np.abs(C_t.data.toarray()))

        C2_Ks[j,:] = C2
        
    print(f"\nTOTAL --- {time.time() - start_time} seconds ---" )
    flag = '2pC_FFT_with_Tinf_state'
    return [C2_Ks, flag]

#%%
# Define basis size        
N = 2**11 # Tiene que ser par

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

A = P.data.toarray()
B = P.data.toarray()

operators = 'AP_BP'

# Define K values for the simulation

# Kspelado = np.array([0.5])#(np.arange(0.3, 0.51, 0.2))#

Ks = np.array([19.74])#Kspelado*(4*np.pi**2) # K values np.array([Kpelado])

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
time_lim = int(5e1) # number of kicks

### compute OTOC with O1 and O2 in the "Heisenberg picture"
C2_Ks, flag = C2_Evolution_FFT_numpy_Tinf(time_lim, N, Ks, A, B, op ='X')

C2 = np.abs(C2_Ks[0])/N
# define file name
file = flag+f'_k{Ks}_basis_size{N}_time_lim{time_lim}'+operators+'.npz'#+'_evolucion_al_reves' _centro{n0}

#%%
K = Ks[0]

times = np.arange(time_lim)

Cs = C2[0]

y_C2_U = np.log10(C2)
yfit = np.log(C2)

tmin = 4
tmax = 12

xs = times[tmin:tmax]
yp = yfit[tmin:tmax]

coefp = np.polyfit(xs,yp,1)
poly1d_fn_p = np.poly1d(coefp) #[b,m]

print(r'$C_2$',poly1d_fn_p[1])
plt.plot(times,y_C2_U, 'o-g', ms=1, lw=2, label='$N=$'+f'{N}', alpha=0.8)
# plt.plot(times, 1/4*np.exp(-times/6)), '-.k', lw=1, label='0.25exp(-t/6)')
# plt.text(20, -1, r'$m_2=$'+f'{poly1d_fn_p[1].real:.2}',
#         verticalalignment='bottom', horizontalalignment='right',
#         color='red', fontsize=15)
plt.ylim(-5,0)
plt.ylabel(r'$log_{10}(C(t))$')
plt.xlabel(r'$t$')

plt.xlim(-0,20)
plt.grid()
plt.legend(loc='best')
# opi = str(int(N/2))
opA = 'X'#+opi
opB = opA#'X'
operators = '_A'+opA+'_B'+opB
flag = '2p_KI_with_Tinf_state_'
plt.savefig(flag+f'_time_lim{time_lim}_K{K:.2f}_basis_size{N}'+operators+'.png', dpi=80)