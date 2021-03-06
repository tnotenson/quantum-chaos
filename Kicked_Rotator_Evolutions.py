#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 20:55:04 2021

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

def gaussian_state(N, n0, sigma, ket=False):
    # heff = 1/(2*np.pi*N)
    ans = [np.exp(-(i-n0)**2/(2*sigma**2)) for i in range(-N, N)] # heff**2*
    psi0 = sum([ans[i]*basis(2*N,i) for i in range(0,2*N)])
    if ket:
        return psi0.unit()
    else:
        # state0 = psi0*psi0.dag()
        state0 = ket2dm(psi0)
        return state0.unit()

@jit#(nopython=True, parallel=True, fastmath = True) 
def gaussian_state_numpy(N, n0, sigma, ket=False):
    # heff = 1/(2*np.pi*N)
    ans = [np.exp(-(i-n0)**2/(2*sigma**2)) for i in range(-N, N)] # heff**2*
    I = np.identity(2*N)
    psi0 = sum([ans[i]*np.matrix(I[:,i]).T for i in range(0,2*N)])
    psi0 = psi0/np.linalg.norm(psi0)
    if ket:
        return psi0
    else:
        state0 = np.outer(psi0,psi0)
        state0 = np.matrix(state0, dtype=np.complex_)#/np.trace(state0)
        return state0
@jit 
def momentum_state_numpy(N, n0, ket=False):
    n0 += int(N/2)
    I = np.identity(2*N)
    psi0 = np.matrix(I[:,n0]).T
    if ket:
        psi0 = psi0/np.linalg.norm(psi0)
        return psi0
    else:
        state0 = np.outer(psi0,psi0)
        state0 = np.matrix(state0/np.trace(state0), dtype=np.complex_)
        return state0


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
    O2 = np.trace((B_t@B_t)@(A@A))
    return O2

@jit(nopython=True, parallel=True, fastmath = True)
def O1_numpy_state(state, A, B_t):
    O1 = np.trace(state@B_t@A@B_t@A)
    return O1

@jit(nopython=True, parallel=True, fastmath = True)
def O2_numpy_state(state, A, B_t):
    O2 = np.trace(state@(B_t@B_t)@(A@A))
    return O2

@jit(nopython=True, parallel=True, fastmath = True)
def O1_numpy_ket(state, statedag, A, B, U, Udag, i):
    O1 = statedag@Udag**(i)@B@U**(i)@A@Udag**(i)@B@U**(i)@A@state
    return O1

@jit(nopython=True, parallel=True, fastmath = True)
def O2_numpy_ket(state, statedag, A, B, U, Udag, i):
    O2 = statedag@(Udag**(i)@B@U**(i))**2@A**2@state
    return O2

@jit(nopython=True, parallel=True, fastmath = True)
def C_t_commutator_numpy_Tinf(A, B_t):
    com = B_t@A - A@B_t
    C_t = np.trace(com.H@com)/N
    return C_t
    
@jit(nopython=True, parallel=True, fastmath = True)
def C_t_commutator_numpy_state(state, A, B_t):
    com = B_t@A - A@B_t
    C_t = np.trace(state@com.H@com)
    return C_t

def Evolution_FFT_Tinf(time_lim, N, Ks, A, B, op = 'X', soloOTOC=False):
    
    start_time = time.time() 
    
    OTOC_Ks = np.zeros((len(Ks), time_lim))#[] # OTOC for each Ks
    if not soloOTOC:
        O1_Ks = np.zeros((len(Ks), time_lim))#[] # O1 for each Ks
        O2_Ks = np.zeros((len(Ks), time_lim))#[] # O2 for each Ks      
    
    for j, K in tqdm(enumerate(Ks), desc='Primary loop'):
        
        OTOC = np.zeros((time_lim))#[]
        if not soloOTOC:
            O1s = np.zeros((time_lim))#[]
            O2s = np.zeros((time_lim))#[]
        # Distinct evolution for each operator X or P
        U = Qobj(UU(K, N, op))
        
        # Calculo el OTOC para cada tiempo pero para un K fijo
        for i in tqdm(range(time_lim), desc='Secondary loop'):
            
            if i==0:
                B_t = B
            else:
                # FFT for efficient evolution
                # diagonal evolution
                 B_t = B_t.transform(U.dag())# U*B_t*U.dag()
            
            if soloOTOC:
                # commutator [B(t), A]
                com = commutator(B_t,A)
                # average of |[B(t), A]|**2
                C_t = (com.dag()*com).tr()/N
                # print(prom)
                OTOC.append(np.abs(C_t))
                
            else:
                
                O1 = (B_t*A*B_t*A).tr()
                O2 = (B_t**2*A**2).tr()
                            
                C_t = -2*( O1 - O2 )/N
                
                OTOC[i] = np.abs(C_t)#OTOC.append(np.abs(C_t.data.toarray()))
                O1s[i] = np.abs(O1)#O1s.append(np.abs(O1.data.toarray()))
                O2s[i] = np.abs(O2)#O2s.append(np.abs(O2.data.toarray()))

        OTOC_Ks[j,:] = OTOC#OTOC_Ks.append(OTOC)
        if not soloOTOC:
            O1_Ks[j,:] = O1s#O1_Ks.append(O1s)
            O2_Ks[j,:] = O2s#O2_Ks.append(O2s)  
        
    print(f"\nTOTAL --- {time.time() - start_time} seconds ---" )
    flag = 'FFT_with_Tinf_state'
    if not soloOTOC:
        return [OTOC_Ks, O1_Ks, O2_Ks, flag, soloOTOC]
    else:
        flag = flag+'_soloOTOC'
        return [OTOC_Ks, flag, soloOTOC]

def Evolution_FFT_numpy_Tinf(time_lim, N, Ks, A, B, op = 'X', soloOTOC=False):
    
    start_time = time.time() 
    
    OTOC_Ks = np.zeros((len(Ks), time_lim), dtype=np.complex_)#[] # OTOC for each Ks
    if not soloOTOC:
        O1_Ks = np.zeros((len(Ks), time_lim), dtype=np.complex_)#[] # O1 for each Ks
        O2_Ks = np.zeros((len(Ks), time_lim), dtype=np.complex_)#[] # O2 for each Ks      
    
    for j, K in tqdm(enumerate(Ks), desc='Primary loop'):
        
        OTOC = np.zeros((time_lim), dtype=np.complex_)#[]
        if not soloOTOC:
            O1s = np.zeros((time_lim), dtype=np.complex_)#[]
            O2s = np.zeros((time_lim), dtype=np.complex_)#[]
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
            
            if soloOTOC:
                # average of |[B(t), A]|**2
                C_t = C_t_commutator_numpy_Tinf(A, B_t)
                # print(prom)
                OTOC.append(C_t)
                
            else:
                
                O1 = O1_numpy_Tinf(A, B_t)
                O2 = O2_numpy_Tinf(A, B_t)
                            
                C_t = -2*( O1 - O2 )/N
                
                OTOC[i] = C_t#OTOC.append(np.abs(C_t.data.toarray()))
                O1s[i] = O1#O1s.append(np.abs(O1.data.toarray()))
                O2s[i] = O2#O2s.append(np.abs(O2.data.toarray()))

        OTOC_Ks[j,:] = OTOC#OTOC_Ks.append(OTOC)
        if not soloOTOC:
            O1_Ks[j,:] = O1s#O1_Ks.append(O1s)
            O2_Ks[j,:] = O2s#O2_Ks.append(O2s)  
        
    print(f"\nTOTAL --- {time.time() - start_time} seconds ---" )
    flag = 'FFT_with_Tinf_state'
    if not soloOTOC:
        return [OTOC_Ks, O1_Ks, O2_Ks, flag, soloOTOC]
    else:
        flag = flag+'_soloOTOC'
        return [OTOC_Ks, flag, soloOTOC]
    

def Evolution_FFT_state(time_lim, state, N, Ks, A, B, op = 'X', ket=False, soloOTOC=False):
    
    # start time of rutine
    start_time = time.time() 
    
    OTOC_Ks = np.zeros((len(Ks), time_lim))#[] # OTOC for each Ks
    if not soloOTOC:
        O1_Ks = np.zeros((len(Ks), time_lim))#[] # O1 for each Ks
        O2_Ks = np.zeros((len(Ks), time_lim))#[] # O2 for each Ks      
    
    for j, K in tqdm(enumerate(Ks), desc='Primary loop'):
        OTOC = np.zeros((time_lim))#[]
        if not soloOTOC:
            O1s = np.zeros((time_lim))#[]
            O2s = np.zeros((time_lim))#[]
            
        # Distinct evolution for each operator X or P
        U = Qobj(UU(K, N, op))
            
        # Calculo el OTOC para cada tiempo pero para un K fijo
        for i in tqdm(range(time_lim), desc='Secondary loop'):
            

            # evolution in the Schrodinger picture             
            if ket:
                O1 = state.dag()*U.dag()**(i)*B*U**(i)*A*U.dag()**(i)*B*U**(i)*A*state
                O2 = state.dag()*(U.dag()**(i)*B*U**(i))**2*A**2*state
                
                C_t = -2*( O1 - O2 )
                
                # print(C_t)
            
                OTOC[i] = np.abs(C_t[0,0])#OTOC.append(np.abs(C_t.data.toarray()))
                O1s[i] = np.abs(O1[0,0])#O1s.append(np.abs(O1.data.toarray()))
                O2s[i] = np.abs(O2[0,0])#O2s.append(np.abs(O2.data.toarray()))
                
            # evolution in the Heisenberg picture             
            else:
                if i==0:
                    B_t = B
                else: 
                    # FFT for efficient evolution
                    # diagonal evolution
                    B_t = B_t.transform(U.dag())#U*B_t*U.dag() # .dag()
                
                if soloOTOC:
                    # commutator [B(t), A]
                    com = commutator(B_t,A)
                    # average of |[B(t), A]|**2
                    C_t = (state*com.dag()*com).tr()
                    # print(prom)
                    OTOC.append(np.abs(C_t))
                else:
                        
                    O1 = (state*B_t*A*B_t*A).tr()
                    O2 = (state*B_t**2*A**2).tr()
                    
                    C_t = -2*( O1 - O2 )
                    
                    print(C_t)
                    
                    OTOC[i] = np.abs(C_t)#OTOC.append(np.abs(C_t.data.toarray()))
                    O1s[i] = np.abs(O1)#O1s.append(np.abs(O1.data.toarray()))
                    O2s[i] = np.abs(O2)#O2s.append(np.abs(O2.data.toarray()))
        
        OTOC_Ks[j,:] = OTOC#OTOC_Ks.append(OTOC)
        if not soloOTOC:
            O1_Ks[j,:] = O1s#O1_Ks.append(O1s)
            O2_Ks[j,:] = O2s#O2_Ks.append(O2s)  
            
    # return time duration of rutine
    print(f"\nTOTAL --- {time.time() - start_time} seconds ---" )
    
    flag = 'FFT_with_arbitrary_state'
    if ket:
        flag = flag+f'_ket_sigma{sigma}_sinheff'
    else:
        flag = flag+f'_density_sigma{sigma}_sinheff'
        
    if not soloOTOC:
        return [OTOC_Ks, O1_Ks, O2_Ks, flag, ket, soloOTOC]
    else:
        flag = flag+'_soloOTOC'
        return [OTOC_Ks, flag, ket, soloOTOC]
    
def Evolution_FFT_numpy_state(time_lim, state, N, Ks, A, B, op = 'X', ket=False, soloOTOC=False):
    
    statedag = state.H
    # start time of rutine
    start_time = time.time() 
    
    OTOC_Ks = np.zeros((len(Ks), time_lim), dtype=np.complex_)#[] # OTOC for each Ks
    if not soloOTOC:
        O1_Ks = np.zeros((len(Ks), time_lim), dtype=np.complex_)#[] # O1 for each Ks
        O2_Ks = np.zeros((len(Ks), time_lim), dtype=np.complex_)#[] # O2 for each Ks      
    
    for j, K in tqdm(enumerate(Ks), desc='Primary loop'):
        OTOC = np.zeros((time_lim), dtype=np.complex_)#[]
        if not soloOTOC:
            O1s = np.zeros((time_lim), dtype=np.complex_)#[]
            O2s = np.zeros((time_lim), dtype=np.complex_)#[]
            
        # Distinct evolution for each operator X or P
        U = UU(K, N, op)
        Udag = U.H
        # Calculo el OTOC para cada tiempo pero para un K fijo
        for i in tqdm(range(time_lim), desc='Secondary loop'):
            

            # evolution in the Schrodinger picture             
            if ket:
                    
                O1 = O1_numpy_ket(state, statedag, A, B, U, Udag, i)
                O2 = O2_numpy_ket(state, statedag, A, B, U, Udag, i)
                
                C_t = -2*( O1 - O2 )
                
                # print(C_t)
            
                OTOC[i] = C_t[0,0]#OTOC.append(np.abs(C_t.data.toarray()))
                O1s[i] = O1[0,0]#O1s.append(np.abs(O1.data.toarray()))
                O2s[i] = O2[0,0]#O2s.append(np.abs(O2.data.toarray()))
                
            # evolution in the Heisenberg picture             
            else:
                if i==0:
                    B_t = B
                else: 
                    # FFT for efficient evolution
                    # diagonal evolution
                    B_t = Evolucion_numpy(B_t, Udag, U)# U*B_t*U.dag()
                
                if soloOTOC:
                    # average of |[B(t), A]|**2
                    C_t = C_t_commutator_numpy_state(state, A, B_t)
                    # print(prom)
                    OTOC.append(np.abs(C_t))
                else:
                        
                    O1 = O1_numpy_state(state, A, B_t)
                    O2 = O2_numpy_state(state, A, B_t)
                    
                    C_t = -2*( O1 - O2 )
                    
                    print(C_t)
                    
                    OTOC[i] = C_t#OTOC.append(np.abs(C_t.data.toarray()))
                    O1s[i] = O1#O1s.append(np.abs(O1.data.toarray()))
                    O2s[i] = O2#O2s.append(np.abs(O2.data.toarray()))
        
        OTOC_Ks[j,:] = OTOC#OTOC_Ks.append(OTOC)
        if not soloOTOC:
            O1_Ks[j,:] = O1s#O1_Ks.append(O1s)
            O2_Ks[j,:] = O2s#O2_Ks.append(O2s)  
            
    # return time duration of rutine
    print(f"\n Ks loop --- {time.time() - start_time} seconds ---" )
    
    flag = 'FFT_with_arbitrary_state'
    if ket:
        flag = flag+f'_ket_sigma{sigma}_sinheff'
    else:
        flag = flag+f'_density__sigma{sigma}_sinheff'
        
    if not soloOTOC:
        return [OTOC_Ks, O1_Ks, O2_Ks, flag, ket, soloOTOC]
    else:
        flag = flag+'_soloOTOC'
        return [OTOC_Ks, flag, ket, soloOTOC]

#%%
# Define basis size        
N = 5000#2**8 # Tiene que ser par

# Define position and momentum values in the torus
qs = np.arange(0, N) #take qs in [0;N) with qs integer

t0 = time.time()
# Define Schwinger operators
Us = fU(N, qs)
Vs = fV(N)

# Define Schwinger operators
# Us_numpy = fU_numpy(N, qs)
# Vs_numpy = fV_numpy(N)

# # # Define momentum and position operators??
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
time_lim = int(1e1) # number of kicks

#%% clear all variables
# import sys
# sys.modules[__name__].__dict__.clear()
#%% numpy's Evolutions
# Select type of evolution and state

##### pure state

### compute OTOC with commutator qutip's method 
# OTOC_Ks, flag, ket, soloOTOC = Evolution_FFT_numpy_state(times, state0, N, Ks, A, B, op ='P', ket=False, soloOTOC=True)

### compute OTOC with O1 and O2 in the "Heisenberg picture"
# OTOC_Ks, O1_Ks, O2_Ks, flag, ket, soloOTOC = Evolution_FFT_numpy_state(time_lim, state0_numpy, N, Ks, A, B, op ='P', ket=False, soloOTOC=False)

### compute OTOC with O1 and O2 in the "Schrodinger picture"
# OTOC_Ks, O1_Ks, O2_Ks, flag, ket, soloOTOC = Evolution_FFT_numpy_state(time_lim, state0_numpy, N, Ks, A, B, op ='P', ket=True, soloOTOC=False)


##### T inf thermal state

### compute OTOC with commutator qutip's method
# OTOC_Ks, flag, soloOTOC = Evolution_FFT_numpy_Tinf(time_lim, N, Ks, A, B, op ='X', soloOTOC=True)

### compute OTOC with O1 and O2 in the "Heisenberg picture"
OTOC_Ks, O1_Ks, O2_Ks, flag, soloOTOC = Evolution_FFT_numpy_Tinf(time_lim, N, Ks, A, B, op ='X', soloOTOC=False)


# define file name
file = flag+f'_k{Ks}_basis_size{N}_time_lim{time_lim}'+operators+'.npz'#+'_evolucion_al_reves' _centro{n0}

#%%
# Select type of evolution and state

##### pure state

### compute OTOC with commutator qutip's method 
# OTOC_Ks, flag, ket, soloOTOC = Evolution_FFT_state(times, state0, N, Ks, A, B, op ='P', ket=False, soloOTOC=True)

## compute OTOC with O1 and O2 in the "Heisenberg picture"
# OTOC_Ks, O1_Ks, O2_Ks, flag, ket, soloOTOC = Evolution_FFT_state(time_lim, state0, N, Ks, A, B, op ='P', ket=False, soloOTOC=False)

### compute OTOC with O1 and O2 in the "Schrodinger picture"
# OTOC_Ks, O1_Ks, O2_Ks, flag, ket, soloOTOC = Evolution_FFT_state(time_lim, state0, N, Ks, A, B, op ='P', ket=True, soloOTOC=False)


##### T inf thermal state

### compute OTOC with commutator qutip's method
# OTOC_Ks, flag, soloOTOC = Evolution_FFT_Tinf(time_lim, N, Ks, A, B, op ='X', soloOTOC=True)

### compute OTOC with O1 and O2 in the "Heisenberg picture"
# OTOC_Ks, O1_Ks, O2_Ks, flag, soloOTOC = Evolution_FFT_Tinf(time_lim, N, Ks, A, B, op ='X', soloOTOC=False)


# define file name
# file = flag+f'_k{Ks}_basis_size{N}_time_lim{time_lim}_'+operators+'.npz'#+'_evolucion_al_reves'
#%%
# Compute the lyapunov exponents for Ks values
#STANDARD
def lystandar(q,p,k):
    ### time step for momentum and position
    # pf = np.mod(p+k*np.sin(2.*np.pi*q)/(2.*np.pi),1.)
    # qf = np.mod(q+pf,1.)
    
    # matrix elements (partial derivates)
    a=1.
    b=k*np.cos(2.*np.pi*q)
    c=1.
    d=1.+k*np.cos(2.*np.pi*q)
    
    # matrix
    m=[[a,b],[c,d]]
    
    # determinant
#    print(a*d-b*c)
    
    # eigenvalues and eigenvectors
    l1=np.linalg.eig(m)
    
    return l1[0]

# number of initial conditions
nci=300

# limits of K array
# k0=0.1
# k1=20.
# number of time_lim
# npp=100

# K array
# kk = np.linspace(k0, k1, npp+1)
kk = Ks

# Lyapunov lists
ly1 = []

for count10, k in enumerate(kk):
    print(f'Iteration {count10}',f'K = {k}')
    # nci initial conditions for position and momentum
    ci=np.random.rand(nci,2)
    
    # counts of l1.imag() == 0
    nc=0
    
    # initialize lyapunovs
    lyap=0.
    lyap2=0.
    
    for count0 in range(nci):
        # take one initial condition
        q0, p0 = ci[count0]
        
        # eigenvalues of standar map's matrix
        l1, l2 = lystandar(q0,p0,k)
        
        # check imaginary part of eigenvalues
        if(np.imag(l1) == 0):
            
            # take absolute value
            l1=np.abs(l1)
            l2=np.abs(l2)
            # print(l1, l2)
            
            # select greatest eigenvalue
            if(l1>l2):
                # actualize lyapunovs for average
                lyap=lyap+np.log(l1)
            else:
                # actualize lyapunovs for average
                lyap=lyap+np.log(l2)
            nc+=1
            
    # save average
    ly1.append(lyap/nci)

# save data in .npz file
if not soloOTOC:
    np.savez(file, OTOC=OTOC_Ks, O1_Ks=O1_Ks, O2_Ks=O2_Ks, ly1=ly1)            
else:
    np.savez(file, OTOC=OTOC_Ks, ly1=ly1)
# #%% plot 



# # Ks = [0.05, 0.1,  0.15, 0.2,  0.25, 0.3,  0.35, 0.4,  0.45]
# # Ks = np.array([0.1, 0.2, 0.3, 0.4])*4*np.pi**2
# # Ks = [2.0133993,3.98732018,5.96124106,7.93516194,9.90908282,11.8830037,13.85692458,15.83084546,17.80476634]

# # soloOTOC=False

# # domain for the lyapunov expression
# xdom = np.linspace(0,2,int(1e3))

# # time_lim = 5
# times = np.arange(0,time_lim)

# # N = 2000
# # define colormap
# colors = plt.cm.jet(np.linspace(0,1,len(Ks)))

# # create the figure
# plt.figure(figsize=(16,10), dpi=100)

# # all plots in the same figure
# plt.title(f'OTOC N = {N} A = X, B = P. n0 = {n0}, sigma = {sigma}')
# for i, n in enumerate(Ks):
    
#     # compute the Ehrenfest's time
#     tE = np.log(N)/ly1[i]
    
#     # plot's shift parameters 
#     factor = 1#2**(i)
#     b = 0#np.log2(n)
    
#     # plot log10(C(t))
    
#     if ket: # reshape array for plot
#         OTOC_Ks[i] = np.reshape(OTOC_Ks[i], (time_lim,))
    
#     yaxis = np.log10(OTOC_Ks[i]*factor)+b#OTOC_Ks[i]#
#     plt.plot(times, yaxis,':^r' , label=f'pure state', color=colors[i]);#(2**i)*
    
#     # yaxis_Tinf = np.log10(OTOC_Ks_Tinf[i]*factor)+b#OTOC_Ks[i]#
#     # plt.plot(times, yaxis_Tinf,':^r' , label=r'$T = \infty$')#, color=colors[i]);#(2**i)*
    
#     # plot lyapunov  
#     ylyap = np.log10(np.exp(2*(ly1[i])*(xdom)))+b+np.log10(OTOC[i][0])
#     plt.plot(xdom, ylyap,'b-' ,lw=0.5, label=f'Lyapunov', color=colors[i])
    
#     # plot vertical lines in the corresponding Ehrenfest's time for each K
#     plt.vlines(tE, np.min(np.log10(OTOC_Ks[0]*factor)+b), np.max(np.log10(OTOC_Ks[0]*factor)+b), lw=0.5, ls='dashed', alpha=0.8, color=colors[i])
    
# plt.xlabel('Time');
# plt.ylabel(r'$log \left(C(t) \right)$');
# # plt.xlim((0,10))
# # plt.ylim((-12,0))
# plt.grid();
# plt.legend(loc='best');
# plt.show()
# plt.savefig('OTOC_'+flag+f'_k{Ks}_basis_size{N}_AX_BP.png', dpi=300)#_Gauss_sigma{sigma}

# if not soloOTOC:
    
#     # create the figure
#     plt.figure(figsize=(16,10), dpi=100)
    
#     # all plots in the same figure
#     plt.title(f'O2 N = {N} A = X, B = P. n0 = {n0}, sigma = {sigma}')
    
#     for i, n in enumerate(Ks):
#         # plot shift parameters
#         factor = 1#2**(i)
#         b = 0#np.log2(n)
        
#         # plot log10(O2(t))
        
#         if ket: # reshape array for plot
#             O2_Ks[i] = np.reshape(O2_Ks[i], (time_lim,))
            
#         yO2=np.log10(O2_Ks[i]/N*factor)+b
#         plt.plot(times, yO2,':^r' , label=f'pure state', color=colors[i]);#(2**i)*
        
#         # yaxis_Tinf = np.log10(O2_Ks_Tinf[i]/N*factor)+b#OTOC_Ks[i]#
#         # plt.plot(times, yaxis_Tinf,':^r' , label=r'$T = \infty$')#, color=colors[i]);#(2**i)*
        
#     plt.xlabel('Time');
#     plt.ylabel(r'$log \left(O_2(t) \right)$');
#     # plt.xlim((0,10))
#     # plt.ylim((-8,0))
#     plt.grid();
#     plt.legend(loc='best');
#     plt.show()
#     plt.savefig('O2_'+flag+f'_k{Ks}_basis_size{N}_AX_BP.png', dpi=300)#_Gauss_sigma{sigma}
    
#     alpha = 0.47 # for K = 19.74
    
#     # create the figure
#     plt.figure(figsize=(16,10), dpi=100)
    
#     # all plots in the same figure
#     plt.title(f'O1 N = {N} A = X, B = P. n0 = {n0}, sigma = {sigma}')
    
#     for i, n in enumerate(Ks):
#         # plot shift parameters
#         factor = 1#2**(i)
#         b = 0#np.log2(n)
        
#         # plot log10(O1(t))
#         if ket: # reshape array for plot   
#             O1_Ks[i] = np.reshape(O1_Ks[i], (time_lim,))
#         yO1 = np.log10(O1_Ks[i]/N*factor)+b
#         plt.plot(times, yO1,':^r' , label=f'pure state', color=colors[i]);#(2**i)*
        
#         # yaxis_Tinf = np.log10(O1_Ks_Tinf[i]/N*factor)+b#OTOC_Ks[i]#
#         # plt.plot(times, yaxis_Tinf,':^r' , label=r'$T = \infty$')#, color=colors[i]);#(2**i)*
        
#         # plot PR Resonances
#         plt.plot(xdom,np.log10(abs(alpha)**(2*xdom))+np.log10(O1_Ks[i][0]/N*factor)+b,'b-' ,lw=0.5, label=f'RPR', color=colors[i])#np.log10(OTOC_Ks[i][0])
#     plt.xlabel('Time');
#     plt.ylabel(r'$log \left(O_1(t) \right)$');
#     # plt.xlim((0,10))
#     # plt.ylim((-8,0))
#     plt.grid();
#     plt.legend(loc='best');
#     plt.show()
#     plt.savefig('O1_'+flag+f'_k{Ks}_basis_size{N}_AX_BP.png', dpi=300)#_Gauss_sigma{sigma}
#     # np.savez(f'OTOC_KR_Knoeff_k{Ks}_basis_size{N}_AP_BX.npz', OTOC=OTOC_Ks)
# #%% PP XP XX plot
# # # load data XX
# file = 'FFT_with_Tinf_state_k[19.74]_basis_size2000_time_lim10AX_BX.npz'
# archives = np.load(file)
# OTOC_XX = archives['OTOC']
# O1_XX = archives['O1_Ks']
# O2_XX = archives['O2_Ks']
# # ly1 = archives['ly1']

# # # load data PX
# file = 'FFT_with_Tinf_state_k[19.74]_basis_size2000_time_lim10AP_BX.npz'
# archives = np.load(file)
# OTOC_PX = archives['OTOC']
# O1_PX = archives['O1_Ks']
# O2_PX = archives['O2_Ks']
# # ly1 = archives['ly1']

# # # load data PP
# file = 'FFT_with_Tinf_state_k[19.74]_basis_size2000_time_lim10AP_BP.npz'
# archives = np.load(file)
# OTOC_PP = archives['OTOC']
# O1_PP = archives['O1_Ks']
# O2_PP = archives['O2_Ks']

# xdom = np.linspace(0,2,int(1e3))

# # time_lim = 5
# times = np.arange(0,time_lim)

# # N = 2000
# # define colormap
# colors = plt.cm.jet(np.linspace(0,1,len(Ks)))

# # create the figure
# plt.figure(figsize=(16,10), dpi=100)

# # all plots in the same figure
# # plt.title(f'OTOC N = {N} A = X, B = P. n0 = {n0}, sigma = {sigma}')
# for i, n in enumerate(Ks):
    
#     # compute the Ehrenfest's time
#     tE = np.log(N)/ly1[i]
    
#     # plot's shift parameters 
#     factor = 1#2**(i)
#     b = 0#np.log2(n)
    
#     # plot log10(C(t))
    
#     # if ket: # reshape array for plot
#     #     OTOC_Ks[i] = np.reshape(OTOC_Ks[i], (time_lim,))
    
#     yaxis_XX = np.log10(OTOC_XX[i]*factor)+b#OTOC_Ks[i]#
#     plt.plot(times, yaxis_XX,':^r' , label=f'XX');#(2**i)*
    
#     yaxis_PX = np.log10(OTOC_PX[i]*factor)+b#OTOC_Ks[i]#
#     plt.plot(times, yaxis_PX,':^b' , label=f'PX');#(2**i)*
    
#     yaxis_PP = np.log10(OTOC_XX[i]*factor)+b#OTOC_Ks[i]#
#     plt.plot(times, yaxis_PP,':^g' , label=f'PP');#(2**i)*
    
#     # yaxis_Tinf = np.log10(OTOC_Ks_Tinf[i]*factor)+b#OTOC_Ks[i]#
#     # plt.plot(times, yaxis_Tinf,':^r' , label=r'$T = \infty$')#, color=colors[i]);#(2**i)*
    
#     # plot lyapunov  
#     ylyap = np.log10(np.exp(2*(ly1[i])*(xdom)))+b+np.log10(OTOC[i][0])
#     plt.plot(xdom, ylyap,'b-' ,lw=0.5, label=f'Lyapunov', color=colors[i])
    
#     # plot vertical lines in the corresponding Ehrenfest's time for each K
#     plt.vlines(tE, np.min(np.log10(OTOC_Ks[0]*factor)+b), np.max(np.log10(OTOC_Ks[0]*factor)+b), lw=0.5, ls='dashed', alpha=0.8, color=colors[i])
    
# plt.xlabel('Time');
# plt.ylabel(r'$log \left(C(t) \right)$');
# # plt.xlim((0,10))
# # plt.ylim((-12,0))
# plt.grid();
# plt.legend(loc='best');
# plt.show()
# # plt.savefig('OTOC_'+flag+f'_k{Ks}_basis_size{N}_AX_BP.png', dpi=300)#_Gauss_sigma{sigma}
# if not soloOTOC:
    
#     # create the figure
#     plt.figure(figsize=(16,10), dpi=100)
    
#     # all plots in the same figure
#     # plt.title(f'O2 N = {N} A = X, B = P. n0 = {n0}, sigma = {sigma}')
    
#     for i, n in enumerate(Ks):
#         # plot shift parameters
#         factor = 1#2**(i)
#         b = 0#np.log2(n)
        
#         # plot log10(O2(t))
        
#         # if ket: # reshape array for plot
#         #     O2_Ks[i] = np.reshape(O2_Ks[i], (time_lim,))
            
#         yO2_XX = np.log10(O2_XX[i]/N*factor)+b
#         plt.plot(times, yO2_XX,':^r' , label=f'XX');#(2**i)*
        
#         yO2_PX = np.log10(O2_PX[i]/N*factor)+b
#         plt.plot(times, yO2_PX,':^b' , label=f'PX');#(2**i)*
        
#         yO2_PP = np.log10(O2_PP[i]/N*factor)+b
#         plt.plot(times, yO2_PP,':^g' , label=f'PP');#(2**i)*
        
#         # yaxis_Tinf = np.log10(O2_Ks_Tinf[i]/N*factor)+b#OTOC_Ks[i]#
#         # plt.plot(times, yaxis_Tinf,':^r' , label=r'$T = \infty$')#, color=colors[i]);#(2**i)*
        
#     plt.xlabel('Time');
#     plt.ylabel(r'$log \left(O_2(t) \right)$');
#     # plt.xlim((0,10))
#     # plt.ylim((-8,0))
#     plt.grid();
#     plt.legend(loc='best');
#     plt.show()
#     plt.savefig('O2_'+flag+f'_k{Ks}_basis_size{N}_AX_BP.png', dpi=300)#_Gauss_sigma{sigma}
    
#     alpha = 0.47 # for K = 19.74
    
#     # create the figure
#     plt.figure(figsize=(16,10), dpi=100)
    
#     # all plots in the same figure
#     # plt.title(f'O1 N = {N} A = X, B = P. n0 = {n0}, sigma = {sigma}')
    
#     for i, n in enumerate(Ks):
#         # plot shift parameters
#         factor = 1#2**(i)
#         b = 0#np.log2(n)
        
#         # plot log10(O1(t))
#         # if ket: # reshape array for plot   
#         #     O1_Ks[i] = np.reshape(O1_Ks[i], (time_lim,))
        
#         yO1_XX = np.log10(O1_XX[i]/N*factor)+b
#         plt.plot(times, yO1_XX,':^r' , label=f'XX');#(2**i)*
        
#         yO1_PX = np.log10(O1_PX[i]/N*factor)+b
#         plt.plot(times, yO1_PX,':^b' , label=f'PX');#(2**i)*
        
#         yO1_PP = np.log10(O1_PP[i]/N*factor)+b
#         plt.plot(times, yO1_PP,':^g' , label=f'PP');#(2**i)*
        
#         # yaxis_Tinf = np.log10(O1_Ks_Tinf[i]/N*factor)+b#OTOC_Ks[i]#
#         # plt.plot(times, yaxis_Tinf,':^r' , label=r'$T = \infty$')#, color=colors[i]);#(2**i)*
        
#         # plot PR Resonances
#         plt.plot(xdom,np.log10(abs(alpha)**(2*xdom))+np.log10(O1_Ks[i][0]/N*factor)+b,'b-' ,lw=0.5, label=f'RPR', color=colors[i])#np.log10(OTOC_Ks[i][0])
#     plt.xlabel('Time');
#     plt.ylabel(r'$log \left(O_1(t) \right)$');
#     # plt.xlim((0,10))
#     # plt.ylim((-8,0))
#     plt.grid();
#     plt.legend(loc='best');
#     plt.show()
#     plt.savefig('O1_'+flag+f'_k{Ks}_basis_size{N}_AX_BP.png', dpi=300)#_Gauss_sigma{sigma}
#     # np.savez(f'OTOC_KR_Knoeff_k{Ks}_basis_size{N}_AP_BX.npz', OTOC=OTOC_Ks)