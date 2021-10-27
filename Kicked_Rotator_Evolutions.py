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

@jit(nopython=True, parallel=True, fastmath = True)#, cache=True)#(nopython=True)
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
    return U

def fV(N):
    V = 0
    for j in range(N):
        V += basis(N,(j+1)%N)*basis(N,j).dag()
    return V

# print(V(N))

def fU(N, qs):
    U = 0
    tau = np.exp(2j*np.pi/N)
    for j in range(N):
        U += basis(N,j)*basis(N,j).dag()*tau**(qs[j])
    return U

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
                 B_t = B_t.transform(U)# U*B_t*U.dag()
            
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
                
                OTOC[i] = np.abs(C_t)#OTOC.append(np.abs(C_t.data.toarray()))
                O1s[i] = np.abs(O1)#O1s.append(np.abs(O1.data.toarray()))
                O2s[i] = np.abs(O2)#O2s.append(np.abs(O2.data.toarray()))
                
            # evolution in the Heisenberg picture             
            else:
                if i==0:
                    B_t = B
                else: 
                    # FFT for efficient evolution
                    # diagonal evolution
                    B_t = B_t.transform(U)#U*B_t*U.dag() # .dag()
                
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
#%%
# Define basis size        
N = 3000#2**8

# Define position and momentum values in the torus
qs = np.arange(0, N) #take qs in [0;N) with qs integer

# Define Schwinger operators
Us = fU(N, qs)
Vs = fV(N)

# Define momentum and position operators
P = (Vs - Vs.dag())/(2j)
X = (Us - Us.dag())/(2j)

# Select operators for the out-of-time correlators
A = P
B = X

operators = 'AP_BX'

# Define K values for the simulation

Kspelado = (np.arange(0.1, 0.5, 0.1))#np.array([0.523])#

Ks = Kspelado*(4*np.pi**2) # K values np.array([Kpelado])

# Define pure state
sigma = N/2
state0 = gaussian_state(int(N/2), 0, sigma)
# state0 = gaussian_state(int(N/2), 0, sigma, ket=True)

# Define time array
time_lim = int(3e1) # number of kicks

#%% clear all variables
# import sys
# sys.modules[__name__].__dict__.clear()
#%%
# Select type of evolution and state

##### pure state

### compute OTOC with commutator qutip's method 
# OTOC_Ks, flag, ket, soloOTOC = Evolution_FFT_state(times, state0, N, Ks, A, B, op ='P', ket=False, soloOTOC=True)

### compute OTOC with O1 and O2 in the "Heisenberg picture"
OTOC_Ks, O1_Ks, O2_Ks, flag, ket, soloOTOC = Evolution_FFT_state(time_lim, state0, N, Ks, A, B, op ='P', ket=False, soloOTOC=False)

### compute OTOC with O1 and O2 in the "Schrodinger picture"
# OTOC_Ks, O1_Ks, O2_Ks, flag, ket, soloOTOC = Evolution_FFT_state(time_lim, state0, N, Ks, A, B, op ='P', ket=True, soloOTOC=False)


##### T inf thermal state

### compute OTOC with commutator qutip's method
# OTOC_Ks, flag, soloOTOC = Evolution_FFT_Tinf(time_lim, N, Ks, A, B, op ='X', soloOTOC=True)

### compute OTOC with O1 and O2 in the "Heisenberg picture"
# OTOC_Ks, O1_Ks, O2_Ks, flag, soloOTOC = Evolution_FFT_Tinf(time_lim, N, Ks, A, B, op ='X', soloOTOC=False)


# define file name
file = flag+f'_k{Kspelado}_basis_size{N}_time_lim{time_lim}_'+operators+'.npz'#+'_evolucion_al_reves'
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
#%% plot 

# load data
# file = 'FFT_with_arbitrary_state_density_sigma500.0_sinheff_k[2.0133993_3.98732018_5.96124106_7.93516194_9.90908282_11.8830037_13.85692458_15.83084546_17.80476634]_basis_size1000_time_lim10_AP_BX.npz'
# archives = np.load(file)
# OTOC_Ks = archives['OTOC']
# O1_Ks = archives['O1_Ks']
# O2_Ks = archives['O2_Ks']
# ly1 = archives['ly1']

# domain for the lyapunov expression
xdom = np.linspace(0,2,int(1e3))

times = np.arange(0,time_lim)

# define colormap
colors = plt.cm.jet(np.linspace(0,1,len(Ks)))

# create the figure
plt.figure(figsize=(16,10), dpi=100)

# all plots in the same figure
plt.title(f'OTOC N = {N} A = X, B = P')
for i, n in enumerate(Ks):
    
    # compute the Ehrenfest's time
    tE = np.log(N)/ly1[i]
    
    # plot's shift parameters 
    factor = 1#2**(i)
    b = 0#np.log2(n)
    
    # plot log10(C(t))
    
    if ket: # reshape array for plot
        OTOC_Ks[i] = np.reshape(OTOC_Ks[i], (time_lim,))
    
    yaxis = np.log10(OTOC_Ks[i]*factor)+b
    plt.plot(times, yaxis,':^r' , label=f'K={Ks[i]}', color=colors[i]);#(2**i)*
    
    # plot lyapunov  
    ylyap = np.log10(np.exp(2*(ly1[i])*(xdom)))+b+np.log10(OTOC_Ks[i][0])
    plt.plot(xdom, ylyap,'b-' ,lw=0.5, label=f'Lyapunov', color=colors[i])
    
    # plot vertical lines in the corresponding Ehrenfest's time for each K
    plt.vlines(tE, np.min(np.log10(OTOC_Ks[0]*factor)+b), np.max(np.log10(OTOC_Ks[0]*factor)+b), lw=2, ls='dashed', alpha=0.8, color=colors[i])
    
plt.xlabel('Time');
plt.ylabel(r'$log \left(C(t) \right)$');
# plt.xlim((0,5))
# plt.ylim((-12,0))
plt.grid();
plt.legend(loc='best');
plt.show()
# plt.savefig('OTOC_'+flag+f'_k{Ks}_basis_size{N}_AX_BP.png', dpi=300)#_Gauss_sigma{sigma}

# if not soloOTOC:
    
#     # create the figure
#     plt.figure(figsize=(16,10), dpi=100)
    
#     # all plots in the same figure
#     plt.title(f'O2 N = {N} A = X, B = P'+flag)
    
#     for i, n in enumerate(Ks):
#         # plot shift parameters
#         factor = 1#2**(i)
#         b = 0#np.log2(n)
        
#         # plot log10(O2(t))
        
#         if ket: # reshape array for plot
#             O2_Ks[i] = np.reshape(O2_Ks[i], (time_lim,))
            
#         yO2=np.log10(O2_Ks[i]*factor)+b
#         plt.plot(times, yO2,':^r' , label=f'K={Ks[i]}', color=colors[i]);#(2**i)*
        
#     plt.xlabel('Time');
#     plt.ylabel(r'$log \left(O_2(t) \right)$');
#     # plt.xlim((0,10))
#     # plt.ylim((-8,0))
#     plt.grid();
#     plt.legend(loc='best');
#     plt.show()
#     plt.savefig('O2_'+flag+f'_k{Ks}_basis_size{N}_AX_BP.png', dpi=300)#_Gauss_sigma{sigma}
    
#     # alpha = 0.47 # for K = 19.74
    
#     # create the figure
#     plt.figure(figsize=(16,10), dpi=100)
    
#     # all plots in the same figure
#     plt.title(f'O1 N = {N} A = X, B = P'+flag)
    
#     for i, n in enumerate(Ks):
#         # plot shift parameters
#         factor = 1#2**(i)
#         b = 0#np.log2(n)
        
#         # plot log10(O1(t))
#         if ket: # reshape array for plot   
#             O1_Ks[i] = np.reshape(O1_Ks[i], (time_lim,))
#         yO1 = np.log10(O1_Ks[i]*factor)+b
#         plt.plot(times, yO1,':^r' , label=f'K={Ks[i]}', color=colors[i]);#(2**i)*
        
#         # plot PR Resonances
#         # plt.plot(xdom,np.log10(abs(alpha)**(2*xdom))+np.log10(O1_Ks[i][0]*factor)+b,'b-' ,lw=0.5, label=f'RPR', color=colors[i])#np.log10(OTOC_Ks[i][0])
#     plt.xlabel('Time');
#     plt.ylabel(r'$log \left(O_1(t) \right)$');
#     # plt.xlim((0,10))
#     # plt.ylim((-8,0))
#     plt.grid();
#     plt.legend(loc='best');
#     plt.show()
#     plt.savefig('O1_'+flag+f'_k{Ks}_basis_size{N}_AX_BP.png', dpi=300)#_Gauss_sigma{sigma}
#     # np.savez(f'OTOC_KR_Knoeff_k{Ks}_basis_size{N}_AP_BX.npz', OTOC=OTOC_Ks)
