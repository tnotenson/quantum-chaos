#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 20:42:17 2022

Kicked Rotor O1(t) for tofu@df.uba.ar

@author: tomasnotenson
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numba import jit
from qutip import *
from time import time
from tqdm import tqdm # customisable progressbar decorator for iterators
from cmath import phase
from sklearn.linear_model import LinearRegression #Regresión Lineal con scikit-learn

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
def O1_numpy(phi, A, B_t):
    O1 = np.trace(B_t@phi@A@B_t@A)
    return O1

@jit(nopython=True, parallel=True, fastmath = True)
def O2_numpy(phi, A, B_t):
    O2 = np.trace(B_t@B_t@phi@A@A)
    return O2

@jit(nopython=True, parallel=True, fastmath = True)
def C2_numpy_Tinf(A, B_t):
    C2 = np.trace(B_t@A)
    return C2

def C2_Evolution_FFT_numpy_Tinf(time_lim, N, Ks, A, B, op = 'X'):
    
    start_time = time() 
    
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
        
    print(f"\nTOTAL --- {time() - start_time} seconds ---" )
    flag = '2pC_FFT_with_Tinf_state'
    return [C2_Ks, flag]

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

def four_times_C_Evolution_FFT_numpy(time_lim, N, Ks, A, B, op = 'X', r=True,phi=[]):
       
    start_time = time() 
    
    O1_Ks = np.zeros((len(Ks), time_lim), dtype=np.complex_); O2_Ks = np.zeros((len(Ks), time_lim), dtype=np.complex_)#[] # OTOC for each Ks
    if r:
        r_Ks = np.zeros((len(Ks)))
    for j, K in tqdm(enumerate(Ks), desc='Primary loop'):
        
        O1 = np.zeros((time_lim), dtype=np.complex_); O2 = np.zeros((time_lim), dtype=np.complex_)#[]
        
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
            
            if len(phi)==0:
                C1 = O1_numpy_Tinf(A, B_t)
                C2 = O2_numpy_Tinf(A, B_t)
                state = '_Tinf_state'
            else:
                C1 = O1_numpy(phi, A, B_t)
                C2 = O2_numpy(phi, A, B_t)
                state = '_phi_state'
            O1[i] = C1#OTOC.append(np.abs(C_t.data.toarray()))
            O2[i] = C2
        O1_Ks[j,:] = O1
        O2_Ks[j,:] = O2
        if r:
            r_Ks[j] = r_normed
    print(f"\nTOTAL --- {time() - start_time} seconds ---" )
    flag = '4pC_FFT_with'+state
    if r:
        return [O1_Ks, O2_Ks, r_Ks, flag]
    return [O1_Ks, O2_Ks, flag]

def eigenvec_j_to_qp(eigenvector, mapa='normal'):
    N = len(eigenvector)
    Nx = int(np.sqrt(N))
    paso = 1/Nx
    eig_res = np.zeros((Nx,Nx))
    for j in range(N):
        q,p = qp_from_j(j, Nx, paso, mapa)
        # print(int(q*Nx),int(p*Nx))
        eig_res[int(q*Nx),int(p*Nx)] = eigenvector[j]
    return eig_res
#%% simulation C2
opA = 'X'
opB = 'P'

operatorss = 'A'+opA+'_B'+opB

Kpaso = 0#1/5
Ks = [6]#np.arange(15,20.1,Kpaso)

Ns = [15000]#np.arange(5000,20000,00)#[1000,2000,3000,4000,5000]

# for n,N in enumerate(Ns):
    
    
#     # Define basis size        
#     N = int(N)
    
#     # Define position and momentum values in the torus
#     qs = np.arange(0, N) #take qs in [0;N) with qs integer
    
#     t0 = time()
#     # Define Schwinger operators
#     Us = fU(N, qs)
#     Vs = fV(N)
        
#     # # # Define momentum and position operatorsÇ
#     P = (Vs - Vs.dag())/(2j)
#     X = (Us - Us.dag())/(2j)

#     t1 = time()
#     print(f'Tiempo operadores: {t1-t0}')
    
#     # Select operators for the out-of-time correlators
    
#     A = X.data.toarray()
#     B = P.data.toarray()
    
    
#     # Define pure state
#     t2 = time()
#     # sigma = 1/(2*np.pi*N)/10000
#     # state0 = gaussian_state(int(N/2), 0, sigma)
#     # state0 = gaussian_state(int(N/2), 0, sigma, ket=True)
    
#     # numpy states
#     # n0 = 0
#     # state0_numpy = gaussian_state_numpy(int(N/2), n0, sigma)
#     # state0_numpy = gaussian_state_numpy(int(N/2), 0, sigma, ket=True)
#     t3 = time()
#     print(f'Tiempo estado: {t3-t2}')
#     # Define time array
#     time_lim = int(5e1+1) # number of kicks
    
#     phi = []
        
#     # ### compute OTOC with O1 and O2 in the "Heisenberg picture"
#     C2_Ks, flag = C2_Evolution_FFT_numpy_Tinf(time_lim, N, Ks, A, B)
    
#     # # define file name
    
#     file = flag+f'_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_basis_size{N}_time_lim{time_lim}'+operatorss+'.npz'#+'_evolucion_al_reves' _centro{n0}
#     np.savez(file, C2_Ks=C2_Ks, Ks=Ks)

#%% simulation O1

# opA = 'X'
# opB = 'P'

# operatorss = 'A'+opA+'_B'+opB

# # Define K values for the simulation

# # Kspelado = np.array([0.5])#(np.arange(0.3, 0.51, 0.2))#

# Kpaso = 1/5
# Ks = np.arange(0,20.1,Kpaso)##np.array([2, 5, 8, 12, 15, 17, 19.74])#Kspelado*(4*np.pi**2) # K values np.array([Kpelado])

# Ns = [5000]#np.arange(1,9)*1e3

for n,N in enumerate(Ns):
    
    
    # Define basis size        
    # N = 8000#**11 # Tiene que ser par
    N = int(N)
    
    # Define position and momentum values in the torus
    qs = np.arange(0, N) #take qs in [0;N) with qs integer
    
    t0 = time()
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
    t1 = time()
    print(f'Tiempo operadores: {t1-t0}')
    
    # Select operators for the out-of-time correlators
    # A = P
    # B = X
    
    A = X.data.toarray()
    B = P.data.toarray()
    
    
    # Define pure state
    t2 = time()
    # sigma = 1/(2*np.pi*N)/10000
    # state0 = gaussian_state(int(N/2), 0, sigma)
    # state0 = gaussian_state(int(N/2), 0, sigma, ket=True)
    
    # numpy states
    # n0 = 0
    # state0_numpy = gaussian_state_numpy(int(N/2), n0, sigma)
    # state0_numpy = gaussian_state_numpy(int(N/2), 0, sigma, ket=True)
    t3 = time()
    print(f'Tiempo estado: {t3-t2}')
    # Define time array
    time_lim = int(5e1+1) # number of kicks
    
    # Neff = 80
    # ruido = 1e-6
    # mapa = 'normal'
    # cx = 1
    # K = Ks[0]
    # Nc = int(1e6)
    
    # flag = f'Ulam_approximation_mapa{mapa}_Sij_eigenvals_N{Neff}_ruido{ruido}_grilla{cx}N_K{K}_Nc{Nc}'
    # archives = np.load(flag+'.npz')
    # evec = archives['evec']
    # sustate = eigenvec_j_to_qp(evec[0])
    
    phi = []#np.identity(N)#-sustate
    
    modif = ''#'_sust_evec0'
    
    # ### compute OTOC with O1 and O2 in the "Heisenberg picture"
    O1_Ks, O2_Ks, r_Ks, flag = four_times_C_Evolution_FFT_numpy(time_lim, N, Ks, A, B, op ='X', r=True, phi=phi)
    
    # O1 = np.abs(O1_Ks)
    # # define file name
    
    file = flag+f'_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_basis_size{N}_time_lim{time_lim}'+operatorss+modif+'.npz'#+'_evolucion_al_reves' _centro{n0}
    np.savez(file, O1=O1_Ks, O2=O2_Ks, r_Ks=r_Ks, Ks=Ks)
#%% plot O1, O2 y C(t) vs t
   

# O1 = np.abs(O1_Ks[0])/N
# O2 = np.abs(O2_Ks[0])/N
# C_t = -2*(O1-O2)

# times = np.arange(0,time_lim)

# plt.figure(figsize=(16,8))
# plt.plot(times, O1, '.-', ms=10, lw=1.5, label=f'N={N}')
# plt.ylabel(r'$O_1$')
# plt.xlabel(r'$t$')
# plt.xticks(times[::2])
# plt.yscale('log')
# # plt.ylim(1e-7,1.1)
# # plt.xlim(-0.01,21)
# plt.grid()
# # plt.legend(loc = 'best')
# plt.savefig('O1_vs_t_Ns.png')

# plt.figure(figsize=(16,8))
# plt.plot(times, O2, '.-', ms=10, lw=1.5, label=f'N={N}')
# plt.ylabel(r'$O_2$')
# plt.xlabel(r'$t$')
# plt.xticks(times[::2])
# plt.yscale('log')
# # plt.ylim(1e-7,1.1)
# # plt.xlim(-0.01,21)
# plt.grid()
# # plt.legend(loc = 'best')
# plt.savefig('O2_vs_t_Ns.png')

# plt.figure(figsize=(16,8))
# plt.plot(times, C_t, '.-', ms=10, lw=1.5, label=f'N={N}')
# plt.ylabel(r'$C(t)$')
# plt.xlabel(r'$t$')
# plt.xticks(times[::2])
# plt.yscale('log')
# # plt.ylim(1e-7,1.1)
# # plt.xlim(-0.01,21)
# plt.grid()
# # plt.legend(loc = 'best')
# plt.savefig('Commutator_vs_t_Ns.png')

#%% plot O1 vs t para varios N

Ns = [1000,2000,5000,10000,15000]

markers = ['o','v','^','>','<','8','s','p','*']

fit = np.linspace(0,1,len(Ns))
cmap = mpl.cm.get_cmap('viridis')
color_gradients = cmap(fit)  

K = 6#Ks[0]

plt.figure(figsize=(12,8))
# plt.title(f'A={opA}, B={opB}')

time_lim = 51

times = np.arange(0,time_lim)
x = times#/np.log(N)
# pendientes = np.zeros((len(Ns)))

for n,N in enumerate(Ns):
    
    # Define basis size        
    N = int(N)
    
    if N!=15000:
            
        
        file = flag+f'_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_basis_size{N}_time_lim{time_lim}'+operatorss+modif+'.npz'#+'_evolucion_al_reves' _centro{n0}
        archives = np.load(file)
        O1_Ks = archives['O1']
           
    
        y = np.abs(O1_Ks[0])/N*4
        plt.plot(x[1:]/np.log(N), y[:-1], '.-', ms=8, lw=2.5, label=f'D={N}',  color=color_gradients[n])
    
    else:
        
        file = f'O1.{N}._std_k6.0.txt'
        O1_t = np.loadtxt(file)
        y = np.abs(O1_t[:,1])/N*4
        time_lim = O1_t[-1,0]+1
        x = np.arange(0,time_lim)#/np.log(N)    
        plt.plot(x/np.log(N), y, '.-', ms=8, lw=2.5, label=f'D={N}',  color=color_gradients[n])
    
    
    
    # oo = O1s[0] - resonancias[k]**(2*x[0])
    
    # # if N < 5000:
    # linf=3+n
    # lsup=7+n
    # # else: 
    # #     linf=6
    # #     lsup=10
    # xt = x[linf:lsup].reshape(-1,1)
    # yt = np.log10(y[linf:lsup])
    
    # regresion_lineal = LinearRegression() # creamos una instancia de LinearRegression
    
    # regresion_lineal.fit(xt, yt) 
    # # vemos los parámetros que ha estimado la regresión lineal
    # m = regresion_lineal.coef_
    # b = regresion_lineal.intercept_
    
    # pend = np.sqrt(10**m[0])
    # pendientes[n] = pend
    # print(f'pendiente = {pend}')
    
    
    # plt.plot(x[linf:lsup+1]+4, 10**(m*x[linf:lsup+1]+b), '-', color=color_gradients[n], lw=1.5, alpha=0.8)#, label=f'{np.sqrt(10**m[0]):.3f}**(2*t) '
    
    # plt.plot(times/np.log(N), y, '.-', ms=8, lw=2.5, label=f'D={N}',  color=color_gradients[n])
    # plt.yaxis.set_minor_locator(locmin)
    # plt.yaxis.set_minor_locator(mpl.ticker.LogLocator(numticks=9, subs="auto"))
plt.ylabel(r'$O_1$')
plt.xlabel(r'$t/\ln(D)$')
plt.xticks(times[0:int(time_lim):2])#, labels=[0,2])
plt.yscale('log')
# locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009, 0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09, 0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
# plt.set_minor_locator(locmin)
# plt.set_minor_locator(mpl.ticker.LogLocator(numticks=9, subs="auto"))
plt.xlim(0,time_lim/np.log(15000))
plt.ylim(1e-4,1.2)
# plt.xlim(-10.01,40)
# plt.grid()
plt.tight_layout()
plt.legend(loc = 'best', framealpha=0.1, fontsize=24)
plt.savefig('O1_vs_t_Ns.pdf', dpi=100)

# # #%% plot pendientes vs N
# # plt.figure(figsize=(16,8))
# # plt.plot(Ns, pendientes, '.-', ms=10, lw=1.5)
# # plt.ylabel(r'$pendientes$')
# # plt.xlabel(r'$N$')
# # plt.xticks(times[::2])
# # plt.xscale('log')
# # # plt.ylim(1e-7,1.1)
# # # plt.xlim(-0.01,21)
# # plt.grid()
# # # plt.legend(loc = 'best')
# # plt.savefig('pend_vs_N.png')