#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 20:22:02 2022

Prueba qutip numpy

@author: tomasnotenson
"""
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import blackman
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

def IPR(state, tol = 0.0001):
    if (np.linalg.norm(state)-1) > tol:
        print(np.linalg.norm(state))
    # state = state.full()
    pi = np.abs(state)**2 # calculo probabilidades
    IPR = np.sum(pi)**2/np.sum(pi**2) # calculo el IPR
    return IPR # devuelvo el IPR

def normalize(array):# normalizo valores entre 0 y 1
    return (array - min(array))/(max(array)-min(array)) # devuelvo la variable normalizada

@jit
def coherent_state_Augusto(N, q0, p0, lim_suma=4):#, cuasiq=0
    state = np.zeros((N), dtype=np.complex_) # creo el estado como array de ceros 
    # cte = np.power(2/N,4) # cte de norm. 1
    # coefq = np.exp(np.pi/(2*N)*(q0**2+p0**2)) # cte de norm. 2
    for q in range(N): #calculo el coeficiente para cada q
        coefsnu = np.zeros((2*lim_suma+1), dtype=np.complex_) # creo un array de ceros
        for i,nu in enumerate(np.arange(-lim_suma,lim_suma+1)):
            # print('q',q,', nu',nu)
            coefnu = np.exp(-np.pi/N*(nu*N-q0+q)**2-1j*2*np.pi/N*p0*(nu*N+q0/2-q)) 
            coefsnu[i] = coefnu # lo voy llenando con los términos de la sumatoria
        coef = np.sum(coefsnu) # hago la sumatoria
        state[q] = coef # agrego el coeficiente para el q fijo
    nrm = np.linalg.norm(state) # calculo la norma
    # inrm = cte*coefq
    # print('norm state', nrm) # chequeo la norma del estado
    return state/nrm # devuelvo el estado normalizado como numpy darray

@jit
def base_coherente_Augusto(N, Nstates):
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
            b = coherent_state_Augusto(N, q0, p0, lim_suma=2)
            # print(np.linalg.norm(b))
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

@jit
def ovrlp(a,b):
    adag = np.conj(a)
    res = np.inner(adag,b)
    return res

@jit
def projectr(b):
    bdag = np.conj(b)
    res = np.outer(bdag,b)
    return res

@jit
def mtrx_element(M,a,b):
    adag = np.matrix(np.conj(a))
    bp = np.matrix(b).T
    # print(adag.shape, M.shape, bp.shape)
    res = adag@M@bp
    return res[0,0]

def plot_psi2(n,q0,p0):
    psi = coherent_state_Augusto(n, q0, p0, lim_suma=4)
    M = len(psi)
    psi2 = np.abs(psi)**2
    # w = blackman(n)
    fftpsi = fft(psi)#*w)
    fftpsi2 = 1/M*(np.abs(fftpsi))**2

    # parte1 = fftpsi2[:M//2]
    # parte2 = fftpsi2[M//2:]
    # fftpsi2 = np.concatenate((parte2,parte1))

    qs = np.arange(n); 
    # fftps = fftfreq(n)
    # x1 = fftps[:M//2]
    # x2 = fftps[M//2:]
    # fftps = np.concatenate((x2,x1))

    fig, ax = plt.subplots(2, 1, figsize=(16,10))

    # plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.5, wspace=0.4)

    ax[0].plot(qs, psi2, 'r.-', label=r'$|\psi(q)|^2$')
    ax[0].vlines(q0, min(psi2), max(psi2), alpha=0.5, color='red')
    # ax[i].set_xlabel(r'$q\,\,p$')
    ax[1].plot((qs%n), fftpsi2, 'b.-', label=r'$|\psi(p)|^2$')
    ax[1].vlines((p0%n), min(psi2), max(psi2), alpha=0.5, color='blue')
    for i in range(len(ax)):
        if i==0:
            xlab=r'$q$'
            ylab = r'$|\psi(q)|^2$'
        if i==1:
            xlab=r'$p$'
            ylab = r'$|\psi(p)|^2$'
        ax[i].set_xlabel(xlab)
        ax[i].set_ylabel(ylab)
        ax[i].grid(True)

    fig.tight_layout() 
    plt.savefig(f'coherent_state_n{n}_q0{q0}_p0{p0}.png', dpi=100)
    return 

def plot_Husimi_Tomi(state):
    statedag = np.conj(state)
    n = len(state)
    
    nr=2
    if(n<80): nr=4
    elif(n<40): nr=6
    elif(n<20): nr=8
    elif(n<10): nr=10
    elif(n<6): nr=16
    elif(n<4): nr=20
      
    Nstates= np.int32(nr*n)
    nnr2 = int(nr*n/2)
    
    hus = np.zeros((Nstates,Nstates), dtype=np.complex_)
    
    for iq in range(Nstates):
        for ip in range(Nstates):
            b = coherent_state_Augusto(n, iq/nr, ip/nr)
            hus[ip, iq] = np.inner(statedag, b) # falta |.|^2
    hus = np.abs(hus)**2
    ax = sns.heatmap(hus)
    return hus
#%% plot in position and momentum space

n=2**8

q0 = int(n/2)
p0 = -5 

plot_psi2(n, q0, p0)

#%%
nqubits = 6;
N = 2**nqubits#11
# hbar = 1/(2*np.pi*N)
nr=2
if(n<80): nr=4
elif(n<40): nr=6
elif(n<20): nr=8
elif(n<10): nr=10
elif(n<6): nr=16
elif(n<4): nr=20
  
Nstates= np.int32(nr*N)#N/2)  #% numero de estados coherentes de la base
paso = N/Nstates

# armo la base coherente
start_time = time()
base = base_coherente_Augusto(N,Nstates)
# base = base_integrable(N)
print('Duration: ', time()-start_time)

# seed
np.random.seed(0)

# defino 2 estados complejos random
a=np.random.random(N)+np.random.random(N)*1j
norm=np.linalg.norm(a)
an = a/norm
# a = Qobj(a/norm)

c=np.random.random(N)+np.random.random(N)*1j
norm=np.linalg.norm(c)
cn = c/norm
# c = Qobj(c/norm)

# calculo el overlap entre a y b (directamente)
over1n = ovrlp(an,cn)
# over1 = a.overlap(c)

# print(over1n, over1)

# defino la identidad
Idenn = np.zeros([N,N], dtype=np.complex_)
# Iden = Qobj(Idenn)

# est_a = np.zeros(Nstates**2, dtype=np.complex_)
est_an = np.zeros(Nstates**2, dtype=np.complex_)
# est_c = np.zeros(Nstates**2, dtype=np.complex_)
est_cn = np.zeros(Nstates**2, dtype=np.complex_)

for q in tqdm(range(Nstates), desc='q loop'):
    for p in range(Nstates):
        # indice
        i=q+p*Nstates
        #centros

        q0=q*paso
        p0=p*paso
        # estado coherente
        # print(i, 'de ',Nstates**2)
        bn = base[:,i]
        # b = Qobj(bn)
                
        # expando los estados a y b en la base de estados coherentes
        # overlap entre el estado a y el coherente centrado en q0 y p0
        # est_a[i] = a.overlap(b)
        est_an[i] = ovrlp(bn,an)
        # est_c[i] = a.overlap(b)
        est_cn[i] = ovrlp(bn,cn)
        
        # sumo el proyector asociado a ese estado coherente
        # ident1 = b.proj()
        ident1n = projectr(bn)
        Idenn = Idenn+ident1n
        # Iden = Iden+ident1
        
# convierto arrays en Qobj
# est_a = Qobj(est_a)
# est_c = Qobj(est_c)

# calculo el overlap en la base de estados coherentes
# over3 = est_a.overlap(est_c)
over3n = ovrlp(est_an,est_cn)
# calculo las normas y el overlap tomando los elementos de matriz de la...
# ... identidad calculada a partir de los proyectores de la base coherente
# nor1=Iden.matrix_element(a,a)
nor1n=mtrx_element(Idenn,an,an)
# nor2=Iden.matrix_element(c,c)
nor2n=mtrx_element(Idenn,cn,cn)
# over=Iden.matrix_element(a,c)
overn=mtrx_element(Idenn,an,cn)

# imprimo resultados
print('nor1')
# print(nor1)
print(nor1n)
print('nor2/nor1')
# print(nor2/nor1)
print(nor2n/nor1n)
print('over/nor1')
# print(over/nor1)
print(overn/nor1n)
print('over1')
# print(over1)
print(over1n)
print('over3/nor1')
# print(over3/nor1)
print(over3n/nor1n)

#%%
Kpaso = 1
Ks = np.arange(0,10.1,Kpaso)#

norma = np.sqrt(nor1n)
IPR_means = np.zeros((len(Ks)))
rs = np.zeros((len(Ks)))

    
for k,K in tqdm(enumerate(Ks), desc='K loop'):
    IPRs = np.zeros((N))    
    U = UU(K, N)
    ev, evec = np.linalg.eig(U)
    # r = diagU_r(U)
    # rs[k] = r
    
    for j in tqdm(range(evec.shape[1]), desc='vec loop'):
        vec = np.array(evec[:,j]).flatten()
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
                # b = Qobj(base[:,i])
                b = base[:,i]
                
                # expando los estados a y b en la base de estados coherentes
                # overlap entre el estado a y el coherente centrado en q0 y p0
                # est_vec[i] = vec.overlap(b) #Iden.matrix_element(b,vec)#
                est_vec[i] = ovrlp(b,vec)#mtrx_element(Idenn,b,vec)#
                
        aux = IPR(est_vec/norma)
        IPRs[j] = aux
       
    IPR_means[k] = np.mean(IPRs)
np.savez(f'IPR_vs_Ks_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_N{N}_coherent_basis_grid{Nstates}x{Nstates}_numpy.npz', IPR_means = IPR_means, rs = rs)#integrable_K0
