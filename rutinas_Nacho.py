#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 13:47:15 2022

@author: tomasnotenson
"""
from scipy.fft import fft, ifft, fftfreq, fft2, ifft2
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from qutip import *
from time import time 
from tqdm import tqdm # customisable progressbar decorator for iterators
from cmath import phase
from scipy.stats import skew
import seaborn as sns; sns.set_theme()
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

@jit
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
def coherent_state_Augusto(N, q0, p0, lim_suma=4, norma=True):#, cuasiq=0
    state = np.zeros((N), dtype=np.complex_) # creo el estado como array de ceros 
    cte = np.power(2/N,4) # cte de norm. 1
    coefq = np.exp(np.pi/(2*N)*(q0**2+p0**2)) # cte de norm. 2
    for q in range(N): #calculo el coeficiente para cada q
        coefsnu = np.zeros((2*lim_suma+1), dtype=np.complex_) # creo un array de ceros
        for i,nu in enumerate(np.arange(-lim_suma,lim_suma+1)):
            # print('q',q,', nu',nu)
            coefnu = np.exp(-np.pi/N*(nu*N-(q0-q))**2-1j*2*np.pi/N*p0*(nu*N-q+q0/2)) 
            coefsnu[i] = coefnu # lo voy llenando con los términos de la sumatoria
        coef = np.sum(coefsnu) # hago la sumatoria
        state[q] = coef # agrego el coeficiente para el q fijo
    if norma:
        nrm = np.linalg.norm(state) # calculo la norma
        # inrm = cte*coefq
        # print('norm state', nrm) # chequeo la norma del estado
        return state/nrm # devuelvo el estado normalizado como numpy darray
    else:
        return state

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
            b = Qobj(coherent_state_Augusto(N, q0, p0, lim_suma=4))
            # print(np.linalg.norm(b))
            base[:,i] = np.array(b).flatten()
    return base

@jit
def base_integrable(N, K=0):
    U = UU(K, N)
    e, evec = np.linalg.eig(U)
    return evec

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

def coherent_coef(q, N, q0, p0, lim_suma=4):
    coefsnu = np.zeros((2*lim_suma+1), dtype=np.complex_) # creo un array de ceros
    for i,nu in enumerate(np.arange(-lim_suma,lim_suma+1)):
        # print('q',q,', nu',nu)
        coefnu = np.exp(-np.pi/N*(nu*N-(q0-q))**2-1j*2*np.pi/N*p0*(nu*N-q+q0/2)) 
        coefsnu[i] = coefnu # lo voy llenando con los términos de la sumatoria
    coef = np.sum(coefsnu) # hago la sumatoria
    return coef

def coherent_Fcoef(p, N, q0, p0, lim_suma=4):
    b = coherent_state_Augusto(N, q0, p0, lim_suma=4, norma=False)
    M = len(b)
    bf = fft(b)*1/np.sqrt(M)
    return bf[p]
    
    
def state2kirk(phi,ndim,tip=0,tiq=0):
    
# !
# ! constructs the kirkwood representation of the pure state vector phi
# !
# ! on entry phi(j)=<j|phi> in the coordinate rep. (unchanged on exit)
# ! on exit rho(k,n) containd the Kirkwood rep. rho(k,n)=<k|rho|n>/<k|n>
# ! normalization is such that sum_{k,n}=1
    n = len(phi)
    # print('n',n)
    # assert n<15000, 'n too big' 
      
    cunit=2*1j*np.pi/n
    
    # work = np.zeros(n, dtype=np.complex_)

    work = phi*np.exp(-cunit*tip*np.arange(n))
        
    M = len(work)
    workf = fft(work)*1/np.sqrt(M)
    # workf = workf#*2.0/M#[:M//2]
    
    # print('work',len(workf))
    # print('phi',len(phi))
    rho = np.zeros((ndim, ndim), dtype=np.complex_)
    for k in range(n):
       for i in range(n):
           # print('k,i',k,i)
           rho[k,i]=workf[k]*np.conj(phi[i])*np.exp(cunit*i*(k+tip))
           # acá me queda la duda si tengo que usar workf (fft) o work
    # !rho[k,i]=<k|phi><phi|i>/<k|i>
    # print('termina hus')
    return rho

# @jit
def kirk2hus(n,rho):
# ! assumes workfft has been initialized by prop_init
# ! computes the Husimi distribution from the Kirkwood array
# ! on input rho is the kirkwood matrix matrix  <k|rho|n>/<k/n> (unchanged)
# ! on output hus is the real array <p,q|rho|p,q> discretized
# ! on a phase space grid (n*nr) x (n*nr)
# ! nr (even) is chosen by the program to provide nice smooth plots
# ! and passed on to the plotting program
# ! for N>50 nr=2 is appropriate
# ! when rho is a pure state this is the Husimi distribution

    hus = np.zeros((n,n), dtype=np.complex_)
    aim = 1j*2*np.pi/n    #!2*i*pi/n
      
# !      if(n>ndim+1)stop 'dimension of map too large in kirk2hus'
    nr=2
    if(n<80): nr=4
    elif(n<40): nr=6
    elif(n<20): nr=8
    elif(n<10): nr=10
    elif(n<6): nr=16
    elif(n<4): nr=20
    # elif(n*nr>nhus):
        # print('nhus=',nhus)
        # print('n*nr=',n*nr)
    # assert n*nr<nhus, f'dimension of hus too large in kirk2hus n*nr={n*nr} >= nhus={nhus}'
    
    nnr2=n*nr/2
    for ik in range(n):
        for ip in range(n):
            hus[ik,ip] = rho[ik,ip]

    M = len(hus)
    husf = ifft2(hus)*M
    # husf = husf#*2/M
      
# !  hus now contains the N*N generating function
    hus = np.zeros((n*nr, n*nr), dtype=np.complex_)
    for iq in range(n*nr):
        for ip in range(n*nr):
            hus[iq,ip]=husf[iq%n,ip%n]
    # print('\n\npaso el primer loop de n*nr')
    for iq in range(n*nr):
        for ip in range(n*nr):
            exp1 = np.exp(-0.5*np.pi/n*((nnr2-iq)**2+(nnr2-ip)**2))
            exp2 = np.exp(0.5*aim*(((ip-nnr2)*(iq-nnr2))%(2*n)))
            hus[iq,ip]=hus[iq,ip]*exp1*exp2

# !  hus now contains the p,q fourier components of the husimi function
    M = len(hus)
    husfif = fft2(hus)*1/M
    # husfif = husfif#*2/M
    for ip in range(n*nr):
        for iq in range(n*nr):
            husfif[ip,iq]=husfif[ip,iq]*(-1)**(ip+iq)/(n*n*nr)    #!normalization
    # print('algo')
# !  hus is the husimi on a grid refined by nr
# ! notice that first index is momentum and second is coordinate!
    return husfif

def kirk2hus_Tomi(n, rho):
    nr=2
    if(n<80): nr=4
    elif(n<40): nr=6
    elif(n<20): nr=8
    elif(n<10): nr=10
    elif(n<6): nr=16
    elif(n<4): nr=20
      
     
    hus = np.zeros((n*nr, n*nr), dtype=np.complex_)
    for iq in range(n*nr):
        for ip in range(n*nr):
            b = coherent_state_Augusto(N, q0, p0, lim_suma=4, norma=False)
            M = len(b)
            bf = fft(b)*1/np.sqrt(M)      
            
            hus[iq,ip]=mtrx_element(rho,bf,b)
   
    return hus

def plot_Husimi(state):
    n = len(state)
    print(n)
    # nhus = int(5*n)
    # print('llego')
    rho = state2kirk(state, n)
    hus = kirk2hus(n, rho)
    hus /= np.sum(np.abs(hus))
    hus = np.abs(hus)#**2
    ax = sns.heatmap(hus)
    return 

def flattenear(vec):
    return np.array(vec).flatten()

# @jit
def IPR_Husimi(state, tol = 0.0001):
    # print('entra')
    n = len(state)
    # nhus = int(5*n)
    # print('llego')
    rho = state2kirk(state, n)
    hus = kirk2hus(n, rho)
    hus /= np.sum(np.abs(hus))
    aux = IPR(np.abs(hus), tol)
    return aux
#%%
nqubits = 6;
N = 2**nqubits#11
# hbar = 1/(2*np.pi*N)
nr=2
if(N<80): nr=4
elif(N<40): nr=6
elif(N<20): nr=8
elif(N<10): nr=10
elif(N<6): nr=16
elif(N<4): nr=20
  
Nstates= np.int32(nr*N)#N/2)  #% numero de estados coherentes de la base

# # armo la base coherente
# start_time = time()
# base = base_coherente_Augusto(N,Nstates)
# # base = base_integrable(N)
# print('Duration: ', time()-start_time)

# # seed
# np.random.seed(0)

# # defino 2 estados complejos random
# a=np.random.random(N)+np.random.random(N)*1j
# norm=np.linalg.norm(a)
# an = a/norm
# # a = Qobj(a/norm)

# # defino la identidad
# Idenn = np.zeros([N,N], dtype=np.complex_)
# # Iden = Qobj(Idenn)

# # est_a = np.zeros(Nstates**2, dtype=np.complex_)
# est_an = np.zeros(Nstates**2, dtype=np.complex_)

# for q in tqdm(range(Nstates), desc='q loop'):
#     for p in range(Nstates):
#         # indice
#         i=q+p*Nstates
#         #centros

#         q0=q*paso
#         p0=p*paso
#         # estado coherente
#         # print(i, 'de ',Nstates**2)
#         bn = base[:,i]
#         # b = Qobj(bn)
                
#         # expando los estados a y b en la base de estados coherentes
#         # overlap entre el estado a y el coherente centrado en q0 y p0
#         # est_a[i] = a.overlap(b)
#         est_an[i] = ovrlp(an,bn)

        
#         # sumo el proyector asociado a ese estado coherente
#         # ident1 = b.proj()
#         ident1n = projectr(bn)
#         Idenn = Idenn+ident1n
#         # Iden = Iden+ident1


# # calculo las normas y el overlap tomando los elementos de matriz de la...
# # ... identidad calculada a partir de los proyectores de la base coherente
# # nor1=Iden.matrix_element(a,a)
# nor1n=mtrx_element(Idenn,an,an)
#%%
plt.figure(figsize=(10,10))
plot_Husimi_Tomi(state)
plt.savefig('heatmap_Husimi_autoestado_Tomi.png', dpi=80)

plt.figure(figsize=(10,10))
plot_Husimi(state)
plt.savefig('heatmap_Husimi_autoestado_Nacho.png', dpi=80)

#%%
Kpaso = 1
Ks = np.arange(0,10.1,Kpaso)#

# norma = np.sqrt(nor1n)
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
        # est_vec = np.zeros(Nstates**2, dtype=np.complex_)
        # for q in range(Nstates):
        #     for p in range(Nstates):
        #         # indice
        #         i=q+p*Nstates
        #         #centros

        #         q0=q*paso
        #         p0=p*paso
        #         # estado coherente
        #         # print(i, 'de ',Nstates**2)
        #         # b = Qobj(base[:,i])
        #         b = base[:,i]
                
        #         # expando los estados a y b en la base de estados coherentes
        #         # overlap entre el estado a y el coherente centrado en q0 y p0
        #         # est_vec[i] = vec.overlap(b) #Iden.matrix_element(b,vec)#
        #         est_vec[i] = ovrlp(b,vec)#mtrx_element(Idenn,b,vec)#
                
        # aux = IPR_Husimi(est_vec/norma)
        # print('\nvec',vec.shape)
        # print('\n\n\n\n\n')
        aux = IPR_Husimi(vec)
        IPRs[j] = aux
       
    IPR_means[k] = np.mean(IPRs)
np.savez(f'IPR_Husimi_vs_Ks_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_N{N}_coherent_basis_grid{Nstates}x{Nstates}_numpy.npz', IPR_means = IPR_means, rs = rs)#integrable_K0
#%%
y_IPR = normalize(IPR_means)

plt.figure(figsize=(16,8))
plt.title(f'IPR rutinas Nacho')
plt.plot(Ks, y_IPR, '-b', lw=1.5)
# plt.plot(Kx, r_normed, '-r', lw=1.5, label='r')
# plt.plot(Ks, 1-r_one, '-g', lw=1.5, label='1-r paridad +1')
plt.xlabel(r'$K$')
plt.ylabel('IPR')
# plt.ylim(0,1)
# plt.xlim(-0.2,max(times)+0.2)
plt.grid(True)
# plt.legend(loc = 'best')