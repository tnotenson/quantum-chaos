#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 14:26:35 2021

@author: tomasnotenson
"""

import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from math import factorial
from time import time
from tqdm import tqdm # customisable progressbar decorator for iterators
from numba import jit
# importing "cmath" for complex number operations
from cmath import phase
plt.rcParams.update({
"text.usetex": True,
"font.family": "sans-serif",
"font.sans-serif": ["Helvetica"], "font.size": 30})

# define usefull one body operators 
sx=sigmax()
sy=sigmay()
sz=sigmaz()
si=qeye(2)
s0=(si-sz)/2


def TFIM(N, hx, hz, J):

    # construct the spin (super)operators for each site
    s_ops = [sigmax(), sigmay(), sigmaz()]
    
    sx_list = []
    sy_list = []
    sz_list = []
    
    s_lists = [sx_list, sy_list, sz_list]
    
    for n in range(N):
        op_list = [si for m in range(N)]

        for s_op, s_list in zip(s_ops, s_lists):
            op_list[n] = s_op
            s_list.append(tensor(op_list))

    # construct the hamiltonian
    H = 0

    # energy splitting terms      
    for n in range(N):
        H += hx[n] * sx_list[n]
        H += hz[n] * sz_list[n]
        
    # interaction terms
    
    #PBC
    for n in range(N):
        H += J[n] * sz_list[n] * sz_list[(n+1)%N]
        
    #OBC
    # for n in range(N-1):
    #     H += J[n] * sz_list[n] * sz_list[n+1]
    
    return H

def fH0(N, hx, hz, sx_list, sz_list):
    
    # construct the hamiltonian
    H = 0

    # energy splitting terms
    for n in range(N):
        H += hz[n] * sz_list[n]
        H += hx[n] * sx_list[n]
   
    return H

def fH1(N, J, sz_list):
            
    # construct the hamiltonian
    H = 0
    
    # interaction terms
    
    #PBC
    for n in range(N):
        H += J[n] * sz_list[n] * sz_list[(n+1)%N]
        
    #OBC
    
    # for n in range(N-1):
    #     H += J[n] * sz_list[n] * sz_list[n+1]
    return H

def fU(N, J, hx, hz):
    
    # construct the spin (super)operators for each site
    s_ops = [sigmax(), sigmay(), sigmaz()]
    
    sx_list = []
    sy_list = []
    sz_list = []
    
    s_lists = [sx_list, sy_list, sz_list]

    for n in range(N):
        op_list = [si for m in range(N)]

        for s_op, s_list in zip(s_ops, s_lists):
            op_list[n] = s_op
            s_list.append(tensor(op_list))
    
    # # define the hamiltonians
    H0 = fH0(N, hx, hz, sx_list, sz_list)
    H1 = fH1(N, J, sz_list)
    
    U0 = propagator(H0,1).full()
    U1 = propagator(H1,1).full()

    U = U1@U0
    
    return Qobj(U)

# construct permute operator
def pij(l,i,j):
    geye=[si for k in range(l)]
    
    H=0*tensor(geye)
    g=geye.copy(); g[i]=sx;g[j]=sx; H+= tensor(g)
    g=geye.copy(); g[i]=sy;g[j]=sy; H+= tensor(g)
    g=geye.copy(); g[i]=sz;g[j]=sz; H+= tensor(g)
    
    H+=tensor(geye); return H/2

# construct parity operator
def parity(l):
    geye=[si for k in range(l)]
    
    P=tensor(geye)
    for m in range(l//2):
        P=pij(l,m,l-m-1)*P
    return P

# construct tensor product of sz operators of each site
def Sj(N, j='z'):
        s_list = []
        for n in range(N):
            op_list = []
            for m in range(N):
                op_list.append(si)
            if j == 'z':
                op_list[n] = sz
            elif j == 'x':
                op_list[n] = sx
            elif j == 'y':
                op_list[n] = sy
            s_list.append(tensor(op_list))
        return sum(s_list)

# construct Pauli matrix in direction j=x,y,z for site i=0,...,N-1
def sigmai_j(N,i,j='z'):

    op_list = [si for m in range(N)]

    if j == 'z':
        op_list[i] = sz
    elif j == 'x':
        op_list[i] = sx
    elif j == 'y':
        op_list[i] = sy
    return tensor(op_list)


# Time evolution step (Heisenberg picture)
@jit(nopython=True, parallel=True, fastmath = True)
def Evolucion_numpy(B_t, U, Udag):
    res = Udag@B_t@U# U@B_t@Udag # evolucion al revés
    return res

# 2 operator correlator dim * <B(t) A> 
@jit(nopython=True, parallel=True, fastmath = True)
def twopC_numpy_Tinf(A, B_t):
    C = np.trace(B_t@A)
    return C

# operator average dim * <A>**2 
@jit(nopython=True, parallel=True, fastmath = True)
def A_average_sqd(A, dims=1):
    res = np.trace(A)**2/dims
    return res

# squared commutator average <|[B(t), A]|**2>
@jit(nopython=True, parallel=True, fastmath = True)
def C_t_commutator_numpy_Tinf(A, B_t, N):
    com = B_t@A - A@B_t
    C_t = np.trace(com.H@com)/N
    return C_t

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

# level spacing histogram
def histo_level_spacing(ener):
    spac = np.diff(ener)
    print('espaciado',spac)
    plt.figure(figsize=(16,8))
    plt.hist(spac)
    plt.xlabel('level spacing')
    return 

# r chaometer for hamiltonian 
# @jit(nopython=True)#, parallel=True, fastmath = True)
def diagH_r(H_sub):
    ener = np.linalg.eigvals(H_sub)
    ener = np.sort(ener)
    # histo_level_spacing(ener)
    r_normed = r_chaometer(ener, plotadjusted=True) 
    return r_normed

# r chaometer for evolution operator (Floquet)
# @jit(nopython=True)#, parallel=True, fastmath = True)
def diagU_r(U_sub):
    ener = np.linalg.eigvals(U_sub)
    fases = [phase(en) for en in ener]
    fases = np.sort(fases)
    # histo_level_spacing(fases)
    r_normed = r_chaometer(fases, plotadjusted=True) 
    return r_normed

def r_thetas(N, B, J, theta, x, z, dt, pot):
     
    hx = np.sin(theta)*x*B*np.ones(N)
    hz = np.cos(theta)*z*B*np.ones(N)
    Jz = J*np.ones(N)
    
    start_time = time()
    # let's try it
    
    ######## H evolution ##########
    # H = TFIM_Nico(N, hx, hz, Jz)
    H = TFIM(N, hx, hz, Jz)
    ######## U evolution ##########
    U = fU(N, Jz, hx, hz, dt)**pot
    
    # separate symmetries
    start_time = time()

    # separate symmetries
    
    P = parity(N)
    ep, epvec = P.eigenstates()
    
    n_mone, n_one = np.unique(ep, return_counts = True)[1]
    
    C = np.column_stack([vec.data.toarray() for vec in epvec])
    Cinv = np.linalg.inv(C)
    
    ######## H evolution ##########
    
    H_par = H.transform(Cinv)
    # print(H_par)
    
    ######## U evolution ##########
    U_par = U.transform(Cinv)
    # print(U_par)
    
    ######## H evolution ##########
    H_sub = H_par.extract_states(np.arange(n_mone,n_one+n_mone))
    # print(H_sub)
    
    ######## U evolution ##########
    U_sub = U_par.extract_states(np.arange(n_mone,n_one+n_mone)).data.toarray()
    U_sub = np.matrix(U_sub)
    
    # #########################################################################
    r_normed_H = diagH_r(H_sub)
    r_normed_U = diagU_r(U_sub)
    
    # print(f"\n r-chaometer theta = {theta} --- {time() - start_time} seconds ---" )
    print(f"\n r-chaometer theta = {theta} --- {time() - start_time} seconds ---" )
    return [r_normed_H, r_normed_U]

#%%

def C2p_time(time_lim, A, B, U, Udag, Cs, N):
    # compute OTOC, O1 and O2 for each time
    for i in tqdm(range(time_lim), desc='Evolution loop'):
        
        if i==0:
            B_t = B
        else:
            # Evolution
            B_t = Evolucion_numpy(B_t, U, Udag)
            
        # dim = A.shape[0]
        
        # compute 2-point correlator with numpy
        C_t = twopC_numpy_Tinf(A, B_t)# - A_average_sqd(A, dim)
        # C_t = C_t/dim/N
        
        # print(C_t)
        Cs[i] = C_t#np.abs(C_t)
    return Cs

def Evolution2p_H_KI_Tinf(H, time_lim, N, A, B):
    
    start_time = time() 
    
    # define arrays for data storage
    Cs = np.zeros((time_lim), dtype=np.complex_)#, dtype=np.complex_)#[]
        
    # define time evolution operator
    U = propagator(H,1).full()
    U = np.matrix(U, dtype=np.complex_)
    Udag = U.H
    # print(U)
    
    Cs = C2p_time(time_lim, A, B, U, Udag, Cs, N)
        
    print(f"\nTOTAL --- {time() - start_time} seconds ---" )
    flag = '2p_H_KI_with_Tinf_state'
    return [Cs, flag]

def Evolution2p_U_KI_Tinf(U, time_lim, N, A, B):
    
    start_time = time() 
    
    # define arrays for data storage
    Cs = np.zeros((time_lim), dtype=np.complex_)#[]
    
    # define dagger floquet operator
    Udag = U.H

    Cs = C2p_time(time_lim, A, B, U, Udag, Cs, N)
    
    print(f"\nTOTAL --- {time() - start_time} seconds ---" )
    flag = '2p_U_KI_with_Tinf_state'
    return [Cs, flag]

def TFIM_2pC_chaos_parameter(N, J, x, z, time_lim, evolution='U'):
        
    # define parameters of Heisenberg chain with inclined field 
    
    hx = x*np.ones(N)#np.sin(theta)*B*np.ones(N)
    hz = z*np.ones(N)#np.cos(theta)*B*np.ones(N)
    Jz = J*np.ones(N)
    
    
    start_time = time()
    # let's try it
    if evolution=='H':
        # ######## H evolution ##########
        H = TFIM(N, hx, hz, Jz)
    elif evolution=='U':
        ######## U evolution ##########
        U = fU(N, Jz, hx, hz)
        # U = np.matrix(U.data.toarray(), dtype=np.complex_)
    
    # choose operators A and B
    A = Sj(N, j='x')#/N
    B = A
    # A = A.data.toarray()
    # i = int(N/2)
    # A = sigmai_j(N,i,j='x')
    # j = i#+1
    # B = sigmai_j(N,j,j='z')
    
    # op = 'X'
    
    # operators = '_A'+op+'_B'+op
    
    print(f"\n Create Floquet operator --- {time() - start_time} seconds ---" )
    
    # separate symmetries
    sym_time = time()
    
    # define and diagonalize parity operator 
    P = parity(N)
    ep, epvec = P.eigenstates()
    
    # print parity eigenvalues (+-1)
    print('Parity eigenvalues:\n',ep)
    
    # number of +1 or -1 values in ep
    n_mone, n_one = np.unique(ep, return_counts = True)[1]
    
    # +1 and -1 parity subspaces sizes
    print('tamaños de subespacios de par.', n_mone,'+',n_one,'=', n_mone+n_one)
    
    # change to parity basis
    # change-of-basis matrix
    C = np.column_stack([vec.data.toarray() for vec in epvec])
    Cinv = np.linalg.inv(C)
    
    # transform hamiltonian/floquet operator
    if evolution=='H':
        ######## H evolution ##########
        
        H_par = H.transform(Cinv)
        # print(H_par)
        
    elif evolution=='U':
        # ######## U evolution ##########
        U_par = U.transform(Cinv)
        # # hinton(U_par)
  
    # transform operators
    A_par = A.transform(Cinv)
    B_par = B.transform(Cinv)
    # hinton(A_par)
    
    # divide in parity subspace
    if evolution=='H':
       
        # # ######## H evolution ##########
        H_mone = H_par.extract_states(np.arange(n_mone))
        H_one = H_par.extract_states(np.arange(n_mone,n_one+n_mone))
        # print(H_sub)
    
        
    elif evolution=='U':
        # ######## U evolution ##########
        U_mone = U_par.extract_states(np.arange(n_mone)).data.toarray()
        U_one = U_par.extract_states(np.arange(n_mone, n_mone+n_one)).data.toarray()
        U_mone = np.matrix(U_mone, dtype=np.complex_)
        U_one = np.matrix(U_one, dtype=np.complex_)
        
 
    A_mone = A_par.extract_states(np.arange(n_mone)).data.toarray()
    A_one = A_par.extract_states(np.arange(n_mone, n_mone+n_one)).data.toarray()
    
    B_mone = B_par.extract_states(np.arange(n_mone)).data.toarray()
    B_one = B_par.extract_states(np.arange(n_mone, n_mone+n_one)).data.toarray()
    # A_sub = np.matrix(A_sub, dtype=np.complex_)
    print(f"\n Separate parity eigenstates --- {time() - sym_time} seconds ---" )
    
    # compute r chaometer 
    # r_normed_H = diagH_r(H_sub)
    # r_normed_U = diagU_r(U_sub)
    
    evol_time = time()
    #
    
    # compute 2 operator correlator
    if evolution=='H':
                
        Cs_mone, _ = Evolution2p_H_KI_Tinf(H_mone, time_lim, N, A_mone, B_mone)
        Cs_one, _ = Evolution2p_H_KI_Tinf(H_one, time_lim, N, A_one, B_one)
        
    elif evolution=='U':
            
        Cs_mone, _ = Evolution2p_U_KI_Tinf(U_mone, time_lim, N, A_mone, B_mone)
        Cs_one, _ = Evolution2p_U_KI_Tinf(U_one, time_lim, N, A_one, B_one)
        # np.savez('2pC_'+flag+f'_time_lim{time_lim}_J{J:.2f}_hx{x:.2f}_hz{z:.2f}_basis_size{N}'+operators+'.npz', Cs=Cs)
    
    print(f"\n Evolution 2pC --- {time() - evol_time} seconds ---" )
    print(f"\n TOTAL --- {time() - start_time} seconds ---" )
    return [Cs_mone, Cs_one]#, r_normed_U# [Cs] #

#%% Calcular los correladores para un valor de theta y de B
# define parameters of Heisenberg chain with inclined field 
N = 11
J = 1

time_lim = 51
times = np.arange(0,time_lim)

# tomo los valores de Prosen
hx = 1.4
hz = 1.4

alpha = 2*J
beta = 2*hx

DL = ((max(np.abs(np.cos(alpha)),np.abs(np.cos(beta)))-(np.cos(beta))**2))/(np.sin(beta)**2)
print(DL)

B = np.sqrt(hx**2 + hz**2)
if hz == 0:
    theta = np.pi/2
else:
    theta = np.arctan(hx/hz)
print('B=',B,'\ntheta=',theta)
#%%
Cs_mone, Cs_one = TFIM_2pC_chaos_parameter(N, J, hx, hz, time_lim)
# Cs = TFIM_2pC_chaos_parameter(N, B, J, theta, time_lim)[0]

Cs = np.abs((Cs_mone + Cs_one)/2**N/N)

# Cs_numpy = Cs
# flag = 'Evol_alreves_2p_KI_with_Tinf_state_'
# opi = str(int(N/2))
# opj = str(int(N/2))
opA = 'X'#'X_'+opi
opB = opA#'Z_'+opj
paridad = 'par'
operators = '_A'+opA+'_B'+opB
BC = 'PBC'#'OBC'
flag = 'propagator_2p_KI_with_Tinf_state_'
np.savez('2pC_'+flag+f'_time_lim{time_lim}_J{J:.2f}_hx{hx:.2f}_hz{hz:.2f}_basis_size{N}'+operators+'.npz', Cs=Cs)
#%% Comparación de contribuciones Cs por paridad

# Cs_mone = Cs_mone/2**N/N
# Cs_one = Cs_one/2**N/N

# plt.figure(figsize=(16,8))

# y_Cs_U_mone = np.log10(Cs_mone/Cs_mone[0])
# y_Cs_U_one = np.log10(Cs_one/Cs_one[0])

# yfit_mone = np.log(Cs_mone)
# yfit_one = np.log(Cs_one)

# tmin = 0
# tmax = 16

# xs = times[tmin:tmax]
# yp_mone = yfit_mone[tmin:tmax]
# yp_one = yfit_one[tmin:tmax]

# coefp_mone = np.polyfit(xs,yp_mone,1)
# coefp_one = np.polyfit(xs,yp_one,1)

# poly1d_fn_p_mone = np.poly1d(coefp_mone) #[b,m]
# poly1d_fn_p_one = np.poly1d(coefp_one) #[b,m]

# plt.plot(times, np.log10(1/4*np.exp(-times/6)), '-.k', lw=1, label='0.25exp(-t/6)')
# plt.text(20, -.75, r'$m_2^{impar}=$'+f'{poly1d_fn_p_mone[1].real:.2}',
#         verticalalignment='bottom', horizontalalignment='right',
#         color='red', fontsize=15)
# plt.text(20, -1.25, r'$m_2^{par}=$'+f'{poly1d_fn_p_one[1].real:.2}',
#         verticalalignment='bottom', horizontalalignment='right',
#         color='blue', fontsize=15)
# plt.ylim(-4,0)
# plt.ylabel(r'$log_{10}(C(t))$')
# plt.plot(times,y_Cs_U_mone, 'o-r', ms=1, lw=2, label='impar', alpha=0.8)
# plt.plot(times,y_Cs_U_one, 'o-b', ms=1, lw=2, label='par', alpha=0.8)
# # plt.plot(xs, poly1d_fn_p(xs), '--r', lw=1)


# print('impar',Cs_mone[0])
# print('par',Cs_one[0])

# plt.xlabel(r'$t$')

# plt.xlim(-0.2,50.2)
# plt.grid()
# plt.legend(loc='best')

# opA = 'X'#+opi
# opB = opA#'X'
# paridad = 'par'
# operators = '_A'+opA+'_B'+opB
# BC = 'PBC'
# flag = 'Comparacion_contr_paridad_2p_KI_with_Tinf_state_'
# plt.savefig(flag+BC+f'_time_lim{time_lim}_J{J:.2f}_hx{hx:.2f}_hz{hz:.2f}_theta{theta:.2f}_basis_size{N}'+paridad+operators+'_suma_de_contr_subesp_paridad.png', dpi=80)


#%% Compare values with Tomaz Prosen paper

plt.figure(figsize=(16,8))

if hx == hz:
    
    y_C2_U = np.log10(Cs)
    yfit = np.log(Cs)
    
    tmin = 0
    tmax = 16
    
    xs = times[tmin:tmax]
    y = y_C2_U[tmin:tmax]
    yp = yfit[tmin:tmax]
    
    coefp = np.polyfit(xs,yp,1)
    poly1d_fn_p = np.poly1d(coefp) #[b,m]
    mp, bp = poly1d_fn_p
    
    coef10 = np.polyfit(xs,y,1)
    poly1d_fn10 = np.poly1d(coef10) #[b,m]
    m, b = poly1d_fn10

    # m = -0.0723824
    print(r'$C_2$',poly1d_fn_p[1])
    
    plt.plot(xs, np.log10(np.exp(mp*xs+bp)), '-.k', lw=1, label=f'D exp(-t/{1/np.abs(mp):.0f})')
    plt.plot(xs, m*xs+b, '-.k', lw=1, label=f'C 10**(-t/{1/np.abs(m):.0f})')
    plt.text(20, -1, r'$m_2=$'+f'{poly1d_fn_p[1].real:.2}',
            verticalalignment='bottom', horizontalalignment='right',
            color='red', fontsize=15)
    # plt.ylim(-4,0)
    plt.ylabel(r'$log_{10}(C(t))$')
    # plt.plot(times,np.log10((Cs_mone+Cs_one)/2**N/N), 'o-c', ms=1, lw=2, label='$L=$'+f'{N}', alpha=0.8)
else:
    y_C2_U = Cs
    
    if hz == 0:
        DL = 0.485
    elif hz == 0.4:
        DL = 0.293
    
    plt.plot(times, DL*np.ones(time_lim), '-.k', lw=2, label=r'$D/L = $'+f'{DL:.3f}')
    plt.ylabel(r'$C(t)$')
plt.plot(times,y_C2_U, 'o-g', ms=1, lw=2, label='$L=$'+f'{N}', alpha=0.8)
# plt.plot(xs, poly1d_fn_p(xs), '--r', lw=1)


# check t=0 correlator value 
print(Cs[0])

plt.xlabel(r'$t$')

plt.xlim(-0,50)
plt.grid()
plt.legend(loc='best')

plt.savefig(flag+BC+f'_time_lim{time_lim}_J{J:.2f}_hx{hx:.2f}_hz{hz:.2f}_theta{theta:.2f}_basis_size{N}'+paridad+operators+'_suma_de_contr_subesp_paridad.png', dpi=80)
#%% another check
if hz == 0.4:
    
    yplot = np.log10(np.abs(Cs-DL))#np.abs(Cs-DM)#
    
    yfit = np.log(np.abs(Cs-DL))
    
    tmin = 0
    tmax = 50
    
    xs = times[tmin:tmax]
    yp = yfit[tmin:tmax]
    
    coefp = np.polyfit(xs,yp,1)
    poly1d_fn_p = np.poly1d(coefp) #[b,m]
    
    pendiente = poly1d_fn_p[1]
    oo = poly1d_fn_p[0]
    
    print(r'$C_2$',pendiente)
    
    
    plt.figure(figsize=(16,8))
    plt.plot(times, pendiente*times, '-.k', lw=2, label=r'$D/L = $'+f'{DL:.3f}')
    plt.ylabel(r'$log_{10}(C(t)-D/L)$')
    plt.plot(times, yplot, 'o-g', ms=1, lw=2, label='$L=$'+f'{N}', alpha=0.8)
    plt.xlabel(r'$t$')
    plt.xlim(-0,50)
    plt.ylim(-4,0)
    plt.grid()
    plt.legend(loc='best')
    plt.savefig('inset_1b_'+flag+BC+f'_time_lim{time_lim}_J{J:.2f}_hx{hx:.2f}_hz{hz:.2f}_theta{theta:.2f}_basis_size{N}'+paridad+operators+'_suma_de_contr_subesp_paridad.png', dpi=80)
#%%
# plt.figure(figsize=(16,8))

# if hx == hz:
#     y_C2_U = np.log10(Cs)

#     yfit = np.log(Cs)
    
#     tmin = 0
#     tmax = 16
    
#     xs = times[tmin:tmax]
#     yp = yfit[tmin:tmax]
    
#     coefp = np.polyfit(xs,yp,1)
#     poly1d_fn_p = np.poly1d(coefp) #[b,m]
    
#     print(r'$C_2$',poly1d_fn_p[1])
    
#     plt.plot(times, np.log10(1/4*np.exp(-times/6)), '-.k', lw=2, label='0.25exp(-t/6)')
#     plt.text(20, -1, r'$m_2=$'+f'{poly1d_fn_p[1].real:.2}',
#             verticalalignment='bottom', horizontalalignment='right',
#             color='red', fontsize=15)
#     plt.ylim(-4,0)
#     plt.ylabel(r'$log_{10}(C(t))$')
# else:
#     y_C2_U = Cs
    
#     if hz == 0:
#         DL = 0.485
#     elif hz == 0.4:
#         DL = 0.293
    
#     plt.plot(times, np.mean(y_C2_U)*np.ones(time_lim), '-.k', lw=2, label=r'$D/L = $'+f'{DL:.3f}')
#     plt.ylabel(r'$C(t)$')
# plt.plot(times,y_C2_U, 'o:k', ms=1, lw=2, label='$L=$'+f'{N}', alpha=0.8)
# # plt.plot(xs, poly1d_fn_p(xs), '--r', lw=1)


# print(Cs[0])

# plt.xlabel(r'$t$')

# plt.xlim(-0.2,50.2)
# plt.grid()
# plt.legend(loc='best')
# # opi = str(int(N/2))
# opA = 'X'#+opi
# opB = opA#'X'
# paridad = 'par'
# operators = '_A'+opA+'_B'+opB
# BC = 'PBC'
# flag = '2p_KI_with_Tinf_state_'
# plt.savefig(flag+BC+f'_time_lim{time_lim}_J{J:.2f}_hx{hx:.2f}_hz{hz:.2f}_theta{theta:.2f}_basis_size{N}'+paridad+operators+'_suma_de_contr_subesp_paridad.png', dpi=80)


