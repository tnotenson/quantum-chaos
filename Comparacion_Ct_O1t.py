#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 11:20:00 2022

@author: tomasnotenson
"""
import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from time import time
from tqdm import tqdm # customisable progressbar decorator for iterators
from numba import jit
# importing "cmath" for complex number operations
from cmath import phase
plt.rcParams.update({
"text.usetex": True,
"font.family": "sans-serif",
"font.sans-serif": ["Helvetica"], "font.size": 12})

# define usefull one body operators 
sx=sigmax()
sy=sigmay()
sz=sigmaz()
si=qeye(2)
s0=(si-sz)/2

def fH0(N, hx, hz, sx_list, sz_list):
    
    # construct the hamiltonian
    H = 0

    # energy splitting terms
    for n in range(N):
        H += hz[n] * sz_list[n]
        
    for n in range(N):
        H += hx[n] * sx_list[n]
   
    return H

def fH1(N, J, sz_list):
            
    # construct the hamiltonian
    H = 0
    
    # interaction terms
    for n in range(N-1):
        H += J[n] * sz_list[n] * sz_list[(n+1)]
    
    return H

def fU(N, J, hx, hz):
    
    # construct the spin (super)operators for each site
    s_ops = [sigmax(), sigmay(), sigmaz()]
    
    sx_list = []
    sy_list = []
    sz_list = []
    
    s_lists = [sx_list, sy_list, sz_list]

    for n in range(N):
        op_list = [qeye(2) for m in range(N)]

        for s_op, s_list in zip(s_ops, s_lists):
            op_list[n] = s_op
            s_list.append(tensor(op_list))
    
    # # define the hamiltonians
    H0 = fH0(N, hx, hz, sx_list, sz_list)
    H1 = fH1(N, J, sz_list)
    
    # define the floquet operator
    U = (-1j*H1).expm()*(-1j*H0).expm()
    
    return U

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

def sigmai_j(N,i,j='z'):

    op_list = [si for m in range(N)]

    if j == 'z':
        op_list[i] = sz
    elif j == 'x':
        op_list[i] = sx
    elif j == 'y':
        op_list[i] = sy
    return tensor(op_list)

def can_bas(N,i):
    e = np.zeros(N)
    e[i] = 1.0
    return e

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
def twopC_numpy_Tinf(A, B_t):
    C = np.trace(B_t@A)
    return C

@jit(nopython=True, parallel=True, fastmath = True)
def A_average(A, dims):
    res = np.trace(A)**2/dims
    return res

@jit(nopython=True, parallel=True, fastmath = True)
def C_t_commutator_numpy_Tinf(A, B_t, dims):
    com = B_t@A - A@B_t
    C_t = np.trace(com.H@com)/dims
    return C_t

def TFIM(N, hx, hz, J):

    # construct the spin (super)operators for each site
    s_ops = [sigmax(), sigmay(), sigmaz()]
    
    sx_list = []
    sy_list = []
    sz_list = []
    
    s_lists = [sx_list, sy_list, sz_list]
    
    for n in range(N):
        op_list = [qeye(2) for m in range(N)]

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
    for n in range(N-1):
        H += J[n] * sz_list[n] * sz_list[n+1]
    
    return H

def Evolution4p_and_2p_H_KI_Tinf(H, time_lim, N, A, B):
    
    start_time = time() 
    
    # define arrays for data storage
    O1s, O2s, Cs = np.zeros((time_lim), dtype=np.complex_),np.zeros((time_lim), dtype=np.complex_),np.zeros((time_lim), dtype=np.complex_)
    C2 = np.zeros((time_lim), dtype=np.complex_)#[]
    # define time evolution operator
    U = (-1j*H).expm().data.toarray()
    U = np.matrix(U, dtype=np.complex_)
    Udag = U.H
    # print(U.shape)
    # print(Udag.shape)
    # print(B.shape)
    
    # compute OTOC, O1 and O2 for each time
    for i in tqdm(range(time_lim), desc='Evolution loop'):
        
        if i==0:
            B_t = B
        else:

            # qutip evolution
            # B_t = B_t.transform(U.dag())
            # numpy evolution
            B_t = Evolucion_numpy(B_t, U, Udag)
            
        dim = A.shape[0]        
        # compute 4-point correlator with qutip
        # O1 = (B_t*A*B_t*A).tr()
        # O2 = (B_t**2*A**2).tr()
                    
        # C_t = -2*( O1 - O2 )/dim
        
        # compute 4-point correlator with numpy
        O1 = O1_numpy_Tinf(A, B_t)
        O2 = O2_numpy_Tinf(A, B_t)
        
        C_t = -2*(O1 - O2)/dim
        
        C_2 = twopC_numpy_Tinf(A, B_t) - A_average(A, dim)
        C_2 = C_2/dim
        # print(C_t)
        # store data
        O1s[i] = np.abs(O1); O2s[i] = np.abs(O2); Cs[i] = np.abs(C_t); C2[i] = np.abs(C_2)

        
    print(f"\nTOTAL --- {time() - start_time} seconds ---" )
    flag = '4p_and_2p_H_KI_with_Tinf_state'
    return [O1s, O2s, Cs, C2, flag]

def Evolution4p_and_2p_U_KI_Tinf(U, time_lim, N, A, B):
    
    start_time = time() 
    
    # define arrays for data storage
    O1s, O2s, Cs = np.zeros((time_lim), dtype=np.complex_),np.zeros((time_lim), dtype=np.complex_),np.zeros((time_lim), dtype=np.complex_)
    C2 = np.zeros((time_lim), dtype=np.complex_)#[]
    # define floquet operator
#    U = fU(N, J, hx, hz, theta)
    Udag = U.H


    # compute OTOC, O1 and O2 for each time
    for i in tqdm(range(time_lim), desc='Evolution loop'):
        
        if i==0:
            B_t = B
        else:

            # qutip evolution
            # B_t = B_t.transform(U.dag())
            # numpy evolution
            B_t = Evolucion_numpy(B_t, U, Udag)
            
        dim = A.shape[0]
        # compute 4-point correlator with qutip
        # O1 = (B_t*A*B_t*A).tr()
        # O2 = (B_t**2*A**2).tr()
        
        # compute 4-point correlator with numpy
        O1 = O1_numpy_Tinf(A, B_t)
        O2 = O2_numpy_Tinf(A, B_t)
        
        C_t = -2*(O1 - O2)/dim
        
        C_2 = twopC_numpy_Tinf(A, B_t) - A_average(A, dim)
        C_2 = C_2/dim
        # print(C_t)
        # store data
        O1s[i] = np.abs(O1); O2s[i] = np.abs(O2); Cs[i] = np.abs(C_t); C2[i] = np.abs(C_2)

        # print(C_t)
        # store data

        
    print(f"\nTOTAL --- {time() - start_time} seconds ---" )
    flag = '4p_and_2p_U_KI_with_Tinf_state'
    return [O1s, O2s, Cs, C2, flag]

# @jit(nopython=True)#, parallel=True, fastmath = True)
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

def histo_level_spacing(ener):
    # ener = np.sort(ener)
    spac = np.diff(ener)
    # print('espaciado',spac)
    plt.figure(figsize=(16,8))
    plt.hist(spac)#, normed=True)#, bins='auto')
    plt.xlabel('level spacing')
    return 

# @jit(nopython=True)#, parallel=True, fastmath = True)
def diagH_r(H_sub):
    ener = np.linalg.eigvals(H_sub)
    ener = np.sort(ener)
    histo_level_spacing(ener)
    r_normed = r_chaometer(ener, plotadjusted=True) 
    return r_normed

# @jit(nopython=True)#, parallel=True, fastmath = True)
def diagU_r(U_sub):
    ener = np.linalg.eigvals(U_sub)
    # ener = np.sort(ener)
    fases = [phase(en) for en in ener]
    fases = np.sort(fases)
    histo_level_spacing(fases)
    r_normed = r_chaometer(fases, plotadjusted=True) 
    return r_normed

def TFIM_O1_and_C_chaos_parameter(N, B, J, theta, x, z, time_lim):
        
    # define parameters of Heisenberg chain with inclined field 
    
    hx = np.sin(theta)*x*B*np.ones(N)
    hz = np.cos(theta)*z*B*np.ones(N)
    Jz = J*np.ones(N)
    
    start_time = time()
    # let's try it
    
    ######## H evolution ##########
    H = TFIM(N, hx, hz, Jz)
    ######## U evolution ##########
    U = fU(N, Jz, hx, hz)
    
    A = Sj(N, j='z')#/N
    B = Sj(N, j='z')
    
    # opA = 'X'
    # opB = 'X'
    # operators = '_A'+opA+'_B'+opB
    
    print(f"\n Create Floquet operator --- {time() - start_time} seconds ---" )
    
    # separate symmetries
    start_time = time()
    
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
    
    A_par = A.transform(Cinv)
    B_par = B.transform(Cinv)
    
    ######## H evolution ##########
    H_sub = H_par.extract_states(np.arange(n_mone,n_one+n_mone))
    # print(H_sub)
    
    ######## U evolution ##########
    U_sub = U_par.extract_states(np.arange(n_mone,n_one+n_mone)).data.toarray()
    U_sub = np.matrix(U_sub)
    
    A_sub = A_par.extract_states(np.arange(n_mone,n_one+n_mone)).data.toarray()
    B_sub = B_par.extract_states(np.arange(n_mone,n_one+n_mone)).data.toarray()
    print(f"\n Separate parity eigenstates --- {time() - start_time} seconds ---" )
    
    r_normed_H = diagH_r(H_sub)
    r_normed_U = diagU_r(U_sub)
    
    start_time = time()
    #
    
    O1s_U, O2s_U, Cs_U, C2_U, flag = Evolution4p_and_2p_U_KI_Tinf(U_sub, time_lim, N, A_sub, B_sub)
    # np.savez(flag+f'_time_lim{time_lim}_J{J:.2f}_hx{x:.2f}_hz{z:.2f}_theta{theta:.2f}_basis_size{N}'+operators+'.npz', O1s=O1s_U, O2s=O2s_U, Cs=Cs_U)
    
    O1s_H, O2s_H, Cs_H, C2_H, flag = Evolution4p_and_2p_H_KI_Tinf(H_sub, time_lim, N, A_sub, B_sub)
    # np.savez(flag+f'_time_lim{time_lim}_J{J:.2f}_hx{x:.2f}_hz{z:.2f}_theta{theta:.2f}_basis_size{N}'+operators+'.npz', O1s=O1s_H, O2s=O2s_H, Cs=Cs_H)
    
    
    print(f"\n Evolution 4p and 2pC --- {time() - start_time} seconds ---" )
    
    return [[O1s_H, O2s_H, Cs_H, C2_H], [O1s_U, O2s_U, Cs_U, C2_U], r_normed_H, r_normed_U]

#%% define operators

N = 12
J = 1

x = 1
z = 1

time_lim = 50

points = 100
#%% Calcular los correladores para un valor de theta y de B
# tomo los valores de Prosen
hx = 1.4
hz = 1.4
B = np.sqrt(hx**2 + hz**2)
theta = np.arctan(hz/hx)
print('B=',B,'\ntheta=',theta)
[O1s_H, O2s_H, Cs_H, C2_H], [O1s_U, O2s_U, Cs_U, C2_U], r_normed_H, r_normed_U = TFIM_O1_and_C_chaos_parameter(N, B, J, theta, x, z, time_lim)
#%% Calcular los correladores para varios valores de theta y uno de B

thetas = np.linspace(0.01, np.pi/2+0.01,points)

# O1_H_array, O2_H_array, Cs_H_array, C2_H_array = np.zeros((time_lim, points)), np.zeros((time_lim, points)), np.zeros((time_lim, points)), np.zeros((time_lim, points))
# O1_U_array, O2_U_array, Cs_U_array, C2_U_array = np.zeros((time_lim, points)), np.zeros((time_lim, points)), np.zeros((time_lim, points)), np.zeros((time_lim, points))

# rs_H = np.zeros((points))
# rs_U = np.zeros((points))

# for i,theta in enumerate(thetas):
#     [O1s_H, O2s_H, Cs_H, C2_H], [O1s_U, O2s_U, Cs_U, C2_U], r_normed_H, r_normed_U = TFIM_O1_and_C_chaos_parameter(N, B, J, theta, x, z, time_lim)
#     O1_H_array[:,i], O2_H_array[:,i], Cs_H_array[:,i], C2_H_array[:,i] = O1s_H, O2s_H, Cs_H, C2_H
#     O1_U_array[:,i], O2_U_array[:,i], Cs_U_array[:,i], C2_U_array[:,i] = O1s_U, O2s_U, Cs_U, C2_U
#     rs_H[i] = r_normed_H
#     rs_U[i] = r_normed_U
    
# op = 'X'

# operators = '_A'+op+'_B'+op

# flag = '4p_and_2p_KI_with_Tinf_state'
# np.savez(flag+f'_time_lim{time_lim}_J{J:.2f}_hx{x:.2f}_hz{z:.2f}_theta_min{min(thetas):.2f}_theta_max{max(thetas):.2f}_theta_len{len(thetas):.2f}_basis_size{N}'+operators+'.npz', O1s_H=O1_H_array, O2s_H=O2_H_array, Cs_H=Cs_H_array, C2_H=C2_H_array, O1s_U=O1_U_array, O2s_U=O2_U_array, Cs_U=Cs_U_array, C2_U=C2_U_array)

#%%
times = np.arange(0,time_lim)

y_O1_U = np.log10(O1s_U) - np.log10(O1s_U)[0] + 1
y_C2_U = np.log10(C2_U)

tmin = 1
tmax = 21

xs = times[tmin:tmax]
y = y_O1_U[tmin:tmax]
yp = y_C2_U[tmin:tmax]

coef = np.polyfit(xs,y,1)
coefp = np.polyfit(xs,yp,1)
poly1d_fn = np.poly1d(coef) #[b,m]
poly1d_fn_p = np.poly1d(coefp) #[b,m]

print(r'$O_1$',poly1d_fn[1])
print(r'$C_2$',poly1d_fn_p[1])

plt.figure(figsize=(16,8))
plt.plot(times,y_O1_U, '*--b', ms=1, lw=1, label='$O_1^U(t)$', alpha=0.8)
plt.plot(xs, poly1d_fn(xs), '-b', lw=2)
plt.plot(times,y_C2_U, '*--r', ms=1, lw=1, label='$C_2^U(t)$', alpha=0.8)
plt.plot(xs, poly1d_fn_p(xs), '-r', lw=2)

plt.text(20, 0, r'$m_1=$'+f'{poly1d_fn[1].real:.2}',
        verticalalignment='bottom', horizontalalignment='right',
        color='blue', fontsize=15)

plt.text(20, -0.25, r'$m_2=$'+f'{poly1d_fn_p[1].real:.2}',
        verticalalignment='bottom', horizontalalignment='right',
        color='red', fontsize=15)

plt.xlabel(r'$\theta$')
plt.ylabel(r'$log_{10}(C(t))$')
# plt.ylim(-4,1)
plt.grid()
plt.legend(loc='best')
opA = 'Z'
opB = 'Z'
operators = '_A'+opA+'_B'+opB
flag = '4p_and_2p_KI_with_Tinf_state'
plt.savefig(flag+f'_time_lim{time_lim}_J{J:.2f}_hx{x:.2f}_hz{z:.2f}_theta_min{min(thetas):.2f}_theta_max{max(thetas):.2f}_theta_len{len(thetas):.2f}_basis_size{N}'+operators+'.png', dpi=80)