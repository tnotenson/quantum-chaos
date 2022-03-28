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
"font.sans-serif": ["Helvetica"], "font.size": 24})

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
        
    for n in range(N):
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

@jit(nopython=True, parallel=True, fastmath = True)
def Evolucion_numpy(B_t, U, Udag):
    res = Udag@B_t@U
    return res

@jit(nopython=True, parallel=True, fastmath = True)
def O1_numpy_Tinf(A, B_t):
    O1 = np.trace(B_t@A@B_t@A)
    return O1

@jit(nopython=True, parallel=True, fastmath = True)
def twopC_numpy_Tinf(A, B_t):
    C = np.trace(B_t@A)
    return C

def Evolution4p_and_2p_H_KI_Tinf(H, time_lim, A, B):
    
    start_time = time() 
    
    # define arrays for data storage
    O1s, Cs = np.zeros((time_lim), dtype=np.complex_),np.zeros((time_lim), dtype=np.complex_)
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
            
        # compute 4-point correlator with qutip
        # O1 = (B_t*A*B_t*A).tr()
        # O2 = (B_t**2*A**2).tr()
                    
        # C_t = -2*( O1 - O2 )/dim
        
        # compute 4-point correlator with numpy
        O1 = O1_numpy_Tinf(A, B_t)
        
        C_t = twopC_numpy_Tinf(A, B_t)

        # print(C_t)
        # store data
        O1s[i] = np.abs(O1); Cs[i] = np.abs(C_t)

        
    print(f"\nTOTAL --- {time() - start_time} seconds ---" )
    flag = '4p_and_2p_H_KI_with_Tinf_state'
    return [O1s, Cs, flag]

def Evolution4p_and_2p_U_KI_Tinf(U, time_lim, A, B):
    
    start_time = time() 
    
    # define arrays for data storage
    O1s, Cs = np.zeros((time_lim), dtype=np.complex_),np.zeros((time_lim), dtype=np.complex_)
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
              
        # compute 4-point correlator with qutip
        # O1 = (B_t*A*B_t*A).tr()
        # O2 = (B_t**2*A**2).tr()
                    
        # C_t = -2*( O1 - O2 )/dim
        
        # compute 4-point correlator with numpy
        O1 = O1_numpy_Tinf(A, B_t)
        
        C_t = twopC_numpy_Tinf(A, B_t)

        # print(C_t)
        # store data
        O1s[i] = np.abs(O1); Cs[i] = np.abs(C_t)

        
    print(f"\nTOTAL --- {time() - start_time} seconds ---" )
    flag = '4p_and_2p_H_KI_with_Tinf_state'
    return [O1s, Cs, flag]

def TFIM4pC_and_2pC_chaos_parameter(N, J, x, z, time_lim, evolution='U'):
        
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
        
    A = Sj(N, j='x')#/N
    B = Sj(N, j='x')
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
    
    P = parity(N)
    ep, epvec = P.eigenstates()
    
    print(ep)
    
    n_mone, n_one = np.unique(ep, return_counts = True)[1]
    
    print('tamaños de subespacios de par.', n_mone,'+',n_one,'=', n_mone+n_one)
    C = np.column_stack([vec.data.toarray() for vec in epvec])
    Cinv = np.linalg.inv(C)
    
    if evolution=='H':
        ######## H evolution ##########
        
        H_par = H.transform(Cinv)
        # print(H_par)
        
    elif evolution=='U':
        # ######## U evolution ##########
        U_par = U.transform(Cinv)
        # # hinton(U_par)
  
    
    A_par = A.transform(Cinv)
    B_par = B.transform(Cinv)
    # hinton(A_par)
    
    
        
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
    
    # r_normed_H = diagH_r(H_sub)
    # r_normed_U = diagU_r(U_sub)
    
    evol_time = time()
    #
    
    if evolution=='H':
                
        O1_mone, Cs_mone, _ = Evolution4p_and_2p_H_KI_Tinf(H_mone, time_lim, A_mone, B_mone)
        O1_one, Cs_one, _ = Evolution4p_and_2p_H_KI_Tinf(H_one, time_lim, A_one, B_one)
        
    elif evolution=='U':
            
        O1_mone, Cs_mone, _ = Evolution4p_and_2p_U_KI_Tinf(U_mone, time_lim, A_mone, B_mone)
        O1_one, Cs_one, _ = Evolution4p_and_2p_U_KI_Tinf(U_one, time_lim, A_one, B_one)
        # np.savez('2pC_'+flag+f'_time_lim{time_lim}_J{J:.2f}_hx{x:.2f}_hz{z:.2f}_basis_size{N}'+operators+'.npz', Cs=Cs)
    
    print(f"\n Evolution 2pC --- {time() - evol_time} seconds ---" )
    print(f"\n TOTAL --- {time() - start_time} seconds ---" )
    return [[O1_mone, Cs_mone], [O1_one, Cs_one]]#, r_normed_U# [Cs] #

#%% define operators
N = 13
J = 1

time_lim = 31
#%% Calcular los correladores para un valor de theta y de B
# tomo los valores de Prosen
hx = 1.4
hz = 1.4
B = np.sqrt(hx**2 + hz**2)
if hz == 0:
    theta = np.pi/2
else:
    theta = np.arctan(hx/hz)
print('B=',B,'\ntheta=',theta)

[[O1_mone, Cs_mone], [O1_one, Cs_one]] = TFIM4pC_and_2pC_chaos_parameter(N, J, hx, hz, time_lim, evolution='U')
dimension = 2**N
O1s = (O1_mone+O1_one)/dimension/N
Cs = (Cs_mone+Cs_one)/dimension/N
#%% Calcular los correladores para varios valores de theta y uno de B
# points = 100
# thetas = np.linspace(0.01, np.pi/2+0.01,points)

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

y_O1 = np.log10(O1s) - np.log10(O1s)[0]
y_Cs = np.log10(Cs)

y1fit = np.log(O1s)
y2fit = np.log(Cs)

tmin = 0
tmax = 16

xs = times[tmin:tmax]
y = y1fit[tmin:tmax]
yp = y2fit[tmin:tmax]

coef = np.polyfit(xs,y,1)
coefp = np.polyfit(xs,yp,1)
poly1d_fn = np.poly1d(coef) #[b,m]
poly1d_fn_p = np.poly1d(coefp) #[b,m]

print(r'$O_1$',poly1d_fn[1])
print(r'$C_2$',poly1d_fn_p[1])

plt.figure(figsize=(16,8))
plt.plot(times,y_O1, '*-b', ms=1, lw=1, label='$O_1^U(t)$', alpha=0.8)
# plt.plot(xs, poly1d_fn(xs), '-b', lw=2)
plt.plot(times,y_Cs, '*-r', ms=1, lw=1, label='$C_2^U(t)$', alpha=0.8)
# plt.plot(xs, poly1d_fn_p(xs), '-r', lw=2)
plt.plot(times, np.log10(1/4*np.exp(-times/6)), '--k', lw=1, label='0.25exp(-t/6)')

plt.text(20, -1, r'$m_1=$'+f'{poly1d_fn[1].real:.2}',
        verticalalignment='bottom', horizontalalignment='right',
        color='blue', fontsize=15)

plt.text(20, -1.5, r'$m_2=$'+f'{poly1d_fn_p[1].real:.2}',
        verticalalignment='bottom', horizontalalignment='right',
        color='red', fontsize=15)

plt.xlabel(r'$t$')
plt.ylabel(r'$log_{10}(C(t))$')
# plt.ylim(-4,1.1)
plt.xlim(-0.2,50.2)
plt.grid()
plt.legend(loc='best')
opA = 'X'
opB = 'X'
operators = '_A'+opA+'_B'+opB
flag = '4p_and_2p_KI_with_Tinf_state'
plt.savefig(flag+f'_time_lim{time_lim}_J{J:.2f}_hx{hx:.2f}_hz{hz:.2f}_theta{theta:.2f}_basis_size{N}'+operators+'suma_subparidad.png', dpi=80)