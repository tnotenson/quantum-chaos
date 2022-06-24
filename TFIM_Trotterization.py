#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 14:02:04 2021

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
# s0=(si-sz)/2


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
        H += -J[n] * sz_list[n] * sz_list[n+1]
    
    return H

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

@jit(nopython=True)#, parallel=True, fastmath = True)
def Evolucion_numpy(B_t, U, Udag):
    res = Udag@B_t@U
    return res

@jit(nopython=True)#, parallel=True, fastmath = True)
def O1_numpy_Tinf(A, B_t):
    O1 = np.trace(B_t@A@B_t@A)
    return O1

@jit(nopython=True)#, parallel=True, fastmath = True)
def O2_numpy_Tinf(A, B_t):
    O2 = np.trace(B_t@B_t@A@A)
    return O2

@jit(nopython=True)#, parallel=True, fastmath = True)
def A_average(A, dims):
    res = np.trace(A)**2/dims
    return res

@jit(nopython=True)#, parallel=True, fastmath = True)
def C_t_commutator_numpy_Tinf(A, B_t):
    com = B_t@A - A@B_t
    C_t = np.trace(com.H@com)/N
    return C_t

def Evolution4p_H_KI_Tinf(H, time_lim, dt, N, A, B):
    
    start_time = time() 
    
    time_dim = int(time_lim/dt)
    
    # define arrays for data storage
    O1s, O2s, Cs = np.zeros((time_dim), dtype=np.complex_),np.zeros((time_dim), dtype=np.complex_),np.zeros((time_dim), dtype=np.complex_)
        
    # define time evolution operator
    U = propagator(H,1).full()
    U = np.matrix(U, dtype=np.complex_)
    Udag = U.H
    # print(U.shape)
    # print(Udag.shape)
    # print(B.shape)
    
    # compute OTOC, O1 and O2 for each time
    for i in tqdm(range(time_dim), desc='Evolution loop'):
        
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

        # print(C_t)
        # store data
        O1s[i] = np.abs(O1); O2s[i] = np.abs(O2); Cs[i] = np.abs(C_t)

        
    # print(f"\nTOTAL --- {time() - start_time} seconds ---" )
    flag = '4p_H_KI_with_Tinf_state'
    return [O1s, O2s, Cs, flag]

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



#%% define operators

N = 10

J = 1

x = 1
z = 1

time_lim = 1
dt = 1/time_lim/1000
time_dim = int(time_lim/dt)

theta = np.pi/4# + 0.0001

B = 1/np.sin(theta)

def TFIM_O1_Trotterization_parameter(N, B, J, x, z, time_lim, dt):
        
    # define parameters of Heisenberg chain with inclined field 
    
    hx = np.sin(theta)*x*B*np.ones(N)
    hz = np.cos(theta)*z*B*np.ones(N)
    Jz = J*np.ones(N)
    
    start_time = time()
    # let's try it
    
    ######## H evolution ##########
    H = TFIM(N, hx, hz, Jz)
    
    A = Sj(N, j='x')#/N
    
    # op = 'X'
    
    # operators = '_A'+op+'_B'+op
    
    print(f"\n Create Floquet operator --- {time() - start_time} seconds ---" )
    
    # separate symmetries
    start_time = time()
    
    ene_H,est_H = H.eigenstates()
    
    ind_impar_H = []
    ind_par_H = []
    
    for x in range(2**N):
        # calculo la paridad a mano para cada autoestado
        if np.real((est_H[x].permute(np.arange(N-1,-1,-1)).dag()*est_H[x])[0][0][0]) < 0:
            ind_impar_H.append(x)
        else:
            ind_par_H.append(x)
    
    C_H = np.column_stack([vec.data.toarray() for vec in est_H])
    C_H = np.matrix(C_H)
    Cinv_H = C_H.H
        
    ######## H evolution ##########
    
    H_ener = H.transform(Cinv_H)
    # print(H_ener)

    
    A_ener_H = A.transform(Cinv_H)
    ######## H evolution ##########
    H_sub = H_ener.extract_states(ind_par_H)#.data.toarray()
    # print(H_sub)
   
    A_sub_H = A_ener_H.extract_states(ind_par_H).data.toarray()

    # P = parity(N)
    # ep, epvec = P.eigenstates()
    
    # n_mone, n_one = np.unique(ep, return_counts = True)[1]
    
    # C = np.column_stack([vec.data.toarray() for vec in epvec])
    # C = np.matrix(C)
    # Cinv = C.H
    
    # ######## H evolution ##########
    
    # H_par = H.transform(Cinv)
    # # print(H_par)
    
    # ######## U evolution ##########
    # U_par = U.transform(Cinv)
    # # print(U_par)
    
    # A_par = A.transform(Cinv)
    
    # ######## H evolution ##########
    # H_sub = H_par.extract_states(np.arange(n_mone,n_one+n_mone))
    # # print(H_sub)
    
    # ######## U evolution ##########
    # U_sub = U_par.extract_states(np.arange(n_mone,n_one+n_mone)).data.toarray()
    # # U_sub = np.matrix(U_sub)
    
    # A_sub = A_par.extract_states(np.arange(n_mone,n_one+n_mone)).data.toarray()
    
    # C2 = np.column_stack([vec.data.toarray() for vec in epvec[n_mone:n_one+n_mone]])
    # C2 = np.matrix(C2)
    # C2inv = C2.H
    
    # H_fin = C2@H_sub.data.toarray()@C2inv
    
    # U_fin = C2@U_sub@C2inv
    # U_fin = np.matrix(U_fin)
    
    # A_fin = C2@A_sub@C2inv
    
    # print(f"\n Separate parity eigenstates --- {time() - start_time} seconds ---" )
    
    r_normed_H = diagH_r(H_sub)
    
    start_time = time()
    #

    O1s_H, O2s_H, Cs_H, flag = Evolution4p_H_KI_Tinf(H_sub, time_lim, dt, N, A_sub_H, A_sub_H)
    # O1s_H, O2s_H, Cs_H, flag = Evolution4p_H_KI_Tinf(H_fin, time_lim, N, A_fin, A_fin)
    # np.savez('4pC_'+flag+f'_time_lim{time_lim}_J{J:.2f}_hx{x:.2f}_hz{z:.2f}_basis_size{N}'+operators+'.npz', O1s=O1s_H, O2s=O2s_H, Cs=Cs_H)
    
    
    # print(f"\n Evolution 4pC --- {time() - start_time} seconds ---" )
    
    return [[O1s_H, O2s_H, Cs_H], r_normed_H]

points = 20

zs = np.linspace(0.01,2.5+0.01,points)
# thetas = np.linspace(0.01, np.pi/2+0.01,points)
z = 1

O1_H_array, O2_H_array, Cs_H_array = np.zeros((time_dim, points)), np.zeros((time_dim, points)), np.zeros((time_dim, points))

rs_H = np.zeros((points))

start_time = time()

for i,z in enumerate(zs):
    [O1s_H, O2s_H, Cs_H], r_normed_H = TFIM_O1_Trotterization_parameter(N, B, J, x, z, time_lim, dt)
    O1_H_array[:,i], O2_H_array[:,i], Cs_H_array[:,i] = O1s_H, O2s_H, Cs_H

    rs_H[i] = r_normed_H
    
print(f'TOTAL --- {time() - start_time} seg')
op = 'X'

operators = '_A'+op+'_B'+op

#%% plot 4pC
# figure for H trotterization
# define time array
times = np.arange(0,time_lim,dt)
# define colormap
colors = plt.cm.jet(np.linspace(0,1,points))
# create figure
fig = plt.figure(figsize=(16,8))
# plt.title(r'$O_1 = \langle X(t)X\,X(t)X \rangle $ TFIM OBC N = %i hx = %.1f hz = %.1f J = %.1f '%(N,x,z,J))
for i,z in enumerate(zs):
    O1_H = O1_H_array[:,i]
    
    yaxis_H = np.log10(O1_H)# - np.log10(Cs[0])

    
    plt.title('TFIM no pateado')
    plt.plot(times, yaxis_H,':^r' , label=f'ang={z:.1f}', color=colors[i], ms=0.5);
    # plt.set_xlabel('Time');
    plt.ylabel(r'$log \left(O_1(t) \right)$');
    # plt.set_xlim(0,20)
    # plt.legend(loc='best')
    plt.grid()

# plt.show()
plt.savefig('short_time_H_Trotterization_O1s_4pC_comp_KI_with_Tinf_state'+f'_dt{dt}_time_lim{time_lim}_J{J:.2f}_hx{x:.2f}_hz{z:.2f}_basis_size{N}'+operators+'.png', dpi=80)#_AX_BX
np.savez('short_time_H_Trotterization_O1s_4pC_comp_KI_with_Tinf_state'+f'_dt{dt}_time_lim{time_lim}_J{J:.2f}_hx{x:.2f}_hz{z:.2f}_basis_size{N}'+operators+'.npz', O1s=O1_H_array, O2s=O2_H_array, Cs=Cs_H_array)
