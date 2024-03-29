#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:16:06 2022

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
# s0=(si-sz)/2

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

# 4 operator correlator dim * <B(t) * A * B(t) * A>
@jit(nopython=True)#, parallel=True, fastmath = True)
def O1_numpy_Tinf(A, B_t):
    O1 = np.trace(B_t@A@B_t@A)
    return O1

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

def Evolution4p_and_2p_H_KI_Tinf(H, time_lim, A, B):
    
    start_time = time() 
    
    # define arrays for data storage
    O1s, Cs = np.zeros((time_lim), dtype=np.complex_),np.zeros((time_lim), dtype=np.complex_)
    # define time evolution operator
    U = propagator(H,1).full()
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
        
        # compute 2-point correlator with numpy
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
        
        # compute 2-point correlator with numpy
        C_t = twopC_numpy_Tinf(A, B_t)

        # print(C_t)
        # store data
        O1s[i] = np.abs(O1); Cs[i] = np.abs(C_t)

        
    print(f"\nTOTAL --- {time() - start_time} seconds ---" )
    flag = '4p_and_2p_H_KI_with_Tinf_state'
    return [O1s, Cs, flag]

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
    # r_normed_U = diagU_r(U.full())
    r_normed_mone = diagU_r(U_mone)
    r_normed_one = diagU_r(U_one)
    
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
    return [[O1_mone, Cs_mone], [O1_one, Cs_one], r_normed_mone, r_normed_one]# [Cs] #


#%% define operators
N = 12
J = 1

time_lim = 31
# Calcular los correladores para un valor de theta y de B
# tomo los valores de Prosen
hx = 1.4
#hz = 0.4
#B = np.sqrt(hx**2 + hz**2)
#if hz == 0:
#    theta = np.pi/2
#else:
#    theta = np.arctan(hx/hz)
#print('B=',B,'\ntheta=',theta)

hzs = np.arange(0.3,0.4,0.2)

for hz in hzs:
        
    [[O1_mone, Cs_mone], [O1_one, Cs_one], r_normed_mone, r_normed_one] = TFIM4pC_and_2pC_chaos_parameter(N, J, hx, hz, time_lim, evolution='U')
    
    # dimension = 2**N
    
    O1s = np.abs(O1_mone+O1_one)#/dimension#/N
    Cs = np.abs(Cs_mone+Cs_one)#/dimension#/N
    
    Var = (O1s - Cs**2)#/dimension#/N
    
    
    
    opA = 'X'
    opB = 'X'
    operators = '_A'+opA+'_B'+opB
    flag = 'Var_KI_with_Tinf_state'
    np.savez(flag+f'_time_lim{time_lim}_J{J:.2f}_hx{hx:.2f}_hz{hz:.2f}_basis_size{N}'+operators+'suma_subparidad.npz', Cs=Cs, O1s=O1s, r_normed_mone=r_normed_mone, r_normed_one=r_normed_one)#%% Calcular los correladores para varios valores de theta y uno de B
#r_normed_mone=r_normed_mone, r_normed_one=r_normed_one
#%% FALTA EDITAR ESTO
N = 12
J = 1
hx = 1.4

plott = 'O1'#'C2'#'Var'#

time_lim = 31
times = np.arange(0,time_lim)
hzs = np.arange(0,1.5,0.1)

pendientes, r_mone_list, r_one_list = np.zeros((len(hzs))), np.zeros((len(hzs))), np.zeros((len(hzs)))
 

for k, hz in enumerate(hzs):
    name = flag+f'_time_lim{time_lim}_J{J:.2f}_hx{hx:.2f}_hz{hz:.2f}_basis_size{N}'+operators+'suma_subparidad.npz'
    archives = np.load(name)
    
    Cs=archives['Cs']
    O1s=archives['O1s']
    r_normed_mone=archives['r_normed_mone']
    r_normed_one=archives['r_normed_one']
    
    r_mone_list[k] = r_normed_mone
    r_one_list[k] = r_normed_one
    
    tmin = 0
    tmax = 16
    xs = times[tmin:tmax]
    
    if plott == 'Var':
        Var = (O1s - Cs**2)#/dimension#/N
        y_Var = np.log10(Var)
        y = y_Var[tmin:tmax]
    elif plott == 'O1':
        y_O1 = np.log10(O1s)
        y = y_O1[tmin:tmax]
    elif plott == 'C2':
        y_Cs = np.log10(Cs**2)
        y = y_Cs[tmin:tmax]

    coef = np.polyfit(xs,y,1)
    poly1d_fn = np.poly1d(coef) #[b,m]
    m, b = poly1d_fn
    
    pendientes[k] = np.abs(m)

pendientes_normed = (pendientes-min(pendientes))/(max(pendientes)-min(pendientes))

r_mone_list = np.nan_to_num(r_mone_list)
r_one_list = np.nan_to_num(r_one_list)

r_mone = (r_mone_list - min(r_mone_list))/(max(r_mone_list)-min(r_mone_list))
r_one = (r_one_list - min(r_one_list))/(max(r_one_list)-min(r_one_list))

plt.figure(figsize=(16,8))
plt.title(f'A={opA}, B={opB}.')
plt.plot(hzs, pendientes_normed, '-b', lw=1.5, label='pendiente')
plt.plot(hzs, 1-r_mone, '-r', lw=1.5, label='1-r paridad -1')
plt.plot(hzs, 1-r_one, '-g', lw=1.5, label='1-r paridad +1')
plt.ylabel(r'$\alpha$')
plt.xlabel(r'$h_z$')
# plt.ylim(-4,1.1)
# plt.xlim(-0.2,max(times)+0.2)
plt.grid(True)
plt.legend(loc = 'best')
plt.savefig('pendientes_y_r_vs_hz_'+plott+'_'+flag+f'_time_lim{time_lim}_J{J:.2f}_hx{hx:.2f}_hzs_basis_size{N}'+operators+'suma_subparidad.png', dpi=80)
    