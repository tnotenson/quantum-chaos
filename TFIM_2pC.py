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
        op_list = [si for m in range(N)]

        for s_op, s_list in zip(s_ops, s_lists):
            op_list[n] = s_op
            s_list.append(tensor(op_list))
    
    # # define the hamiltonians
    H0 = fH0(N, hx, hz, sx_list, sz_list)
    H1 = fH1(N, J, sz_list)
    
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

def can_bas(N,i):
    e = np.zeros(N)
    e[i] = 1.0
    return e

@jit(nopython=True, parallel=True, fastmath = True)
def Evolucion_numpy(B_t, U, Udag):
    res = Udag@B_t@U
    return res

@jit(nopython=True, parallel=True, fastmath = True)
def twopC_numpy_Tinf(A, B_t):
    C = np.trace(B_t@A)
    return C

@jit(nopython=True, parallel=True, fastmath = True)
def A_average(A, dims):
    res = np.trace(A)**2/dims
    return res

@jit(nopython=True, parallel=True, fastmath = True)
def C_t_commutator_numpy_Tinf(A, B_t):
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

def histo_level_spacing(ener):
    # ener = np.sort(ener)
    spac = np.diff(ener)
    print('espaciado',spac)
    plt.figure(figsize=(16,8))
    plt.hist(spac)#, normed=True)#, bins='auto')
    plt.xlabel('level spacing')
    return 


# @jit(nopython=True)#, parallel=True, fastmath = True)
def diagU_r(U_sub):
    ener = np.linalg.eigvals(U_sub)
    # ener = np.sort(ener)
    fases = [phase(en) for en in ener]
    fases = np.sort(fases)
    # histo_level_spacing(fases)
    r_normed = r_chaometer(fases, plotadjusted=True) 
    return r_normed
#%% separate symmetries (1st option)
# start_time = time()

# P = parity(N)
# ep, epvec = P.eigenstates()

# n_mone, n_one = np.unique(ep, return_counts = True)[1]

# C = np.column_stack([vec.data.toarray() for vec in epvec])
# Cinv = np.linalg.inv(C)

# # H_par = H.transform(Cinv)
# #print(H_par)
# U_par = U.transform(Cinv)
# #print(U_par)
# A_par = A.transform(Cinv)

# # H_sub = H_par.extract_states(np.arange(n_mone:n_one+1))
# U_sub = U_par.extract_states(np.arange(n_mone,n_one+1)).data.toarray()
# A_sub = A_par.extract_states(np.arange(n_mone,n_one+1)).data.toarray()
# #print(H_sub)
# U_sub = np.matrix(U_sub)
# #print(U_sub)
# print(f"\n Separate parity eigenstates --- {time() - start_time} seconds ---" )
#%% separate symmetries (2nd option)
# start_time = time()
# dim = 2**N
# e_basis = [Qobj(can_bas(dim,i)) for i in range(dim)]
# par_basis_ones = np.zeros((dim,dim), dtype=np.complex_)
# for i in range(dim):
#     e_basis[i].dims = [[2 for i in range(N)], [1]]
#     par_basis_ones[i] = (1/2*(e_basis[i] + e_basis[i].permute(np.arange(0,N)[::-1]))).data.toarray()[:,0]
#     norma = np.linalg.norm(par_basis_ones[i])
#     if norma != 0:
#         par_basis_ones[i] = par_basis_ones[i]/norma
#     par = par_basis_ones[i].T@par_basis_ones[i]
# #    print(par)
# par_basis_ones = np.unique(par_basis_ones, axis=1)
# #print(par_basis_ones[:,::-1])
# bas = par_basis_ones
# A_red = bas.conj().T@A.data.toarray()@bas
# U_red = bas.conj().T@U.data.toarray()@bas
# #print(U_red)
# print(f"\n Separate parity eigenstates --- {time() - start_time} seconds ---" )
#%% 
#def U_parity(dim, U, basis):
#    dim_par = basis.shape[1]
##    print(basis.shape)
#    U_sub = np.zeros((dim_par,dim_par), dtype=np.complex_)
#    U = U.data.toarray()
#    for row in range(dim_par):
#        for column in range(dim_par):
##            print(basis[:,row].conj().T.shape)
##            print(basis[:,column].shape)
##            print(U.shape)
#            U_sub[row,column] = basis[:,row].conj().T@U@basis[:,column]
#    return U_sub
#
#U_sub = U_parity(dim, U, par_basis_ones)
#%%
def Evolution2p_H_KI_Tinf(H, time_lim, N, A, B):
    
    start_time = time() 
    
    # define arrays for data storage
    Cs = np.zeros((time_lim))#, dtype=np.complex_)#[]
        
    # define time evolution operator
    U = (-1j*H).expm().data.toarray()
    U = np.matrix(U, dtype=np.complex_)
    Udag = U.H
    # print(U)
    
    # compute OTOC, O1 and O2 for each time
    for i in tqdm(range(time_lim), desc='Evolution loop'):
        
        if i==0:
            B_t = B
        else:
            # Evolution
            B_t = Evolucion_numpy(B_t, U, Udag)
            
        dim = A.shape[0]
        # compute 2-point correlator with qutip
        # C_t = (B_t*A).tr() - A.tr()/dim
        # C_t = C_t/dim
        
        # compute 2-point correlator with qutip
        C_t = twopC_numpy_Tinf(A, B_t)# - A_average(A, dim)
        C_t = C_t/dim/N
        
        print(C_t)

        
    print(f"\nTOTAL --- {time() - start_time} seconds ---" )
    flag = '2p_H_KI_with_Tinf_state'
    return [Cs, flag]

def Evolution2p_U_KI_Tinf(U, time_lim, N, A, B):
    
    start_time = time() 
    
    # define arrays for data storage
    Cs = np.zeros((time_lim), dtype=np.complex_)#[]
    
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
        # compute 2-point correlator with qutip
        # C_t = (B_t*A).tr() - A.tr()/dim
        # C_t = C_t/dim
        
        # compute 2-point correlator with qutip
        C_t = twopC_numpy_Tinf(A, B_t)# - A_average(A, dim)
        C_t = C_t/dim/N
        
        # print(C_t)
        
        # store data
        Cs[i] = np.abs(C_t)

        
    print(f"\nTOTAL --- {time() - start_time} seconds ---" )
    flag = '2p_U_KI_with_Tinf_state'
    return [Cs, flag]

def TFIM_2pC_chaos_parameter(N, B, J, theta, time_lim):
        
    # define parameters of Heisenberg chain with inclined field 
    
    hx = np.sin(theta)*B*np.ones(N)
    hz = np.cos(theta)*B*np.ones(N)
    Jz = J*np.ones(N)
    
    
    start_time = time()
    # let's try it
    
    ######## U evolution ##########
    U = fU(N, Jz, hx, hz)
    
    A = Sj(N, j='x')#/N
    
    # op = 'X'
    
    # operators = '_A'+op+'_B'+op
    
    print(f"\n Create Floquet operator --- {time() - start_time} seconds ---" )
    
    # separate symmetries
    start_time = time()
    
    P = parity(N)
    ep, epvec = P.eigenstates()
    
    n_mone, n_one = np.unique(ep, return_counts = True)[1]
    
    print('tamaños de subespacios de par.', n_mone, n_one)
    C = np.column_stack([vec.data.toarray() for vec in epvec])
    Cinv = np.linalg.inv(C)
        
    ######## U evolution ##########
    U_par = U.transform(Cinv)
    # print(U_par)
    
    A_par = A.transform(Cinv)

    ######## U evolution ##########
    U_sub = U_par.extract_states(np.arange(n_mone, n_mone+n_one)).data.toarray()
    U_sub = np.matrix(U_sub)
    
    A_sub = A_par.extract_states(np.arange(n_mone, n_mone+n_one)).data.toarray()
    print(f"\n Separate parity eigenstates --- {time() - start_time} seconds ---" )
    
    r_normed_U = diagU_r(U_sub)
    
    start_time = time()
    #
    
    Cs, flag = Evolution2p_U_KI_Tinf(U_sub, time_lim, N, A_sub, A_sub)
    # np.savez('2pC_'+flag+f'_time_lim{time_lim}_J{J:.2f}_hx{x:.2f}_hz{z:.2f}_basis_size{N}'+operators+'.npz', Cs=Cs)
    
    print(f"\n Evolution 2pC --- {time() - start_time} seconds ---" )
    
    return [Cs, r_normed_U]

#%% Calcular los correladores para un valor de theta y de B
# define parameters of Heisenberg chain with inclined field 
N = 12
J = 1

time_lim = 51
times = np.arange(0,time_lim)
points = 50

# tomo los valores de Prosen
hx = 1.4
hz = 0

B = np.sqrt(hx**2 + hz**2)
if hz == 0:
    theta = np.pi/2
else:
    theta = np.arctan(hx/hz)
print('B=',B,'\ntheta=',theta)
Cs, r_normed_U = TFIM_2pC_chaos_parameter(N, B, J, theta, time_lim)

#%%

plt.figure(figsize=(16,8))

if hx == hz:
    y_C2_U = np.log10(Cs)

    yfit = np.log(Cs)
    
    tmin = 0
    tmax = 16
    
    xs = times[tmin:tmax]
    yp = yfit[tmin:tmax]
    
    coefp = np.polyfit(xs,yp,1)
    poly1d_fn_p = np.poly1d(coefp) #[b,m]
    
    print(r'$C_2$',poly1d_fn_p[1])
    
    plt.plot(times, np.log10(1/4*np.exp(-times/6)), '-.k', lw=2, label='0.25exp(-t/6)')
    plt.text(20, -1, r'$m_2=$'+f'{poly1d_fn_p[1].real:.2}',
            verticalalignment='bottom', horizontalalignment='right',
            color='red', fontsize=15)
    plt.ylim(-4,0.2)
else:
    y_C2_U = Cs
    plt.plot(times, np.mean(y_C2_U)*np.ones(time_lim), '-.k', lw=2, label=r'$\langle C \rangle = $'+f'{np.mean(y_C2_U).real:.2f}')
    
plt.plot(times,y_C2_U, 'o:k', ms=1, lw=2, label='$C_2(t)$', alpha=0.8)
# plt.plot(xs, poly1d_fn_p(xs), '--r', lw=1)




plt.xlabel(r'$\theta$')
plt.ylabel(r'$log_{10}(C(t))$')
plt.xlim(-0.2,50.2)
plt.grid()
plt.legend(loc='best')
opA = 'X'
opB = 'X'
paridad = 'par'
operators = '_A'+opA+'_B'+opB
BC = 'PBC'
flag = '2p_KI_with_Tinf_state_'
plt.savefig(flag+BC+f'_time_lim{time_lim}_J{J:.2f}_hx{hx:.2f}_hz{hz:.2f}_theta{theta:.2f}_basis_size{N}'+paridad+operators+'.png', dpi=80)
    #%%
# Cs_U_array = np.zeros((time_lim, points))

# rs_U = np.zeros((points))

# for i,z in enumerate(zs):
#     Cs, r_normed_U = TFIM_2pC_chaos_parameter(N, B, J, x, z, time_lim)
#     Cs_U_array[:,i] = Cs
#     rs_U[i] = r_normed_U
#     print('cycle',i)
    
# op = 'X'

# operators = '_A'+op+'_B'+op

# Cs, flag = Evolution2p_U_KI_Tinf(U_sub, time_lim, N, A_sub, A_sub)
# Cs, flag = Evolution2p_H_KI_Tinf(H_sub, time_lim, N, A_sub, A_sub)

#%% plot 2-point correlator results
# pendientes_U = np.zeros(points-1)
# t_max = time_lim
# xs = times[:t_max]

# # define colormap
# colors = plt.cm.jet(np.linspace(0,1,points))

# # fig, ax = plt.subplots(int(points/5),5, figsize=(32,8))
# fig = plt.figure(figsize=(16,8))

# for i,z in enumerate(zs[:-1]):
#     Cs_U = Cs_U_array[:,i]
    
#     yaxis_U = np.log10(Cs_U)# - np.log10(Cs[0])

#     yp = yaxis_U[:t_max]
    
#     coefp = np.polyfit(xs,yp,1)
#     poly1d_fn_p = np.poly1d(coefp) 
    
#     pendientes_U[i] = poly1d_fn_p[1]
#     print('U',poly1d_fn_p[1])
    
#     # resto = i%5
    
#     # if i == 0:
#     #     row = 0
    
#     # elif resto==0:
#     #     row+=1
        
#     # column = resto
    
#     # print(row, column)
    
#     # ax[row,column].title.set_text(f'hz = {z:.1f}')
#     # ax[row,column].plot(xs,yp, 'g-o', lw=0.5)
#     # ax[row,column].plot(xs, poly1d_fn_p(xs), '--k', label=f'{poly1d_fn_p[1]:.2f}')
#     ################### one plot ###########################################
#     plt.plot(xs,yp, '-o', ms=0.5, lw=0.5, color=colors[i], label=f'hz = {z:.1f}')
#     # plt.plot(xs, poly1d_fn_p(xs), '--', color=colors[i]) #, label=f'{poly1d_fn_p[1]:.2f}'
#     ########################################################################
#     # if row!=4:
#     #     ax[row,column].xaxis.set_visible(False)
#     # if row==4:
#     #     ax[row,column].set_xlabel('Time');
#     # if column==0:
#     #     ax[row,column].set_ylabel(r'$log \left(O_1(t) \right)$');
#     # plt.ylim(-3,0.1)
#     ################### one plot ###########################################
#     plt.xlabel('Time');
#     plt.ylabel('Cs(t)');
#     plt.legend(loc='best');
#     plt.grid(True)
#     ########################################################################
#     # ax[row,column].legend(loc='best');
#     # ax[row,column].set_xlim(0,20);
#     # ax[row,column].set_ylim(2,4);
#     # ax[row,column].grid();

#     # plt.show()
# plt.savefig('FIT_2pC_comp_KI_with_Tinf_state'+f'_time_lim{time_lim}_J{J:.2f}_hx{x:.2f}_hz_basis_size{N}'+operators+'.png', dpi=80)#_AX_BX

#%%
# # create the figure
# plt.figure(figsize=(12,8), dpi=100)

# # # all plots in the same figure
# plt.title(f'2-point correlator N = {N}')
    
# # # compute the Ehrenfest's time
# # # tE = np.log(N)/ly1[i]
    
# # plot log10(C(t))

# i = 14


# yaxis = np.log10(Cs_U_array[:,i]/N)# - np.log10(Cs[0])
# plt.plot(times, yaxis,':^r' , label=r'$T = \infty$');

# # # plot vertical lines in the corresponding Ehrenfest's time for each K
# # # plt.vlines(tE, np.min(np.log10(OTOC_Ks[0]*factor)+b), np.max(np.log10(OTOC_Ks[0]*factor)+b), lw=0.5, ls='dashed', alpha=0.8, color=colors[i])
    
# plt.xlabel('Time');
# plt.ylabel(r'$log \left(C(t) \right)$');
# # plt.xlim((0,10))
# # plt.ylim((-4,0.1))
# plt.grid();
# plt.legend(loc='best');
# plt.show()
# # plt.savefig('2pC_'+flag+f'_time_lim{time_lim}_J{j:.2f}_hz{x:.2f}_hz{z:.2f}_basis_size{N}_AZ_BZ.png', dpi=300)#_Gauss_sigma{sigma}
#%%
# normed_slope_H = np.abs(pendientes_H)/max(np.abs(pendientes_U))
# abs_pen = np.abs(pendientes_U)
# normed_slope_U = (abs_pen - min(abs_pen))/(max(abs_pen) - min(abs_pen))

# r_U_normed = (rs_U-min(rs_U))/(max(rs_U) - min(rs_U))

# plt.figure(figsize=(16,8))
# # plt.plot(zs[1:],normed_slope_H, '*--r', ms=1, lw=2, label='slope H')
# plt.plot(zs[1:],normed_slope_U, '*--b', ms=1, lw=2, label='slope U')
# plt.xlabel(r'$h_z$')
# # plt.ylabel(r'$\alpha$ (slope)')
# plt.grid()
# #plt.savefig(f'N{N}_slope_vs_hz.png', dpi=80)
# #
# #plt.figure()
# plt.plot(zs,r_U_normed, '^-b', ms=0.8, lw=0.8, label='U')
# #plt.xlabel(r'$h_z$')
# plt.ylabel(r'$r$ (chaometer) ')
# # plt.ylim(-0.1,1.1)
# #plt.grid()
# plt.legend(loc='best')
# #plt.savefig(f'N{N}_r_vs_hz.png', dpi=80)
# plt.savefig(f'HorU_2pC_N{N}_r_vs_hz.png', dpi=80)

