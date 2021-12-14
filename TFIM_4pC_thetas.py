#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:40:13 2021

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
# plt.rcParams.update({
# "text.usetex": True,
# "font.family": "sans-serif",
# "font.sans-serif": ["Helvetica"], "font.size": 12})

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

def can_bas(N,i):
    e = np.zeros(N)
    e[i] = 1.0
    return e

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

def Evolution4p_H_KI_Tinf(H, time_lim, N, A, B):
    
    start_time = time() 
    
    # define arrays for data storage
    O1s, O2s, Cs = np.zeros((time_lim), dtype=np.complex_),np.zeros((time_lim), dtype=np.complex_),np.zeros((time_lim), dtype=np.complex_)
        
    # define time evolution operator
    U = (-1j*H).expm().data.toarray()
    U = np.matrix(U, dtype=np.complex_)
    Udag = U.H
    print(U.shape)
    print(Udag.shape)
    print(B.shape)
    
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

        print(C_t)
        # store data
        O1s[i] = np.abs(O1); O2s[i] = np.abs(O2); Cs[i] = np.abs(C_t)

        
    print(f"\nTOTAL --- {time() - start_time} seconds ---" )
    flag = '4p_H_KI_with_Tinf_state'
    return [O1s, O2s, Cs, flag]

def Evolution4p_U_KI_Tinf(U, time_lim, N, A, B):
    
    start_time = time() 
    
    # define arrays for data storage
    O1s, O2s, Cs = np.zeros((time_lim), dtype=np.complex_),np.zeros((time_lim), dtype=np.complex_),np.zeros((time_lim), dtype=np.complex_)
    
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

        print(C_t)
        # store data
        O1s[i] = np.abs(O1); O2s[i] = np.abs(O2); Cs[i] = np.abs(C_t)

        
    print(f"\nTOTAL --- {time() - start_time} seconds ---" )
    flag = '4p_U_KI_with_Tinf_state'
    return [O1s, O2s, Cs, flag]

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
    print('espaciado',spac)
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

#%% define operators

N = 11
B = 1
J = 1

x = 1
z = 1

time_lim = 10

def TFIM_O1_chaos_parameter(N, B, J, theta, x, z, time_lim):
        
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
    
    A = Sj(N, j='x')#/N
    
    op = 'X'
    
    operators = '_A'+op+'_B'+op
    
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
    
    ######## H evolution ##########
    H_sub = H_par.extract_states(np.arange(n_mone,n_one+n_mone))
    # print(H_sub)
    
    ######## U evolution ##########
    U_sub = U_par.extract_states(np.arange(n_mone,n_one+n_mone)).data.toarray()
    U_sub = np.matrix(U_sub)
    
    A_sub = A_par.extract_states(np.arange(n_mone,n_one+n_mone)).data.toarray()
    print(f"\n Separate parity eigenstates --- {time() - start_time} seconds ---" )
    
    r_normed_H = diagH_r(H_sub)
    r_normed_U = diagU_r(U_sub)
    
    start_time = time()
    #
    
    O1s_U, O2s_U, Cs_U, flag = Evolution4p_U_KI_Tinf(U_sub, time_lim, N, A_sub, A_sub)
    np.savez('4pC_'+flag+f'_time_lim{time_lim}_J{J:.2f}_hx{x:.2f}_hz{z:.2f}_basis_size{N}'+operators+'.npz', O1s=O1s_U, O2s=O2s_U, Cs=Cs_U)
    
    O1s_H, O2s_H, Cs_H, flag = Evolution4p_H_KI_Tinf(H_sub, time_lim, N, A_sub, A_sub)
    np.savez('4pC_'+flag+f'_time_lim{time_lim}_J{J:.2f}_hx{x:.2f}_hz{z:.2f}_basis_size{N}'+operators+'.npz', O1s=O1s_H, O2s=O2s_H, Cs=Cs_H)
    
    
    print(f"\n Evolution 4pC --- {time() - start_time} seconds ---" )
    
    return [[O1s_H, O2s_H, Cs_H], [O1s_U, O2s_U, Cs_U], r_normed_H, r_normed_U]

points = 100

# zs = np.linspace(0.01,2.6,points)
thetas = np.linspace(0.01, np.pi/2+0.01,points)
z = 1

O1_H_array, O2_H_array, Cs_H_array = np.zeros((time_lim, points)), np.zeros((time_lim, points)), np.zeros((time_lim, points))
O1_U_array, O2_U_array, Cs_U_array = np.zeros((time_lim, points)), np.zeros((time_lim, points)), np.zeros((time_lim, points))

rs_H = np.zeros((points))
rs_U = np.zeros((points))

for i,theta in enumerate(thetas):
    [O1s_H, O2s_H, Cs_H], [O1s_U, O2s_U, Cs_U], r_normed_H, r_normed_U = TFIM_O1_chaos_parameter(N, B, J, theta, x, z, time_lim)
    O1_H_array[:,i], O2_H_array[:,i], Cs_H_array[:,i] = O1s_H, O2s_H, Cs_H
    O1_U_array[:,i], O2_U_array[:,i], Cs_U_array[:,i] = O1s_U, O2s_U, Cs_U
    rs_H[i] = r_normed_H
    rs_U[i] = r_normed_U
    
op = 'X'

operators = '_A'+op+'_B'+op
    
#%% plot 4pC
# subfigures with H and U
times = np.arange(0,time_lim)
    
# fig, ax = plt.subplots(5,5, figsize=(32,8))
# # plt.title(r'$O_1 = \langle X(t)X\,X(t)X \rangle $ TFIM OBC N = %i hx = %.1f hz = %.1f J = %.1f '%(N,x,z,J))
# for i,z in enumerate(zs):
#     O1_H = O1_H_array[:,i]
#     O1_U = O1_U_array[:,i]
    
#     yaxis_H = np.log10(O1_H)# - np.log10(Cs[0])
#     yaxis_U = np.log10(O1_U)# - np.log10(Cs[0])
#     if i < 5:
#         row = 0
#         column = i
#     elif 5 <= i < 10: 
#         row = 1
#         column = i-5
#     elif 10 <= i < 15: 
#         row = 2
#         column = i-10
#     elif 15 <= i < 20: 
#         row = 3
#         column = i-15
#     elif 20 <= i < 25: 
#         row = 4
#         column = i-20
#     ax[row,column].title.set_text(f'hz = {z:.1f}')
#     ax[row,column].plot(times, yaxis_H,':^r' , label=r'$H$');
#     ax[row,column].plot(times, yaxis_U,':^b' , label=r'$U$');
#     if row!=4:
#         ax[row,column].xaxis.set_visible(False)
#     if row==4:
#         ax[row,column].set_xlabel('Time');
#     if column==0:
#         ax[row,column].set_ylabel(r'$log \left(O_1(t) \right)$');
#     # plt.ylim(-3,0.1)
#     # plt.legend(loc='best');
#     # ax[row,column].set_xlim(0,20);
#     ax[row,column].set_ylim(0,4);
#     ax[row,column].grid();
# # plt.show()
# plt.savefig('subfigure_O1s_4pC_4p_comp_KI_with_Tinf_state'+f'_time_lim{time_lim}_J{J:.2f}_hx{x:.2f}_hz{z:.2f}_basis_size{N}'+operators+'.pdf', dpi=100)#_AX_BX

#%% plot 4pC
#subfigures with H or with U
# define colormap
colors = plt.cm.jet(np.linspace(0,1,points))
fig, ax = plt.subplots(2, figsize=(16,8))
# plt.title(r'$O_1 = \langle X(t)X\,X(t)X \rangle $ TFIM OBC N = %i hx = %.1f hz = %.1f J = %.1f '%(N,x,z,J))
for i,theta in enumerate(thetas):
    O1_H = O1_H_array[:,i]
    O1_U = O1_U_array[:,i]
    
    yaxis_H = np.log10(O1_H)# - np.log10(Cs[0])
    yaxis_U = np.log10(O1_U)# - np.log10(Cs[0])

    
    ax[0].title.set_text('TFIM no pateado')
    ax[0].plot(times, yaxis_H,':^r' , label=f'ang={theta:.1f}', color=colors[i], ms=0.5);
    # ax[0].set_xlabel('Time');
    ax[0].set_ylabel(r'$log \left(O_1(t) \right)$');
    # ax[0].set_xlim(0,20)
    # ax[0].legend(loc='best')
    ax[0].grid()
    
    ax[1].title.set_text('TFIM pateado')
    ax[1].plot(times, yaxis_U,':^b' , label=f'ang={theta:.1f}', color=colors[i], ms=0.5);
    ax[1].set_xlabel('Time');
    ax[1].set_ylabel(r'$log \left(O_1(t) \right)$');
    # ax[1].set_xlim(0,20)
    # ax[1].legend(loc='best')
    ax[1].grid()
# plt.show()
plt.savefig('short_time_HorU_O1s_4pC_4p_comp_KI_with_Tinf_state'+f'_time_lim{time_lim}_J{J:.2f}_hx{x:.2f}_hz{z:.2f}_basis_size{N}'+operators+'.png', dpi=80)#_AX_BX

#%% fit exponential decay
pendientes_H = np.zeros(points-1)
pendientes_U = np.zeros(points-1)
# fig, ax = plt.subplots(5,5, figsize=(32,8))

t_max = 10

for i,theta in enumerate(thetas[1:]):
    O1_H = O1_H_array[:,i]
    O1_U = O1_U_array[:,i]
    
    yaxis_H = np.log10(O1_H)# - np.log10(Cs[0])
    yaxis_U = np.log10(O1_U)# - np.log10(Cs[0])

    xs = times[:t_max]
    y = yaxis_H[:t_max]
    yp = yaxis_U[:t_max]
    
    coef = np.polyfit(xs,y,1)
    coefp = np.polyfit(xs,yp,1)
    poly1d_fn = np.poly1d(coef) 
    poly1d_fn_p = np.poly1d(coefp) 
    
    pendientes_H[i] = poly1d_fn[1]
    pendientes_U[i] = poly1d_fn_p[1]
    print('H',poly1d_fn[1])
    print('U',poly1d_fn_p[1])
    
    # if i < 5:
    #     row = 0
    #     column = i
    # elif 5 <= i < 10: 
    #     row = 1
    #     column = i-5
    # elif 10 <= i < 15: 
    #     row = 2
    #     column = i-10
    # elif 15 <= i < 20: 
    #     row = 3
    #     column = i-15
    # elif 20 <= i < 25: 
    #     row = 4
    #     column = i-20
    # ax[row,column].title.set_text(f'hz = {z:.1f}')
    # ax[row,column].plot(xs,y, 'yo')
    # ax[row,column].plot(xs, poly1d_fn(xs), '--k', label=f'{poly1d_fn[1]:.2f}')
    # if row!=4:
    #     ax[row,column].xaxis.set_visible(False)
    # if row==4:
    #     ax[row,column].set_xlabel('Time');
    # if column==0:
    #     ax[row,column].set_ylabel(r'$log \left(O_1(t) \right)$');
    # # plt.ylim(-3,0.1)
    # ax[row,column].legend(loc='best');
    # # ax[row,column].set_xlim(0,20);
    # ax[row,column].set_ylim(2,4);
    # ax[row,column].grid();

    # plt.show()
# plt.savefig('FIT_O1s_4pC_4p_comp_KI_with_Tinf_state'+f'_time_lim{time_lim}_J{J:.2f}_hx{x:.2f}_hz{z:.2f}_basis_size{N}'+operators+'.png', dpi=80)#_AX_BX
#%%
# normed_slope_H = np.abs(pendientes_H)/max(np.abs(pendientes_U))
abs_pen = np.abs(pendientes_U)
normed_slope_U = (abs_pen - min(abs_pen))/(max(abs_pen) - min(abs_pen))

r_H_normed = (rs_H-min(rs_H))/(max(rs_H) - min(rs_H))
r_U_normed = (rs_U-min(rs_U))/(max(rs_U) - min(rs_U))

plt.figure(figsize=(16,8))
# plt.plot(thetas[1:],normed_slope_H, '*--r', ms=1, lw=2, label='slope H')
plt.plot(thetas[1:],normed_slope_U, '*--b', ms=1, lw=2, label='slope U')
plt.xlabel(r'$\theta$')
# plt.ylabel(r'$\alpha$ (slope)')
plt.grid()
#plt.savefig(f'N{N}_slope_vs_hz.png', dpi=80)
#
#plt.figure()
plt.plot(thetas,r_H_normed, '^-r', ms=0.8, lw=0.8, label='H')
plt.plot(thetas,r_U_normed, '^-b', ms=0.8, lw=0.8, label='U')
#plt.xlabel(r'$\theta$')
plt.ylabel(r'$r$ (chaometer) ')
# plt.ylim(-0.1,1.1)
#plt.grid()
plt.legend(loc='best')
#plt.savefig(f'N{N}_r_vs_hz.png', dpi=80)
plt.savefig(f'HorU_N{N}_r_vs_hz.png', dpi=80)

