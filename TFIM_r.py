#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:49:48 2021

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
        H += -J[n] * sz_list[n] * sz_list[n+1]
    
    return H

def TFIM_Nico(N, hx, hz, Jz):

    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sx
        sx_list.append(tensor(op_list))

        op_list[n] = sy
        sy_list.append(tensor(op_list))

        op_list[n] = sz
        sz_list.append(tensor(op_list))

    # construct the hamiltonian
    H = 0

    # energy splitting terms
    for n in range(N):
        H +=  hz[n] * sz_list[n]
        
    for n in range(N):
        H += hx[n] * sx_list[n]

    # interaction terms
    for n in range(N-1):
        H += - Jz[n] * sz_list[n] * sz_list[n+1]
        
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
        H += -J[n] * sz_list[n] * sz_list[(n+1)]
    
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

# @jit(nopython=True)#, parallel=True, fastmath = True)
def diagH_r(H_sub):
    ener = np.linalg.eigvals(H_sub)
    ener = np.sort(ener)
    # histo_level_spacing(ener)
    r_normed = r_chaometer(ener, plotadjusted=True) 
    return r_normed

# @jit(nopython=True)#, parallel=True, fastmath = True)
def diagU_r(U_sub):
    ener = np.linalg.eigvals(U_sub)
    # ener = np.sort(ener)
    fases = [phase(en) for en in ener]
    fases = np.sort(fases)
    # histo_level_spacing(fases)
    r_normed = r_chaometer(fases, plotadjusted=True) 
    return r_normed

def r_thetas(N, B, J, theta, x, z):
     
    hx = np.sin(theta)*x*B*np.ones(N)
    hz = np.cos(theta)*z*B*np.ones(N)
    Jz = J*np.ones(N)
    
    start_time = time()
    # let's try it
    
    ######## H evolution ##########
    # H = TFIM_Nico(N, hx, hz, Jz)
    H = TFIM(N, hx, hz, Jz)
    ######## U evolution ##########
    U = fU(N, Jz, hx, hz)
    
    # separate symmetries
    start_time = time()
    
    ######################### NICO ##########################################
    
    # ene_H,est_H = H.eigenstates()
    # ene_U,est_U = U.eigenstates()
    
    # ind_impar_H = []
    # ind_par_H = []
    
    # ind_impar_U = []
    # ind_par_U = []
    
    # for x in range(2**N):
    #     # calculo la paridad a mano para cada autoestado
    #     if np.real((est_H[x].permute(np.arange(N-1,-1,-1)).dag()*est_H[x])[0][0][0]) < 0:
    #         ind_impar_H.append(x)
    #     else:
    #         ind_par_H.append(x)
            
    #     if np.real((est_U[x].permute(np.arange(N-1,-1,-1)).dag()*est_U[x])[0][0][0]) < 0:
    #         ind_impar_U.append(x)
    #     else:
    #         ind_par_U.append(x)
    
    # C_H = np.column_stack([vec.data.toarray() for vec in est_H])
    # C_H = np.matrix(C_H)
    # Cinv_H = C_H.H
    
    # C_U = np.column_stack([vec.data.toarray() for vec in est_U])
    # C_U = np.matrix(C_U)
    # Cinv_U = C_U.H
    
    # ######## H evolution ##########
    
    # H_ener = H.transform(Cinv_H)
    # # print(H_ener)
    
    # ######## U evolution ##########
    # U_ener = U.transform(Cinv_U)
    # # print(U_ener)
    
    # ######## H evolution ##########
    # H_sub = H_ener.extract_states(ind_par_H)#.data.toarray()
    # # print(H_sub)
    
    # ######## U evolution ##########
    # U_sub = U_ener.extract_states(ind_par_U).data.toarray()
    # U_sub = np.matrix(U_sub)
    
    #########################################################################
    
    ############################## TOMI #####################################
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
    print(f"\n r-chaometer hz = {z} --- {time() - start_time} seconds ---" )
    return [r_normed_H, r_normed_U]

#%% define parameters

N = 12

J = 1

x = 1
z = 1

points = 25

# thetas = np.linspace(0.01, np.pi/2+0.01,points)
zs = np.linspace(0+0.001,2.5+0.001,points)

theta = np.pi/4+0.0001

B = 1/np.sin(theta)

rs_H = np.zeros((points))
rs_U = np.zeros((points))

# for i,theta in tqdm(enumerate(thetas), desc='thetas loop'):
for i,z in tqdm(enumerate(zs), desc='zs loop'):
    r_normed_H, r_normed_U = r_thetas(N, B, J, theta, x, z)
    rs_H[i] = r_normed_H
    rs_U[i] = r_normed_U
    
#%%
r_H_normed = rs_H#(rs_H-min(rs_H))/(max(rs_H) - min(rs_H))
r_U_normed = rs_U#(rs_U-min(rs_U))/(max(rs_U) - min(rs_U))

plt.figure(figsize=(16,8))
plt.plot(zs,r_H_normed, '^-r', ms=0.8, lw=0.8, label='H')
plt.plot(zs,r_U_normed, '^-b', ms=0.8, lw=0.8, label='U')
# plt.xlabel(r'$\theta$')
plt.xlabel(r'$h_z$')
plt.ylabel(r'$r$ (chaometer) ')
# plt.ylim(-0.1,1.1)
plt.grid(True)
plt.legend(loc='best')

# plt.savefig(f'HorU_N{N}_r_vs_hz_{points}points_Nico.png', dpi=80)#_Nico
plt.savefig(f'HorU_N{N}_r_vs_hz_{points}points_sinnorm.png', dpi=80)#_Nico
# plt.savefig(f'HorU_N{N}_r_vs_hz_pico_en_hz1.2.png', dpi=80)#_Nico