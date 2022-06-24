#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 21:24:29 2021

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

#%% define parameters

# N = 10

# J = 1

# x = 1
# z = 1
# dt = 1e-1

# pot = 1

# points = 100

# thetas = np.linspace(0.001, np.pi/2+0.001,points)
# # thetas = np.linspace(1.5+0.001, np.pi/2+0.001,points)
# # zs = np.linspace(1.21+0.001,1.215+0.001,points)#np.linspace(1+0.001,1.4+0.001,points)

# theta = np.pi/4+0.0001

# B = np.pi/4+0.001

# rs_H = np.zeros((points))
# rs_U = np.zeros((points))

# for i,theta in tqdm(enumerate(thetas), desc='thetas loop'):
# # for i,z in tqdm(enumerate(zs), desc='zs loop'):
#     r_normed_H, r_normed_U = r_thetas(N, B, J, theta, x, z, dt, pot)
#     rs_H[i] = r_normed_H
#     rs_U[i] = r_normed_U
    
# np.savez(f'Un{pot}_N{N}_theta{theta}_B{B}_points{points}_dt{dt}.npz', rs_U=rs_U)
# np.savez(f'H_N{N}_theta{theta}_B{B}_points{points}_dt{dt}.npz', rs_H=rs_H)
#%%
# r_H_normed = rs_H#(rs_H-min(rs_H))/(max(rs_H) - min(rs_H))
# r_U_normed = rs_U#(rs_U-min(rs_U))/(max(rs_U) - min(rs_U))

# plt.figure(figsize=(16,8))
# plt.title(r'r-chaometer para $U^n$ con '+f'n = {pot}')
# # plt.plot(zs,r_H_normed, '^-r', ms=0.8, lw=0.8, label='H')
# # plt.plot(zs,r_U_normed, '^-b', ms=0.8, lw=0.8, label='U')
# plt.plot(thetas,r_H_normed, '^-r', ms=0.8, lw=0.8, label='$H$')
# plt.plot(thetas,r_U_normed, '^-b', ms=0.8, lw=0.8, label='$U^n$')
# plt.xlabel(r'$\theta$')
# # plt.xlabel(r'$h_z$')
# plt.ylabel(r'$r$ (chaometer) ')
# # plt.ylim(-0.1,1.1)
# plt.grid(True)
# plt.legend(loc='best')

# # plt.savefig(f'HorU_N{N}_r_vs_hz_{points}points_sinnorm_lims_{min(zs)}_{max(zs)}_dt{dt}.png', dpi=80)#_Nico
# plt.savefig(f'HorUn_N{N}_B{B}_r_vs_hz_{points}points_sinnorm_lims_{min(thetas)}_{max(thetas)}_dt{dt}_pot{pot}.png', dpi=80)#_Nico
#%%
# define colormap
# colors = plt.cm.jet(np.linspace(0,1,8))
# rs_H = np.load(f'H_N{N}_theta{theta}_B{B}_points{points}_dt{dt}.npz')['rs_H']
# plt.figure(figsize=(16,8))
# ax = plt.gca()
# plt.title(r'r-chaometer para $U^n$')
# plt.plot(thetas,rs_H, '^--k', ms=1, lw=2.5, label='$H$')
# for pot in range(2,10):
#     rs_U = np.load(f'Un{pot}_N{N}_theta{theta}_B{B}_points{points}_dt{dt}.npz')['rs_U']
#     plt.plot(thetas,rs_U, '^-', color=colors[pot-2], ms=1, lw=1.5, label=f'n={pot}')
# plt.xlabel(r'$\theta$')
# ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 8))
# ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 8))
# # plt.xlabel(['0',r'$\pi/8$',r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
# # plt.xlabel(r'$h_z$')
# plt.ylabel(r'$r$ (chaometer) ')
# # plt.ylim(-0.1,1.1)
# plt.grid(True)
# plt.legend(loc='best')

# # plt.savefig(f'HorU_N{N}_r_vs_hz_{points}points_sinnorm_lims_{min(zs)}_{max(zs)}_dt{dt}.png', dpi=80)#_Nico
# plt.savefig(f'rchaometer_theta_H_Un{pot}_N{N}_B{B}_r_vs_hz_{points}points_sinnorm_lims_{min(thetas)}_{max(thetas)}_dt{dt}.png', dpi=80)#_Nico
#%% dt vs N

Ns = np.arange(6,11)

J = 1

x = 1
z = 1
dt = .1

pot = 1

points = 100

thetas = np.linspace(0.001, np.pi/2+0.001,points)
# thetas = np.linspace(1.5+0.001, np.pi/2+0.001,points)
# zs = np.linspace(1.21+0.001,1.215+0.001,points)#np.linspace(1+0.001,1.4+0.001,points)

e = 0.0001

theta = np.pi/4+e

B = np.pi/4+e

rs_H = np.zeros((points))
rs_U = np.zeros((points))

for j,N in tqdm(enumerate(Ns), desc='Ns loop'):
    for i,theta in tqdm(enumerate(thetas), desc='thetas loop'):
    # for i,z in tqdm(enumerate(zs), desc='zs loop'):
        r_normed_H, r_normed_U = r_thetas(N, B, J, theta, x, z, dt, pot)
        rs_H[i] = r_normed_H
        rs_U[i] = r_normed_U
        
    np.savez(f'Un{pot}_N{N}_theta{theta}_B{B}_points{points}_dt{dt}.npz', rs_U=rs_U)
    np.savez(f'H_N{N}_theta{theta}_B{B}_points{points}_dt{dt}.npz', rs_H=rs_H)
#%% 
# define colormap
colors = plt.cm.jet(np.linspace(0,1,5))
theta = thetas[-1]
plt.figure(figsize=(16,8))
ax = plt.gca()
# plt.title(r'r-chaometer para $U^n$')
for i,N in enumerate(range(6,11)):
    rs_U = np.load(f'Un{pot}_N{N}_theta{theta}_B{B}_points{points}_dt{dt}.npz')['rs_U']
    rs_H = np.load(f'H_N{N}_theta{theta}_B{B}_points{points}_dt{dt}.npz')['rs_H']
    plt.plot(thetas,rs_H-rs_U, '^-', color=colors[i], ms=1, lw=1.5, label=f'N={N}')
plt.xlabel(r'$\theta$')
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 8))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 8))
# plt.xlabel(['0',r'$\pi/8$',r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
# plt.xlabel(r'$h_z$')
plt.ylabel(r'$\Delta r$ (chaometer) ')
# plt.ylim(-0.1,1.1)
plt.grid(True)
plt.legend(loc='best')

# plt.savefig(f'HorU_N{N}_r_vs_hz_{points}points_sinnorm_lims_{min(zs)}_{max(zs)}_dt{dt}.png', dpi=80)#_Nico
plt.savefig(f'Delta_rchaometer_theta_H_UdeH_N_{min(Ns)}_to_{max(Ns)}_B{B:.3f}_r_vs_hz_{points}points_sinnorm_lims_{min(thetas):.3f}_{max(thetas):.3f}_dt{dt}.png', dpi=80)#_Nico