#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 19:51:49 2022

Chaos transition in TFIM model

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
from scipy.fft import fft, ifft, fftfreq

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

def EvolutionO1_H_KI_Tinf(H, time_lim, N, A, B):
    start_time = time() 
    
    # define arrays for data storage
    O1s = np.zeros((time_lim), dtype=np.complex_)
        
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
            
        # dim = A.shape[0]        
        # compute 4-point correlator with qutip
        # O1 = (B_t*A*B_t*A).tr()
        
        # compute 4-point correlator with numpy
        O1 = O1_numpy_Tinf(A, B_t)

        # store data
        O1s[i] = O1

        
    print(f"\nTOTAL --- {time() - start_time} seconds ---" )
    flag = 'O1_H_KI_with_Tinf_state'
    return [O1s, flag]

def EvolutionO1_U_KI_Tinf(U, time_lim, N, A, B):
    
    start_time = time() 
    
    # define arrays for data storage
    O1s = np.zeros((time_lim), dtype=np.complex_)
    
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
            
        # dim = A.shape[0]
        # compute 4-point correlator with qutip
        # O1 = (B_t*A*B_t*A).tr()
        
        # compute 4-point correlator with numpy
        O1 = O1_numpy_Tinf(A, B_t)
   
        # store data
        O1s[i] = np.abs(O1)

        
    print(f"\nTOTAL --- {time() - start_time} seconds ---" )
    flag = 'O1_U_KI_with_Tinf_state'
    return [O1s, flag]

# @jit(nopython=True, parallel=True, fastmath = True)
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

# @jit(nopython=True, parallel=True, fastmath = True)
def diagH_r(H_sub):
    ener = np.linalg.eigvals(H_sub)
    ener = np.sort(ener)
    # histo_level_spacing(fases)
    r_normed = r_chaometer(ener, plotadjusted=True) 
    return r_normed

# @jit(nopython=True, parallel=True, fastmath = True)
def diagU_r(U_sub):
    ener = np.linalg.eigvals(U_sub)
    # ener = np.sort(ener)
    fases = [phase(en) for en in ener]
    fases = np.sort(fases)
    # histo_level_spacing(fases)
    r_normed = r_chaometer(fases, plotadjusted=True) 
    return r_normed

def normalize(array):
    return (array - min(array))/(max(array)-min(array))

def TFIM_O1(N, B, J, theta, time_lim):
        
    # define parameters of Heisenberg chain with inclined field 
    
    hx = B*np.ones(N)*np.sin(theta)
    hz = B*np.ones(N)*np.cos(theta)
    Jz = J*np.ones(N)
    
    start_time = time()
    # let's try it
    
    ######## H evolution ##########
    H = TFIM(N, hx, hz, Jz)
    ######## U evolution ##########
    U = fU(N, Jz, hx, hz)
    
    A = Sj(N)
    B = Sj(N, j='x')
    
    opA = 'Z'
    opB = 'X'
    
    operatorss = '_A'+opA+'_B'+opB
    
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
    
    r_H_normed = diagH_r(H_sub)
    r_U_normed = diagU_r(U_sub)
    
    start_time = time()
    #
    
    O1s_U, flag = EvolutionO1_U_KI_Tinf(U_sub, time_lim, N, A_sub, B_sub)
    np.savez('contributions_O1'+flag+f'_time_lim{time_lim}_J{J:.2f}_hx{np.mean(hx):.2f}_hz{np.mean(hz):.2f}_basis_size{N}'+operatorss+'.npz', O1s=O1s_U)
    
    O1s_H, flag = EvolutionO1_H_KI_Tinf(H_sub, time_lim, N, A_sub, B_sub)
    np.savez('contributions_O1'+flag+f'_time_lim{time_lim}_J{J:.2f}_hx{np.mean(hx):.2f}_hz{np.mean(hz):.2f}_basis_size{N}'+operatorss+'.npz', O1s=O1s_H)
    
    
    print(f"\n Evolution O1 --- {time() - start_time} seconds ---" )
    
    return [O1s_H, O1s_U, r_H_normed, r_U_normed]

def undo_rplotadjust(ra):
    ra = ra*(-0.3863+0.5307) +0.3863
    return ra
#%% try it

# define chain length and then basis size
N = 8
dim = 2**N

# field and interaction
B = 1
J = 1 

thetas = np.linspace(0,1/2,100)*np.pi

time_lim = 6000

O1_H_arr = np.zeros((time_lim, len(thetas)), dtype=np.complex_)
O1_U_arr = np.zeros((time_lim, len(thetas)), dtype=np.complex_)
r_H_arr = np.zeros((len(thetas)))
r_U_arr = np.zeros((len(thetas)))
for ang in tqdm(range(len(thetas)), desc='theta loop'):
    
    theta = thetas[ang]
    O1s_H, O1s_U, r_H_normed, r_U_normed = TFIM_O1(N, B, J, theta, time_lim)
    
    O1_H_arr[:,ang] = O1s_H
    O1_U_arr[:,ang] = O1s_U
    r_H_arr[ang] = r_H_normed
    r_U_arr[ang] = r_U_normed
np.savez(f'TFIM_O1_H_U_J{J}_B{B}_thetamin{min(thetas):.4f}_thetamax{max(thetas):.4f}_timelim{time_lim}_Nspins{N}.npz', O1_H_arr=O1_H_arr, O1_U_arr=O1_U_arr, r_H_arr=r_H_arr, r_U_arr=r_U_arr, thetas=thetas)
#%% Fourier analysis
archives = np.load(f'TFIM_O1_H_U_J{J}_B{B}_thetamin{min(thetas):.4f}_thetamax{max(thetas):.4f}_timelim{time_lim}_Nspins{N}.npz')#.npz')#'

O1_H_arr=archives['O1_H_arr']
O1_U_arr=archives['O1_U_arr']
r_H_arr=np.real(archives['r_H_arr'])
r_U_arr=np.real(archives['r_U_arr'])
thetas=archives['thetas']

O1 = np.abs(O1_U_arr)/N

t_sat = 500

def IPR_normed(vec):
    PR = np.sum(np.abs(vec)**2)**2
    nrm2 = np.sum(np.abs(vec))**2
    IPR = nrm2/PR
    return IPR

IPRs = np.zeros((len(thetas)))

for k in range(len(thetas)):
    y1 = O1[:,k]
    O1_fourier = fft(y1[t_sat:])
    M = len(y1[t_sat:])
    xf = fftfreq(M, t_sat)[:M//2]
    yf = 2.0/M * np.abs(O1_fourier[0:M//2])
    IPRs[k] = IPR_normed(yf)

np.savez(f'IPR_TFIM_O1_U_J{J}_B{B}_thetamin{min(thetas):.4f}_thetamax{max(thetas):.4f}_timelim{time_lim}_Nspins{N}.npz',IPRs=IPRs)

y_IPR = normalize(IPRs)
r_H_one = normalize(r_H_arr[1:])
# r_H_one = undo_rplotadjust(r_H_arr[1:])
r_U_one = normalize(r_U_arr[1:])
# r_U_one = undo_rplotadjust(r_U_arr[1:])

plt.figure(figsize=(16,8))
# plt.title(f'IPR O1')
plt.plot(thetas, y_IPR, '-b', lw=1.5, label='IPR')
plt.plot(thetas[1:], r_U_one, '-g', lw=1.5, label=r'$r$ kicked')
plt.plot(thetas[1:], r_H_one, '-r', lw=1.5, label=r'$r$ not kicked')
# plt.plot(Kx, r_normed, '-r', lw=1.5, label='r')
# plt.plot(Ks, 1-r_one, '-g', lw=1.5, label='1-r paridad +1')
plt.xlabel(r'$\theta$')
plt.ylabel('IPR')
# plt.ylim(0,1)
# plt.xlim(-0.2,max(times)+0.2)
plt.grid(True)
plt.legend(loc = 'best')
plt.tight_layout()
plt.savefig(f'IPR_O1_U_J{J}_B{B}_thetamin{min(thetas):.4f}_thetamax{max(thetas):.4f}_timelim{time_lim}_Nspins{N}.png', dpi=80)


