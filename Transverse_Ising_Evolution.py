#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:09:06 2021

@author: tomasnotenson
"""
from qutip import *
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
import time

# OTOC

def H_inclined(N, hx, hz, theta, Jx, Jy, Jz):

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
        H += hz[n] * np.cos(theta) * sz_list[n]
        
    for n in range(N):
        H += hx[n] * np.sin(theta) * sx_list[n]

    # interaction terms
    for n in range(N-1):
        H += - Jx[n] * sx_list[n] * sx_list[n+1]
        H += - Jy[n] * sy_list[n] * sy_list[n+1]
        H += - Jz[n] * sz_list[n] * sz_list[n+1]
    
    return H

N = 8
B = 2
J = 2
hx = B*np.ones(N)
hz = B*np.ones(N) 
Jx = 0*np.ones(N)
Jy = 0*np.ones(N)
Jz = J*np.ones(N)
# theta = 0.79*np.pi/2
thetas = np.array([0.08, 0.31, 0.79])*np.pi/2
for theta in thetas:
        
    
    H = H_inclined(N,hx,hz,theta,Jx, Jy, Jz)
    e, evec = H.eigenstates()
    
    # elijo sz como operador a tiempo 0
    
    op_list = [qeye(2) for m in range(N)]
            
    op_list[0] = sigmaz()
    sz_list = (tensor(op_list))
    
    l = 1 # evoluciono sz_1
    
    op_list = [qeye(2) for m in range(N)]
            
    op_list[l] = sigmaz()
    sz_l = (tensor(op_list))
        
    times = np.linspace(0.0, 1000.0, 1000)
    start_time0 = time.time()
    
    OTOC = []
    for i in range(len(times)):
        t = (times[i]-times[0])
        start_time = time.time()
    
        
        U = 0
        for j in range(len(e)):
            U+=evec[j]*np.exp(-complex(0,1)*t*e[j])*evec[j].dag()
        
        print("--- %s seconds ---" % (time.time() - start_time))
        # U = Qobj(np.diag(np.exp(-complex(0,1)*t*e)), dims=[[2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2]])
        # U = qdiags(-complex(0,1)*t*e, offsets=0, dims=[[2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2]])
        # U = (-complex(0,1)*t*H).expm()
        sz_t = U.dag()*sz_l*U
        com = commutator(sz_t,sz_list)
        prom = (com.dag()*com/(2**N)).tr()
        OTOC.append(prom)
    
    print("--- %s seconds ---" % (time.time() - start_time0))
    # plot 
    OTOC = OTOC/np.mean(OTOC)
    
    np.savez('OTOC_N%i_l%i_theta%.4f.npz'%(N,l,theta/(np.pi/2)), OTOC=OTOC)
    
    fig, ax = plt.subplots()
    ax.plot(times, OTOC, label=r"$\sigma^z_1$");
    ax.set_xlabel('Time');
    ax.set_ylabel(r'$C^{zz}_{10}$');
    ax.grid();
    ax.legend();
    plt.show()
#%% 
import fnmatch
import os
import pandas as pd

lista = [f for f in os.listdir("/home/tomasnotenson/Escritorio/Caos y control cu__ntico/Repo github/quantum-chaos") if fnmatch.fnmatch(f, '*.npz')]

OTOCs = []
for name in lista:
    OTOCs.append(np.load(name)['OTOC'])    
thetas = np.array([0.08, 0.31, 0.79])*np.pi/2
ncol= [f'{thetas[k]/(np.pi/2)}' for k in range(len(thetas))]#+['Time']

OTOCs = np.array(OTOCs, ndmin=1).T

df = pd.DataFrame(data=OTOCs, columns=ncol)
df['Time'] = times

# plt.rc('text', usetex=True, labelsize=16)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
fig, ax = plt.subplots()
# df.plot(x='Time', y=ncol, kind='line')
for num in range(len(ncol)):
    ax.plot(times, OTOCs[:,num], label=r"$\theta = %s \frac{\pi}{2}$"%ncol[num]);
plt.title(r'Ising chain in a inclined magnetic field with angle $\theta$')
ax.set_xlabel(r'Time $t$');
ax.set_ylabel(r'OTOC $C^{zz}_{10}$');
ax.grid();
ax.legend();
plt.show()
plt.savefig('OTOC_first_try.png', dpi=400)