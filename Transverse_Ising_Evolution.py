#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:09:06 2021

@author: tomasnotenson
"""
from qutip import *
import numpy as np
from matplotlib import pyplot as plt
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
theta = np.pi/4

H = H_inclined(N,hx,hz,theta,Jx, Jy, Jz)

# elijo sz

op_list = [qeye(2) for m in range(N)]
        
op_list[0] = sigmaz()
sz_list = (tensor(op_list))
    
times = np.linspace(0.0, 10.0, 1000)

OTOC = []
for i in range(len(times)):
    t = (times[i]-times[0])
    U = (-complex(0,1)*t*H).expm()#np.diag(np.exp(complex(0,1)*t*ene))
    sz_t = U.dag()*sz_list*U
    com = commutator(sz_t,sz_list)
    prom = (com.dag()*com/N).tr()
    OTOC.append(prom)

# plot 

fig, ax = plt.subplots()
ax.plot(times, OTOC, label=r"$\sigma^z_1$");
ax.set_xlabel('Time');
ax.set_ylabel('OTOC');
ax.grid();
ax.legend();
plt.show()