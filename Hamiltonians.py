#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 22:25:44 2021

Chain's Hamiltonians

@author: tomasnotenson
"""
# import librarys

from qutip import *
import numpy as np

# define hamiltonians 

def Heisenberg_inclined_field(N, hx, hz, theta, Jx, Jy, Jz):

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
        H += hz[n] * np.cos(theta) * sz_list[n]
        
    for n in range(N):
        H += hx[n] * np.sin(theta) * sx_list[n]

    # interaction terms
    for n in range(N-1):
        H += - Jx[n] * sx_list[n] * sx_list[n+1]
        H += - Jy[n] * sy_list[n] * sy_list[n+1]
        H += - Jz[n] * sz_list[n] * sz_list[n+1]
    
    return H

# define parameters of Heisenberg chain with inclined field 
N = 4
B = 2 
J = 2
hx = 0*np.ones(N)
hz = B**np.ones(N) #np.random.uniform(-B,B) #random field in z 
theta = 0 # angle of the field
Jx = 0*np.ones(N)
Jy = 0*np.ones(N)
Jz = J*np.ones(N)

# let's try it
H = Heisenberg_inclined_field(N, hx, hz, theta, Jx, Jy, Jz)
e, ev = H.eigenstates()

