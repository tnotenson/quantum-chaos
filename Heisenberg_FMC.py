#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 20:45:56 2022

Física de Muchos Cuerpos - 1er cuatrimestre 2022

@author: Tomas Notenson
"""

import matplotlib.pyplot as plt
import numpy as np
from qutip import *
import seaborn as sns; sns.set_theme()
# from math import factorial
# from time import time
# from tqdm import tqdm # customisable progressbar decorator for iterators
# from numba import jit
# importing "cmath" for complex number operations
# from cmath import phase
plt.rcParams.update({
"text.usetex": True,
"font.family": "sans-serif",
"font.sans-serif": ["Helvetica"], "font.size": 24})

# define usefull one body operators 
sx=sigmax()
sy=sigmay()
sz=sigmaz()
si=qeye(2)

# construct the spin operators for each site
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
        
def Heisenberg_PBC(N, Jx, Jy, Jz, Hx, Hy, Hz):
    # construct the hamiltonian
    H = 0

    # energy splitting terms      
    for n in range(N):
        H += -0.5 * Hx[n] * sx_list[n]
        H += -0.5 * Hy[n] * sy_list[n]
        H += -0.5 * Hz[n] * sz_list[n]
        
    # interaction terms
    #PBC
    for n in range(N):
        H += -0.25 * Jx[n] * sx_list[n] * sx_list[(n+1)%N]
        H += -0.25 * Jy[n] * sy_list[n] * sy_list[(n+1)%N]
        H += -0.25 * Jz[n] * sz_list[n] * sz_list[(n+1)%N]
    #OBC
    # for n in range(N-1):
    #     H += J[n] * sz_list[n] * sz_list[n+1]
    
    return H

N = 6
dim = 2**N

jx = 1; jy = 1; jz = 1
hx = 0; hy = 0; hz = 0

Jx = jx*np.ones(N); Jy = jy*np.ones(N); Jz = jz*np.ones(N)
Hx = hx*np.ones(N); Hy = hy*np.ones(N); Hz = hz*np.ones(N)


H = Heisenberg_PBC(N, Jx, Jy, Jz, Hx, Hy, Hz)

# eigenvalues and eigenvectors
# evs, evecs = H.eigenstates()
# xplt = np.arange(dim)
# plt.plot(xplt, evs, '.r')

# expectation values of sigma z on site j

# intial state, first spin in state |1>=|down>, the rest in state |0>=|up>
psi_list = []
psi_list.append(basis(2,1))
for n in range(N-1):
    psi_list.append(basis(2,0))
psi0 = tensor(psi_list)

# time array
tlist = np.linspace(0, 50, 200)

# collapse operators
c_op_list = []

result = mesolve(H, psi0, tlist, c_op_list, sz_list)

sz_expt = result.expect

plt.figure(figsize=(10,6))

for n in range(N):
    plt.plot(tlist, np.real(sz_expt[n]), label=r'$\langle\sigma_z^{(%d)}\rangle$'%(n+1), alpha=0.9)

plt.legend(loc='best')
plt.xlabel(r'Time [ns]')
plt.ylabel(r'$\langle\sigma_z\rangle$')
plt.title(r'Dynamics of a Heisenberg spin chain N={}'.format(N));


#%%