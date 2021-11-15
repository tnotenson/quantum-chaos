#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 16:24:31 2021

@author: tomasnotenson
"""
from scipy.linalg import block_diag

sites=3
n_partic=1
BC='open'
hz=2*np.ones(sites)
def Heisenberg_sz_subspace(sites,n_partic,BC,hz):
    H = .25*sz_subspace_spin_interactions(sites,n_partic,1,BC,1,1,1)
    for x in range(len(hz)):
        H+= .5*hz[x]*sz_subspace_S_zi(x,sites,n_partic) 
    e, ev = np.linalg.eigh(H)
    return e, ev, H

Hs = []
for i in range(sites+1):
    print(i)
    ener0, basis0,Haux = Heisenberg_sz_subspace(sites,i,BC,hz)
    Hs.append(Haux)
H_emi=Qobj(block_diag(*Hs))
H_emi.dims = [[2 for i in range(sites)] for j in range(2)]
print("Hamiltoniano Emi",H_emi)
#%%
def Heisenberg_emi(sites,BC,hz):
    H = .25*spin_interactions(sites,1,BC,1,1,1)
    for x in range(len(hz)):
        H+= .25*hz[x]*S_z(sites)
    e, ev = np.linalg.eigh(H)
    return e, ev,H
print("Hamiltoniano Emi2",Qobj(Heisenberg_emi(sites,BC,hz)[2]))
#%%
def Heisenberg(N,h):

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

    H = 0
    for n in range(N):
        H += 0.5*h[n] * sz_list[n]
    for n in range(N-1):
        H +=  0.25* sx_list[n] * sx_list[n+1]
        H +=  0.25* sy_list[n] * sy_list[n+1]
        H +=  0.25* sz_list[n] * sz_list[n+1]
        return H
    
H = Heisenberg(sites, hz)
print(H)