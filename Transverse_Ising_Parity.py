#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 21:48:12 2021

@author: tomasnotenson
"""
from qutip import *
import numpy as np


def spectrum(N, hx, hz, Jx, Jy, Jz):

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
        H += hz[n] * sz_list[n]
        
    for n in range(N):
        H += hx[n] * sx_list[n]

    # interaction terms
    for n in range(N-1):
        H += - Jx[n] * sx_list[n] * sx_list[n+1]
        H += - Jy[n] * sy_list[n] * sy_list[n+1]
        H += - Jz[n] * sz_list[n] * sz_list[n+1]
    
    return H.eigenstates()

# Calcultes r parameter in the 10% center of the energy "ener" spectrum. If plotadjusted = True, returns the magnitude adjusted to Poisson = 0 or WD = 1

def r_chaometer(ener,normed=False):
    ra = np.zeros(len(ener)-2)
    #center = int(0.1*len(ener))
    #delter = int(0.05*len(ener))
    for ti in range(len(ener)-2):
        ra[ti] = (ener[ti+2]-ener[ti+1])/(ener[ti+1]-ener[ti])
        ra[ti] = min(ra[ti],1.0/ra[ti])
    ra = np.mean(ra)
    if normed == True:
        ra = (ra-0.3863) / (0.5307-0.3863)
    return ra


N=14
hx  = np.ones(N)
hz  = 0.5 *  np.ones(N) 
Jx =  0*np.ones(N)
Jy =  0*np.ones(N)
Jz = np.ones(N)


ene,est=spectrum(N, hx,hz,Jx, Jy, Jz)

impar_ene=[]
par_ene=[]

for x in range(2**N): # x in elementos en Hilbert
    # calculo la paridad a mano para cada autoestado
    num = (est[x].permute(np.arange(0,N)[::-1]).dag()*est[x])[0,0].real
    if num < 0:
        impar_ene.append(ene[x])
    else:
        par_ene.append(ene[x])
        
# Calculo el caos con chaometer
r_par=r_chaometer(par_ene, normed=True)
r_impar=r_chaometer(impar_ene, normed=True)
print(f"Mi r par Ising Tilter N={N} ", r_par)        
print(f"Mi r impar Ising Tilter N={N} ", r_impar)       
