#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 18:59:19 2021

@author: tomas.notenson
"""
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time

sx=sigmax()
sy=sigmay()
sz=sigmaz()

# defino algunas funciones que me van a servir

def pij(l,i,j):
    geye=[qeye(2) for k in range(l)]
    
    H=0*tensor(geye)
    g=geye.copy(); g[i]=sx;g[j]=sx; H+= tensor(g)
    g=geye.copy(); g[i]=sy;g[j]=sy; H+= tensor(g)
    g=geye.copy(); g[i]=sz;g[j]=sz; H+= tensor(g)
    
    H+=tensor(geye); return H/2
    
def ciclic(l):
    geye=[qeye(2) for k in range(l)]
    P=tensor(geye)
    for m in range(l-1):
        P=pij(l,m,m+1)*P
    return P

#%% Prueba
# import numpy.linalg as lalg

#1) elijo numero de spines
ns = [12,13]

ecos = []

times = np.linspace(0.0, 1000.0, 2000)

start_time0 = time.time()
for itera, n in enumerate(ns):
    #2) armo hamiltoniano deseado (con simetria ciclica)
    # X=sum([tensor([qeye(2) if j!=k else sx for j in range(n)]) for k in range(n)])
    # H=X
    
    #3) armo operador S "proyector" sobre simetria
    Z=ciclic(n)
    S=1/n*sum([Z**j for j in range(n)])
    
    # print("[H,S]=",commutator(H,S).norm())
    
    #4) armo operador "rectangular" que me reduce
    # la,v = lalg.eigh(S.full())
    # rv = [v[:,i] for i in range(len(la)) if abs(la[i]-1)<1e-10]
    # Qdag=Qobj(rv);Q=Qdag.dag()
    
    # # REDUZCO
    # RH=Qdag*Qobj(H.full())*Q
    # Circular H0
    
    # Armo H0 ################################################################# 
    
    # Armo el hamiltoniano deseado (con simetría cíclica)
    XX = sum([tensor([qeye(2) if j!=k and j!=(k+1) else sx for j in range(n)]) for k in range(n-1)],tensor([qeye(2) if j!=(n-1) and j!=0 else sx for j in range(n)]))
    Z=sum([tensor([qeye(2) if j!=k else sz for j in range(n)]) for k in range(n)])
    h = 0.4
    H0 = XX + h*Z
    # print("[H,S]=",commutator(H0,S).norm())
    
    # Calculo las energias y sus correspondientes autoestados
    e0, evec0 = H0.eigenstates()
    
    # Armo H1 ################################################################# 
    
    # Circular H1 
    zN = tensor([qeye(2) if j!=(n-1) else sz for j in range(n)])
    l = 0.2
    H1 = H0 + l*zN
    
    # Defino el operador paridad
    paridad = tensor([sz for j in range(n)])
    
    # Muestro que ya no tiene simetría cíclica pero sí paridad
    # print("[H,S]=",commutator(H1,S).norm())
    # print(r"[H,Pi^z]=",commutator(H1,paridad).norm())
    
    # Calculo las energias y sus correspondientes autoestados
    e1, evec1 = H1.eigenstates()
    
    # Calculo el eco y lo ploteo
    
    # Calculo el eco y lo ploteo ############################################### 
    psi0 = evec0[0]
    
    result = mesolve(H1, psi0, times, [], [])
    states = result.states
    prob = [abs(states[i].overlap(evec0[0]))**2 for i in range(len(states))]
    # prob = ecos[itera] 
    
    ecos.append(prob)    
    
    plt.figure()
    plt.title('Loschmidt echo in Ising with tranverse field PBC N = %i'%n)
    plt.plot(times/(n**2), prob, label=r'$\mathtt{L}(t)$')
    plt.legend()
    
    # Calculo ⟨n|psi0⟩ con evec1[n] = |n⟩ y ploteo #############################
    dbase = len(evec1)
    overlaps = [abs((evec1[j].dag()*psi0)[0,0])**2 for j in range(dbase)]
    
    plt.figure()
    plt.title('⟨n|\psi_0⟩$ in Ising with tranverse field PBC N = %i'%n)
    plt.plot(np.arange(dbase), overlaps, '.', label=r'$⟨n|\psi_0⟩$')
    plt.legend()
    print("N = %i--- %s seconds ---" % (n, time.time() - start_time0))
print("Total --- %s seconds ---" % (time.time() - start_time0))
np.savez('LE_N%i,%i.npz'%tuple(ns), ecos=ecos)
#%% Comparación
ecos = np.load('LE_N%i,%i.npz'%tuple(ns))['ecos']

colores = ['r-', 'b-']

plt.figure()
plt.title(r'Loschmidt echo in Ising with tranverse field PBC N = %i and N = %i '%tuple(ns))
for ind, [num, prob] in enumerate(zip(ns, ecos)):    
    plt.plot(times, prob, colores[ind], label=r'$\mathtt{L}_{%i}(t)$'%num)
plt.legend()
plt.savefig('Comparacion_LE_N%i,%i'%tuple(ns))


#%% teorico para N = 2k+1 #####################################################
# n = 5

# Calculo el eco de Loschmidt con la expresión teórica
# Ls = np.zeros((len(times)))
# for inde, t in enumerate(times):
#     term = [np.tan((2*k-1)*np.pi/(2*(n-1)))**2 * np.exp(-1j*2*h*t*np.cos((2*k-1)*np.pi/(n-1))) for k in range(int((n-1)/2))]
#     Laux = 2/(n*(n-1)) * sum(term) + 2/n *   np.exp(1j*t*(l+h))
#     L = abs(Laux)**2
#     Ls[inde]=L
    
# plt.figure()
# plt.title('Loschmidt echo analytic')
# plt.plot(times,Ls)