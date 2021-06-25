from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time

plt.rcParams.update({
"font.size": 18,
"text.usetex": True,
"font.family": "sans-serif",
"font.sans-serif": ["Helvetica"]})

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

#plt.rcParams.update({
#"text.usetex": True,
#"font.family": "sans-serif",
#"font.sans-serif": ["Helvetica"]})

#1) elijo numero de spines
ns = [9,10]
p0=0
p1=1.
npert=30

plt.figure(figsize=(16, 9), dpi=80)
for i,n in enumerate(ns):
    
    Z=ciclic(n)
    S=1/n*sum([Z**j for j in range(n)])
    XX = sum([tensor([qeye(2) if j!=k and j!=(k+1) else sx for j in range(n)]) for k in range(n-1)],tensor([qeye(2) if j!=(n-1) and j!=0 else sx for j in range(n)]))
    Z=sum([tensor([qeye(2) if j!=k else sz for j in range(n)]) for k in range(n)])
    h =0.3
    H0 = XX + h*Z
    
    ene=[]
    pert=[]
    
    for j in range(0, npert):
        print(j)    
        l =p0+j*(p1-p0)/npert 
        
        zN = tensor([qeye(2) if j!=(n-1) else sz for j in range(n)])
        H1 = H0 + l*zN
        
        e1 = H1.eigenenergies()
        pert.append(l)
        ene.append(e1)
        
    plt.subplot(1,2,i+1) 
    plt.plot(pert,ene,color='black',lw=0.5)
    plt.xlabel(r'$\lambda$ Perturbation parameter')
    plt.ylabel(r'$E$ Energy')
plt.savefig('espectro_frustracion.png', dpi=400)
# plt.title('Spectrum Ising with tranverse field PBC N = %i and N = %i, h=%.1f'%(ns[0], ns[1],h))   


