#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:10:18 2021

@author: larox
"""
from time import time
from qutip import tensor,sigmax,sigmay,sigmaz,qeye; sx,sy,sz=sigmax(),sigmay(),sigmaz()
from qutip import Qobj
import numpy as np
from itertools import product


def can_bas(N,i):
    e = np.zeros(N)
    e[i] = 1.0
    return e

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
    
    U = (-1j*H1).expm()*(-1j*H0).expm()
    
    return U

def Sj(N, j='z'):
    s_list = []
    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(qeye(2))
        if j == 'z':
            op_list[n] = sz
        elif j == 'x':
            op_list[n] = sx
        elif j == 'y':
            op_list[n] = sy
        s_list.append(tensor(op_list))
    return sum(s_list)

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
    for n in range(N):
        H += J[n] * 0.5 * sz_list[n] * sz_list[(n+1)%N]
    
    return H

#%%

# define parameters of Heisenberg chain with inclined field 
n = 2
dim = 2**n
B = 1.4
J = 1
hx = B*np.ones(n)
hz = B*np.ones(n)
Jz = J*np.ones(n)

j = J
x = np.mean(hx)
z = np.mean(hz)

start_time = time()
# let's try it

# H = TFIM(N, hx, hz, Jz)
U = fU(n, Jz, hx, hz)

A = Sj(n, j='x')#/N



#%% separate symmetries (2nd option)
start_time = time()
e_basis = [Qobj(can_bas(dim,i)) for i in range(dim)]
par_basis_ones = np.zeros((dim,dim), dtype=np.complex_)
for i in range(dim):
    e_basis[i].dims = [[2 for i in range(n)], [1]]
    par_basis_ones[i] = (1/2*(e_basis[i] + e_basis[i].permute(np.arange(0,n)[::-1]))).data.toarray()[:,0]
    norma = np.linalg.norm(par_basis_ones[i])
    if norma != 0:
        par_basis_ones[i] = par_basis_ones[i]/norma
    par = par_basis_ones[i].T@par_basis_ones[i]
#    print(par)
par_basis_ones = np.unique(par_basis_ones, axis=1)
#print(par_basis_ones[:,::-1])
bas = par_basis_ones
A_red = bas.conj().T@A.data.toarray()@bas
U_red = bas.conj().T@U.data.toarray()@bas
#print(U_red)
print(f"\n Separate parity eigenstates --- {time() - start_time} seconds ---" )

#%%

print(bas.shape)

#%%
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
  
def decode(s):
    try:
        return ALPHABET.index(s)
    except ValueError:
        raise Exception("cannot dencode:{}".format(s))

def base_to_dec(s, base = 16, pow = 0):
    if s == "":
        return 0
    else:
        return decode (s[-1]) * (base ** pow) + base_to_dec (s[0:-1], base, pow + 1)
    
def pstos(p):
    p=[str(k) for k in p];     s="".join(p);     return s
    
def to_num(p0): #p string to u number
    p=p0.copy();     s=pstos(p);     u=base_to_dec(s,base=2);     return u-1

def to_bs(x,n,base=2): # binary to string
    x=x+1
    if x == 0:
        return [0 for k in range(n)]

    digs = []
    while x:
        digs.append(int(x % base))
        x = int(x / base)

    for k in range(n-len(digs)):
        digs.append(0)
    digs.reverse()

    return digs

def express(g,n): #print the algebra
    dimg=len(g)
    for i in range(dimg):
        print('\nElement {}'.format(i))
        for u in g[i].keys():
            print(to_bs(u,n),g[i][u])
            
# Swap function
def swapPositions(lista, i, j):
    lista[i], lista[j] = lista[j], lista[i]
    return lista

def parity(lista):
    L = len(lista)
    
    if L%2==0: imax = int(L/2)
    else: imax = int((L-1)/2)
        
    for i in range(imax): lista = swapPositions(lista, i, L-1-i)
    return lista
        
#%%
# n=3
# s = to_bs(to_num([1,1,1]),n)

# g=[]
# a={} ;
# s = [0,0,0]; u = to_num(s) ; coef=1 ; a[u]=coef; 
# g.append(a)
# a={} ; 
# s = [1,0,0]; u = to_num(s) ; coef=1/np.sqrt(2) ; a[u]=coef; 
# s = [0,0,1]; u = to_num(s) ; coef=1/np.sqrt(2) ; a[u]=coef;
# g.append(a)
# # a={} ; 
# # s = [1,0,0]; u = to_num(s) ; coef=1/np.sqrt(2) ; a[u]=coef; 
# # s = [0,0,1]; u = to_num(s) ; coef=-1/np.sqrt(2) ; a[u]=coef;
# # g.append(a)
# a={} ; 
# s = [0,1,0]; u = to_num(s) ; coef=1 ; a[u]=coef; 
# g.append(a)
# a={} ; 
# s = [1,1,0]; u = to_num(s) ; coef=1/np.sqrt(2) ; a[u]=coef; 
# s = [0,1,1]; u = to_num(s) ; coef=1/np.sqrt(2) ; a[u]=coef;
# g.append(a)
# a={} ; 
# s = [1,0,1]; u = to_num(s) ; coef=1 ; a[u]=coef; 
# g.append(a)
# a={} ;
# s = [1,1,1]; u = to_num(s) ; coef=1 ; a[u]=coef; 
# g.append(a)
# # a={} ; 
# # s = [1,1,0]; u = to_num(s) ; coef=1/np.sqrt(2) ; a[u]=coef; 
# # s = [0,1,1]; u = to_num(s) ; coef=-1/np.sqrt(2) ; a[u]=coef;
# # g.append(a)



# express(g,n)

#%%
n=2

g=[]
a={} ;
s = [0,0]; u = to_num(s) ; coef=1 ; a[u]=coef; 
g.append(a)
a={} ; 
s = [1,0]; u = to_num(s) ; coef=1/np.sqrt(2) ; a[u]=coef; 
s = [0,1]; u = to_num(s) ; coef=1/np.sqrt(2) ; a[u]=coef;
g.append(a)
a={} ;
s = [1,1]; u = to_num(s) ; coef=1 ; a[u]=coef; 
g.append(a)



express(g,n)
#%%

# =============================================================================
# #a dic
# #s bitstring
# #u numerito
# 
# <b Mz a>
# 
# a=g[0]  
# b=g[1]
# 
# # |a'>= Mz a>
# =============================================================================

#%%
def scalar2dic(k,a):     return {key:a[key]*k for key in a.keys()}

def Zona(a,n): 
    
    ap={}
    for u,coef in a.items():
        
        s = to_bs(u,n); 
        exc = np.count_nonzero(s) 
        nz = exc - (len(s)-exc)
        
        ap[u] = nz*coef
        
    return ap
        
#%% Z

for i,a in enumerate(g[:-1]):
    #|a'>= Mz a>
    ap = Zona(a,n)
    express([a,ap],n)
   
    print('\n\n')
    
#%% X

def Xonu(u,n):
    s = to_bs(u,n);
    sp=(np.array(s)+1)%2#np.abs(np.array(s)-1)  
    up = to_num(sp)
    return up

def Xona(a,n): 
    
    ap={}
    for u,coef in a.items():
        
        up = Xonu(u,n)
        
        ap[up] = coef
        
    return ap


#%%
for i,a in enumerate(g):
    #|a'>= Mz a>
    ap = Xona(a,n)
    express([a,ap],n)
   
    print('\n\n')
    
#%%

def proy(a1,a2): # keys son numeros entre 0 y 2^n 
    
    idx = list(set(a1.keys()) & set(a2.keys())) #intersection
    
    val = sum([a1[key]*a2[key] for key in idx])

    return val

print(proy(a,ap))

def suma(a1,a2): # keys son numeros entre 0 y 2^n 
    
    a = {};
    
    idx = list(set(a1.keys()) | set(a2.keys())) #intersection
    
    # print(idx)
    
    for key in idx:
        a[key] = 0
        if key in a1.keys():
            a[key] += a1[key]
        if key in a2.keys():
            a[key] += a2[key]

    return a

print(express([suma(g[0],g[0])],n))
#%% ZZ

def ZZona(a,n): #ASSUMES NN
    

    ap={}
    for u,coef0 in a.items():
        
        s = to_bs(u,n); 
        
        coef=0
        for i in range(n-1):
            if s[i]==s[i+1]:
                coef+=1
            else:
                coef-=1
                
        ap[u] = coef*coef0

    return ap
    

#%%

for i,a in enumerate(g):
    #|a'>= Mz a>
    ap = ZZona(a,n)
    express([a,ap],n)
   
    print('\n\n')

#%% Create hamiltonian in parity subspace
dimg = len(g)

H = np.zeros((dimg, dimg), dtype=np.complex_)

for row in range(dimg):
    # row = row-1
    for column in range(dimg):
        # column = column-1
        print(row, column)
        print("\nrow")
        print(express([g[row]],n))
        print("\ncolumn")
        print(express([g[column]],n))
        H0 = suma(Xona(g[column],n),Zona(g[column],n))
        print("\nXona")
        print(express([Xona(g[column],n)],n))
        print("\nZona")
        print(express([Zona(g[column],n)],n))
        print("\nH0")
        print(express([H0],n))
        H1 = suma(H0,ZZona(g[column],n))
        print("\nZZona")
        print(express([ZZona(g[column],n)],n))
        print("\nH1")
        print(express([H1],n))
        H[row, column] = proy(g[row], H1)
        print('\nH')
        print(H[row,column])

print()
print(H)

