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

def a_can(n,i):
    dim = 2**n
    if i < dim:
        a = {};
        a[i-1] = 1
    return a

def a_prod_esc(a,k):
    for key in a.keys():
        a[key] *= k
    return a


#
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
    
def comp_to_a(vec):
    indx = np.nonzero(vec)
    norm = len(indx[0])
    coef = 1/np.sqrt(norm)
    a = {};
    for ind in indx[0]:
        a[ind-1] = coef
    return a

def proy(a1,a2): # keys son numeros entre 0 y 2^n 
    
    idx = list(set(a1.keys()) & set(a2.keys())) #intersection
    
    val = sum([a1[key]*a2[key] for key in idx])

    return val

# print(proy(a,ap))

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

#
def scalar2dic(k,a):     return {key:a[key]*k for key in a.keys()}

def Zona(a,n): 
    
    ap={}
    for u,coef in a.items():
        
        s = to_bs(u,n); 
        exc = np.count_nonzero(s) 
        nz = exc - (len(s)-exc)
        
        ap[u] = nz*coef
        
    return ap
# X

def Xonu(u,n):
    s = to_bs(u,n);
    ups = []
    for i in range(len(s)):
        sp = s.copy()
        # print()
        # print(sp)
        sp[i]=(s[i]+1)%2
        # print(sp)
        up = to_num(sp)
        ups.append(up)
    return ups

def Xona(a,n): 
    
    ap={}
    
    for u,coef in a.items():
        
        ups = Xonu(u,n)
        # print(ups)
        for up in ups:
            if not (up in ap.keys()):
                ap[up] = coef
            else:
                ap[up] += coef
    # print(proy(ap,ap))
    return ap


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
#%% separate symmetries (1st option)


# define parameters of Heisenberg chain with inclined field 
n = 8
dim = 2**n

start_time = time()
g_can = [a_can(n,i) for i in range(0, dim)]
g_par = []
for i in range(dim):
    s = to_bs(list(set(g_can[i].keys()))[0], n)
    s_per = parity(s)
    a_per = {}; u_per = to_num(s_per); a_per[u_per] = 1
    a_par = suma(g_can[i], a_per); a_par = a_prod_esc(a_par, 1/2); a_par = a_prod_esc(a_par, 1/np.sqrt(proy(a_par,a_par)))
    g_par.append(a_par)

g=[]

for i in g_par:
    if i not in g:
        g.append(i)  

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
# n=2

# g=[]
# a={} ;
# s = [0,0]; u = to_num(s) ; coef=1 ; a[u]=coef; 
# g.append(a)
# a={} ; 
# s = [1,0]; u = to_num(s) ; coef=1/np.sqrt(2) ; a[u]=coef; 
# s = [0,1]; u = to_num(s) ; coef=1/np.sqrt(2) ; a[u]=coef;
# g.append(a)
# a={} ;
# s = [1,1]; u = to_num(s) ; coef=1 ; a[u]=coef; 
# g.append(a)



# express(g,n)
#%% Prueba
# g = []
# for col in range(bas.shape[1]):
#     a = comp_to_a(bas[:,col])
#     g.append(a)
#%%

# =============================================================================
# Notación
# #a dic
# #s bitstring
# #u numerito
# 
# <b Z a>
# 
# a=g[0]  
# b=g[1]
# 
# # |a'>= Mz a>
# =============================================================================


        
# Z prueba

# for i,a in enumerate(g[:-1]):
#     #|a'>= Mz a>
#     ap = Zona(a,n)
#     express([a,ap],n)
   
#     print('\n\n')
    



# X prueba
# 
# for i,a in enumerate(g):
#     #|a'>= Mz a>
#     ap = Xona(a,n)
#     express([a,ap],n)
   
#     print('\n\n')
    

# ZZ prueba

# for i,a in enumerate(g):
#     #|a'>= Mz a>
#     ap = ZZona(a,n)
#     express([a,ap],n)
   
#     print('\n\n')

#%% Create hamiltonian in parity subspace
t0 = time()
dimg = len(g)

H = np.zeros((dimg, dimg), dtype=np.complex_)

for row in range(dimg):
    # row = row-1
    for column in range(dimg):
        # column = column-1
        # print(row, column)
        # print("\nrow")
        # print(express([g[row]],n))
        # print("\ncolumn")
        # print(express([g[column]],n))
        H0 = suma(Xona(g[column],n),Zona(g[column],n))
        # print("\nXona")
        # print(express([Xona(g[column],n)],n))
        # print("\nZona")
        # print(express([Zona(g[column],n)],n))
        # print("\nH0")
        # print(express([H0],n))
        H1 = suma(H0,ZZona(g[column],n))
        # print("\nZZona")
        # print(express([ZZona(g[column],n)],n))
        # print("\nH1")
        # print(express([H1],n))
        H[row, column] = proy(g[row], H1)
        # print('\nH')
        # print(H[row,column])

# print()
print(H)

Hdic = H
print(f'Tiempo Hamiltoniano: {time()-t0} s')
np.savez(f'TFIM_N{n}_par1.npz', H = H)