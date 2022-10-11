#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 20:01:29 2022

@author: tomasnotenson
"""
# import libraries and define rutines
import numpy as np
from tqdm import tqdm
from time import time

dpi = 2*np.pi

def finite_delta(x,s,cant=3):
    '''
    Define coarse-grained delta function

    Parameters
    ----------
    x : array_like
        domain
    s : float
        width
    cant : integer, optional
        number of gaussians in sum. The default is 3.

    Returns
    -------
    d : array_like
        coarse-grained delta values.

    '''
    d = 0
    for npr in range(cant):
        n = npr - cant//2
        # print(n)
        d+= 1/np.sqrt(np.pi*s)*np.exp(-(x-n)**2/s)
    return d

# dom = np.linspace(0,1,100)
# plt.plot(dom,finite_delta(dom,s=0.01))

def mat_elem_U(xp,yp,x,y,K,forma='comun',*args,**kwargs):
    '''
    create matrix elements of Perron-Frobenius operator
    
    <xp,yp|U|x,y>

    Parameters
    ----------
    xp : float
        left position value
    yp : float
        left momentum value
    x : float
        right position value
    y : float
        right momentum value
    K : float
        chaos parameter
    forma : string, optional
        'comun', first free evolution then kick o 'alternativa', first kick then free evolution. The default is 'comun'.
    x : array_like
        domain
    s : float
        width
    cant : integer, optional
        number of gaussians in sum. The default is 3.
    Returns
    -------
    U : float
        Perron-Frobenius operator element

    '''
    x0 = x%1
    y0 = y%1
    if forma=='comun':
        ### forma común
        y1 = (y0 + K/dpi*np.sin(dpi*x0))%1
        x1 = (x0 + y1)%1
    elif forma=='alternativa':
        ### forma alternativa del mapa estándar
        x1 = (x0 + y0)%1
        y1 = (y0 + K/dpi*np.sin(dpi*x1))%1
        
    U = finite_delta((yp - y1)%1,s)*finite_delta((xp - x1)%1,s)
    return U

def qp_from_j(j,N):
    '''
    associate (q,p) with cell j

    Parameters
    ----------
    j : integer
        cell
    N : integer
        grid of N x N cells

    Returns
    -------
    qj : float
        position
    pj : float
        momentum

    '''
    paso = 1/N
    qj = (j%N)*paso
    pj = ((j//N)%N)*paso
    return qj, pj

def eigenvec_j_to_qp(eigenvector, mapa='normal'):
    '''
    change representation of vectors from cell to (q,p)

    Parameters
    ----------
    eigenvector : array_like
        vector in cell representation


    Returns
    -------
    eig_res : array_like
        vector in (q,p) representation

    '''
    N = len(eigenvector)
    Nx = int(np.sqrt(N))
    eig_res = np.zeros((Nx,Nx), dtype=np.complex_)
    for j in range(N):
        q,p = qp_from_j(j, Nx)
        # print(int(q),int(p))
        eig_res[int(Nx*q),int(Nx*p)] = eigenvector[j]
    return eig_res

def Blum_Agam_Perron_Frobenius(*args,**kwargs):
    '''
    Blum Agam algorithm 

    Parameters
    ----------
    j : integer
        cell
    N : integer
        grid of N x N cells
    xp : float
        left position value
    yp : float
        left momentum value
    x : float
        right position value
    y : float
        right momentum value
    K : float
        chaos parameter
    forma : string, optional
        'comun', first free evolution then kick o 'alternativa', first kick then free evolution. The default is 'comun'.
    x : array_like
        domain
    s : float
        width
    cant : integer, optional
        number of gaussians in sum. The default is 3.
    Returns
    -------
    e : array_like
        array of eigenvalues
    evec : array_like
        array of eigenvectors

    '''
    U = np.zeros((N**2,N**2), dtype=np.complex_)
    for i in tqdm(range(N**2), desc='loop 1 celdas'):
        for j in range(N**2):    
            x,y = qp_from_j(i,N)
            xp,yp = qp_from_j(j,N)
            U[i,j] = mat_elem_U(xp,yp,x,y,K,forma='alternativa',s=s)
    # U = U
    t0 = time()
    e, evec = np.linalg.eig(U)
    # e, evec = eigs(U,k=nvec)
    print(f'Diagonalization: {time()-t0} s')
    eabs = np.abs(e)
    evec=evec[:,eabs.argsort()[::-1]]
    e = e[eabs.argsort()][::-1]/eabs[0]
    return e, evec


#%% simulation vs parameter

# Kpaso = 20/20
# Ks = np.arange(18,20.1,Kpaso)

cte = 90*0.001
Npaso = 5
Ns = np.arange(80,81,Npaso)
K = 13 
ss = cte/Ns##np.arange(1,5)*1e-3

for N in tqdm(Ns, desc='N loop'):
    
    # N = Ns[-1]
    nvec = N**2

# for K in Ks:
    
    es = np.zeros((len(ss),nvec), dtype=np.complex_)
            
    for num,s in tqdm(enumerate(ss),desc='loop s'):
          
        es[num,:],_ = Blum_Agam_Perron_Frobenius(N,s,K)
                
    np.savez(f'Blum_Agam_evals_K{K:.1f}_Nsfijo_N{N}_smin{min(ss):.4f}_smax{max(ss):.4f}.npz', es=es, ss=ss)
    del es; #del ss;
#%%