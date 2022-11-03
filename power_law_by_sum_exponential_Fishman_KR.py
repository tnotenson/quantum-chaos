#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 18:20:58 2022

@author: tomasnotenson
"""

import numpy as np
# from numba import jit
from copy import deepcopy
from scipy.special import jv
from scipy.sparse.linalg import eigs
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

dpi=2*np.pi

# @jit
def element_U_Fourier(k,m,kp,mp,K,sigma=0.2,s=1):
    '''
    Compute Perron-Frobenius element in Fourier basis

    Parameters
    ----------
    k : float
        fourier congujate of momentum. Left element
    m : float
        fourier congujate of position. Left element
    kp : float
        fourier congujate of momentum. Right element
    mp : float
        fourier congujate of position. Right element
    K : float
        kick amplitude. Chaos parameter
    sigma : float, optional
        noise. The default is 0.2.
    s : integer, optional
        length of p interval. The default is 1.
        0<=p<=2.pi.s

    Returns
    -------
    res : float
        Perron-Frobenius matrix element

    '''
    
    # assert m < 1/sigma, 'effective truncation error m'
    # assert np.abs(m-mp) < kp*K/s, 'effective truncation error |m-mp|'
    # assert np.abs(k-kp) < s/sigma, 'effective truncation error |k-kp|'
    k2 = k
    m2 = m
    k1 = kp
    m1 = mp    
    if k2-k1 == m2*s:# and m < 1/sigma and  np.abs(m-mp) < kp*K/s and np.abs(k-kp) < s/sigma:
        res = jv(m2-m1,k1*K/s)*np.exp(-sigma**2/2*m2**2)
    else:
        res = 0
    if res == np.infty:
        print(k,m,kp,mp)
    return res

# @jit
def n_from_qp(qf,pf,paso,shift=0):
    '''
    Take (q,p) and return number of cell

    Parameters
    ----------
    qf : float 
        position
    pf : float
        momentum
    paso : float
        width of cell. Tipically paso=1/N


    Returns
    -------
    nqi : integer
        index of q cell
    npi : integer
        index of p cell

    '''
    # print(qf+shift)
    # print((qf+shift)/dpi)
    # print((qf+shift)/dpi/paso)
    nqi = (qf+shift)#/paso#/dpi/paso
    npi = (pf+shift)#/paso#/dpi/paso
    
    return int(nqi), int(npi)

def basis_U(N, shift=0):
    '''
    Create k,m,kp,mp arrays to calculate Perron-Frobenius element.

    Parameters
    ----------
    N : integer
        number of k,m,kp,mp values
    shift : integer, optional
        shift for values of k,m,kp,mp. These values range from 0 to N by default. The default is 0.

    Returns
    -------
    Objetive: compute Perron-Frobenius element:
    
    <k,m|U|kp,mp>
    
    ks : array_like
        values of k (Fourier of p). Left vector.
    ms : array_like
        values of m (Fourier of q). Left vector.
    kps : array_like
        values of kp (Fourier of p). Right vector.
    mps : array_like
        values of mp (Fourier of p). Right vector.

    '''
    # T = dpi/N # para usar dominio de Fourier
    # ks = np.fft.fftfreq(N, d=T)
    ks = np.arange(N)-shift
    ms = deepcopy(ks); kps = deepcopy(ks); mps = deepcopy(ks)
    return ks,ms,kps,mps

def matrix_U_Fourier(N,K,sigma,*args,**kwargs):
    '''
    Compute Perron-Frobenius approximation in Fourier basis

    Parameters
    ----------
    N : integer
        number of values for each variable
    K : float
        kick amplitude. Chaos parameter
    sigma : float, optional
        noise. The default is 0.2.
    s : integer, optional
        length of p interval. The default is 1.
        0<=p<=2.pi.s

    Returns
    -------
    U : 2D array_like
        Perron-Frobenius approximation. N^2 dimension

    '''      
    ks,ms,kps,mps = basis_U(N+1,shift=N//2)
    
    shift = N//2
    
    Neff = len(ms)
    
    U = np.zeros((Neff**2,Neff**2),dtype=np.complex_)
    
    for m in tqdm(ms):
        for k in ks:
            for mp in mps:
                for kp in kps:
                    nmi, nki = n_from_qp(m, k, 1/N, shift=shift)
                    i = int(nmi+nki*N)
                    
                    nmpi, nkpi = n_from_qp(mp, kp, 1/N, shift=shift)
                    j = int(nmpi+nkpi*N)
                    # print(i,k,m,j,kp,mp)
                    
                    U[i,j] = element_U_Fourier(k, m, kp, mp, K, sigma=sigma)
    return U
#%% #%% some plot parameters
import matplotlib as mpl
plt.rcParams['text.usetex'] = True

font_size=20
letter_size=22
label_size=25
title_font=28
legend_size=23

from matplotlib import rc
rc('font', family='serif', size=font_size)
rc('text', usetex=True)



mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.labelsize'] = label_size
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['xtick.major.size']=6
mpl.rcParams['xtick.minor.size']=3
mpl.rcParams['xtick.major.width']=1.4
mpl.rcParams['xtick.minor.width']=0.9
mpl.rcParams['xtick.direction']='in'

mpl.rcParams['ytick.minor.visible']=True
mpl.rcParams['ytick.major.size']=6
mpl.rcParams['ytick.minor.size']=3
mpl.rcParams['ytick.major.width']=2.1
mpl.rcParams['ytick.minor.width']=1.3
mpl.rcParams['ytick.direction']='in'

mpl.rcParams['ytick.direction']='in'



mpl.rcParams['legend.fontsize']=legend_size



import matplotlib.ticker
class MyLocator(matplotlib.ticker.AutoMinorLocator):
    def __init__(self, n=2):
        super().__init__(n=n)
matplotlib.ticker.AutoMinorLocator = MyLocator        


marker_sz = 10
location='upper left'
properties={'size':12}
width_plot=8



def get_axis_limits(ax, scalex=.1, scaley=.85):
    return (ax.get_xlim()[0] + (ax.get_xlim()[1]-ax.get_xlim()[0])*scalex, ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])*scaley)

colorlist=[plt.cm.brg(i) for i in np.linspace(0, 1, 6)]
#%% try it. Create Perron-Frobenius matrix
sigma = 0.2
N = 20
K = 3
U = matrix_U_Fourier(N, K, sigma=sigma)
# Diagonalize it
t0=time()
e, evec = np.linalg.eig(U)
# e, evec = eigs(U, k=4)
eabs = np.abs(e)
evec=evec[:,eabs.argsort()[::-1]]
e = e[eabs.argsort()][::-1]
t1=time()
print(f'Diagonalization: {t1-t0} seg')
print(f'K={K}',f'|e|={np.abs(e)[1]}')
#%% plots eigenvalues
thetas=np.linspace(0,dpi,200)
r = 1
plt.figure(figsize=(10,10))
plt.plot(r*np.cos(thetas),r*np.sin(thetas), '-b', lw=1)
plt.plot(np.real(e),np.imag(e), '.r', ms=5)
plt.xlabel(r'Re$(\epsilon)$')
plt.ylabel(r'Im$(\epsilon)$')
plt.tight_layout()
#%%
def power_law_by_exp(t, resonancia):
    return np.exp(-resonancia*t)

# number of eigenvalues in real axis
points = 20
i=0
j=0
resonancias = np.zeros(points)
while i<points and j<len(e):    
    if np.real(e[j])==np.abs(e[j]):
        resonancias[i] = np.abs(e[j])
        i+=1
    j+=1

# time
t = np.arange(40)

# O1
fx = 0 
for r in range(len(resonancias)):
    fx += power_law_by_exp(resonancias[r],t)

plt.figure(figsize=(10,10))
plt.title(f'cant de resonancias = {points}. Fishman')
plt.plot(t, np.abs(fx), '.-', label=r'$O_1$')
plt.xlabel(r'$t$')
# plt.xlim(theta_min_for_plot,theta_max_for_plot)
plt.ylabel(r'$\sum_i \exp(-\epsilon_i.t) $')
plt.yscale('log')
plt.xscale('log')
# plt.xlim(0,50)
# plt.ylim(0,1)
# plt.ylim(I_min_for_plot,I_max_for_plot)
plt.tight_layout()
plt.savefig('power_law_by_exponential_Fishman.png', dpi=80)

