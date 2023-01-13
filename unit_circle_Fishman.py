#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 11:48:26 2022

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

def matrix_U_Fourier(N,K,*args,**kwargs):
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

delta=20

font_size=20+delta
letter_size=22+delta
label_size=25+delta
title_font=28+delta
legend_size=23+delta

from matplotlib import rc
rc('font', family='serif', size=font_size)
rc('text', usetex=True)


delta2=4

mpl.rcParams['lines.linewidth'] = 2+0.25*delta2
mpl.rcParams['axes.labelsize'] = label_size
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['xtick.major.size']=6+delta2
mpl.rcParams['xtick.minor.size']=3+delta2
mpl.rcParams['xtick.major.width']=1.4+delta2
mpl.rcParams['xtick.minor.width']=0.9+delta2
mpl.rcParams['xtick.direction']='in'

mpl.rcParams['ytick.minor.visible']=True
mpl.rcParams['ytick.major.size']=6+delta2
mpl.rcParams['ytick.minor.size']=3+delta2
mpl.rcParams['ytick.major.width']=2.1+delta2
mpl.rcParams['ytick.minor.width']=1.3+delta2
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

def normalize(x):
    return (x-min(x))/(max(x)-min(x))
#%% define parameters
N = 30
Ks = [6.6,17]
sigma = 0.2

es = np.zeros((int((N+1)**2), len(Ks)), dtype=np.complex_)

for ki in range(len(Ks)):
    K = Ks[ki]
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
    es[:,ki] = e
filename = f'Fishman_alleigenvalues_N{N}_Ks_{Ks[0]:.2f}_{Ks[1]:.2f}_sigma{sigma}'
np.savez(filename+'.npz', Ks=Ks, es=es)
#%% plot in complex unit circle

r = 1
theta = np.linspace(0,dpi,1000)
x = r*np.cos(theta)
y = r*np.sin(theta)

fig, axs = plt.subplots(1, 2, figsize=(20, 10.5))

for ei in range(es.shape[1]):
    e = es[:,ei]
    axs[ei].plot(np.real(e[:1]), np.imag(e[:1]), 'r*', ms=18, alpha=0.15)
    axs[ei].plot(np.real(e[1:3]), np.imag(e[1:3]), 'r*', ms=30, mfc='none')
    axs[ei].plot(np.real(e[3:]), np.imag(e[3:]), 'r*', ms=18, alpha=0.15)
    axs[ei].plot(x,y,'b-')
    axs[ei].set_xlabel(r'$\Re{(\lambda)}$')
    axs[0].set_ylabel(r'$\Im{(\lambda)}$')
    axs[ei].set_yticks(np.linspace(-1,1,5))
    axs[ei].set_xticks(np.linspace(-1,1,5))
    plt.setp(axs[1].get_yticklabels(), visible=False)
plt.tight_layout()
plt.savefig('unit_circle.pdf',dpi=80)
