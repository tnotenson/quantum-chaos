#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 19:20:43 2022

@author: tomasnotenson
"""

# import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm # customisable progressbar decorator for iterators
from numba import jit
import seaborn as sb
# import seaborn as sb
# from sklearn.linear_model import LinearRegression #Regresión Lineal con scikit-learn
dpi = 2*np.pi

plt.rcParams['text.usetex'] = True

delta = 4

font_size=20+delta
letter_size=28+delta
label_size=35+delta
title_font=22+delta
legend_size=20+delta

from matplotlib import rc
rc('font', family='serif', size=font_size)
rc('text', usetex=True)

delta2 = 4

mpl.rcParams['lines.linewidth'] = 2+delta2
mpl.rcParams['axes.labelsize'] = label_size
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['xtick.major.size']=6+delta2
mpl.rcParams['xtick.minor.size']=3+delta2
mpl.rcParams['xtick.major.width']=1.4+0.5*delta2
mpl.rcParams['xtick.minor.width']=0.9+0.5*delta2
mpl.rcParams['xtick.direction']='in'

mpl.rcParams['ytick.minor.visible']=True
mpl.rcParams['ytick.major.size']=6+delta2
mpl.rcParams['ytick.minor.size']=3+delta2
mpl.rcParams['ytick.major.width']=2.1+0.5*delta2
mpl.rcParams['ytick.minor.width']=1.3+0.5*delta2
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

@jit
def qp_from_j(j,Nx,paso,mapa='normal'):
    '''
    Take number of cell and return (q,p)

    Parameters
    ----------
    j : integer
        number of cell
    Nx : integer
        sqrt(number of cells). Divide phase space in (Nx)^2 cells
    paso : float
        width of cell
    mapa : string, optional
        Map of interest. The default is 'normal'.

    Returns
    -------
    qj : float
        position 
    pj : float
        momentum

    '''
    if (mapa=='normal' or mapa=='cat' or mapa=='Harper'):
        qj = (j%Nx)*paso
        pj = ((j//Nx)%Nx)*paso
    elif mapa=='absortion':
        a = 2
        qj = (j%Nx)*paso*dpi
        pj = ((j//Nx)%Nx)*paso*(a*K)-a*K/2
    elif mapa=='dissipation':
        # eta = 0.3
        qj = (j%Nx)*paso*dpi
        pj = ((j//Nx)%Nx)*paso*(8*np.pi)-4*np.pi
    return qj, pj

@jit
def eigenvec_j_to_qp(eigenvector, mapa='normal'):
    '''
    Change representation of eigenvalues from cells to (q,p)

    Parameters
    ----------
    eigenvector : array_like
        state in cell representation.
    mapa : string, optional
        Map of interest. The default is 'normal'.

    Returns
    -------
    eig_res : array_like
        state in (q,p) representation.

    '''
    N = len(eigenvector)
    Nx = int(np.sqrt(N))
    paso = 1
    eig_res = np.zeros((Nx,Nx), dtype=np.complex_)
    for j in range(N):
        q,p = qp_from_j(j, Nx, paso, mapa)
        # print(int(q),int(p))
        eig_res[int(q),int(p)] = eigenvector[j]
    return eig_res

def plot_eigenstates(hus1, hus2, i1, i2, *args, **kwargs):
    
    data1 = hus1
    data2 = hus2
    
    cmap = mpl.cm.get_cmap('viridis', 12)
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(22,10),gridspec_kw={'width_ratios': [1, 1.25]})
    heatmap1 = ax1.pcolor(data1, cmap=cmap)
    heatmap2 = ax2.pcolor(data2, cmap=cmap)
    
    
    #legend
    cbar = plt.colorbar(heatmap2)
    # cbar.ax.set_yticklabels(['0','1','2','>3'])
    cbar.set_label(r'$|\psi|^2$', rotation=0)#, fontsize=10)
    # cbar.ax.set_title(r'$|\psi|^2$', rotation=0)
    cbar.set_label(r'$|\psi|^2$', rotation=270, labelpad=+40, y=0.45)
    
    qs = np.arange(0,Neff+1)#/Neff
    
    paso = int(Neff/3)
    
    # put the major ticks at the middle of each cell
    ax1.set_xticks(qs[::paso], minor=False)
    ax2.set_xticks(qs[::paso], minor=False)
    ax1.set_yticks(qs[::paso], minor=False)
    ax2.set_yticks(qs[::paso], minor=False)
    # ax.invert_yaxis()
    
    #labels
    labels = [f'{qs[index]/Neff:.1f}' for index in range(0,len(qs)+1,paso)]
    # column_labels = qs#list('ABCD')
    # row_labels = qs#list('WXYZ')
    ax1.set_xticklabels(labels, minor=False)
    ax2.set_xticklabels(labels, minor=False)
    x_ticks = ax1.xaxis.get_major_ticks()
    x_ticks[0].label1.set_visible(False) ## set first x tick label invisible
    x_ticks = ax2.xaxis.get_major_ticks()
    x_ticks[0].label1.set_visible(False) ## set first x tick label invisible
    ax1.set_yticklabels(labels[:], minor=False)
    ax2.set_yticklabels([])#, minor=False)
    ax1.set_xlabel(r'$q$')
    ax2.set_xlabel(r'$q$')
    ax1.set_ylabel(r'$p$', rotation=90)
    # plt.subplots_adjust(hspace=1)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'autoestado_n{i1}_n{i2}_N{Neff}'+flag+'.pdf', dpi=100)

#%% 
Ns = [90]#np.arange(81,100,2) #np.arange(124,128,2)#np.concatenate((np.arange(20,71,1),np.arange(120,131,2)))#2**np.arange(5,8)
es = [0.00390625]#1/2**np.arange(1,3,2)*110 # abs
resonancias = np.zeros((len(Ns),len(es)))
        
mapa = 'normal'#'absortion'#'dissipation'#'normal'#'cat'#'Harper'#
method = 'Ulam'#'one_trayectory'#
eta = 0.3
a = 2
cx = 1

K = 5#0.971635406

Nc = int(1e3)#int(2.688e7)#int(1e8)#

Neff = Ns[0]
ruido = es[0]
flag = f'Ulam_approximation_method{method}_mapa{mapa}_Sij_eigenvals_N{Neff}_ruido_abs{ruido}_grilla{cx}N_K{K:.6f}_Nc{Nc}'
# flag = 'Ulam_approximation_methodUlam_mapanormal_Sij_eigenvals_N{Neff}_ruido0.00390625_grilla1N_K7_Nc1000'
archives = np.load(flag+'.npz')
e = archives['e']
evec = archives['evec']
# e = np.abs(e)
# evec=evec[:,e.argsort()[::-1]]
# e = np.sort(e)[::-1]
# ies = [1,9]
# guardo los autoestados 
# ni = 0
ri = 0
# for i in range(evec.shape[1]):#ies:#
#     # i = 1
#     if i == 21:
i1 = 1 
i2 = 21
hus1 = np.abs(eigenvec_j_to_qp(evec[:,i1]))#**2
hus2 = np.abs(eigenvec_j_to_qp(evec[:,i2]))
# plt.figure(figsize=(12,10))
# plt.title(f'Standard Map. N={Ns[ni]}, e={es[ri]:.0e}, K={K}, i={i}, eval={np.abs(e[i]):.3f}')
plot_eigenstates(hus1, hus2, i1 ,i2)


