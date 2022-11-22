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

font_size=30
letter_size=32
label_size=35
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

def normalize(x):
    return (x-min(x))/(max(x)-min(x))

@jit
def qp_from_j(j,Nx,paso,mapa='normal'):
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
    N = len(eigenvector)
    Nx = int(np.sqrt(N))
    paso = 1
    eig_res = np.zeros((Nx,Nx), dtype=np.complex_)
    for j in range(N):
        q,p = qp_from_j(j, Nx, paso, mapa)
        # print(int(q),int(p))
        eig_res[int(q),int(p)] = eigenvector[j]
    return eig_res

def plot_eigenstates(hus, i, *args, **kwargs):
    
    data = hus 
    
    cmap = mpl.cm.get_cmap('viridis', 12)
    fig, ax = plt.subplots(figsize=(12,10))
    heatmap = ax.pcolor(data, cmap=cmap)
    
    #legend
    cbar = plt.colorbar(heatmap)
    # cbar.ax.set_yticklabels(['0','1','2','>3'])
    # cbar.set_label(r'$|\psi|^2$', rotation=0)#, fontsize=10)
    cbar.ax.set_title(r'$|\psi|^2$', rotation=0)
    
    qs = np.arange(0,Neff+1)#/Neff
    
    paso = int(Neff/10)
    
    # put the major ticks at the middle of each cell
    ax.set_xticks(qs[::paso], minor=False)
    ax.set_yticks(qs[::paso], minor=False)
    # ax.invert_yaxis()
    
    #labels
    labels = [f'{qs[index]/Neff:.1f}' for index in range(0,len(qs)+1,paso)]
    # column_labels = qs#list('ABCD')
    # row_labels = qs#list('WXYZ')
    ax.set_xticklabels(labels, minor=False)
    ax.set_yticklabels(labels, minor=False)
    ax.set_xlabel(r'$q$')
    ax.set_ylabel(r'$p$', rotation=0)
    # plt.subplots_adjust(hspace=1)

    plt.tight_layout()
    plt.show()
    plt.savefig(f'autoestado_n{i}_N{Neff}'+flag+'.png', dpi=80)

#%% 
Ns = [128]#np.arange(81,100,2) #np.arange(124,128,2)#np.concatenate((np.arange(20,71,1),np.arange(120,131,2)))#2**np.arange(5,8)
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
for i in range(evec.shape[1]):#ies:#
    # i = 1
    if i == 1:
        hus = np.abs(eigenvec_j_to_qp(evec[:,i]))#**2
        # plt.figure(figsize=(12,10))
        # plt.title(f'Standard Map. N={Ns[ni]}, e={es[ri]:.0e}, K={K}, i={i}, eval={np.abs(e[i]):.3f}')
        plot_eigenstates(hus,i)


