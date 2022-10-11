#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 19:40:49 2022

@author: tomasnotenson
"""

import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from time import time
from tqdm import tqdm # customisable progressbar decorator for iterators
from numba import jit
# importing "cmath" for complex number operations
from cmath import phase
from random import random
import seaborn as sns
from scipy.stats import norm, multivariate_normal
from scipy.linalg import eig
import scipy.sparse as ss
from scipy.sparse.linalg import eigs
from sklearn.linear_model import LinearRegression

plt.rcParams.update({
"text.usetex": True,
"font.family": "sans-serif",
"font.sans-serif": ["Helvetica"], "font.size": 16})

dpi = 2*np.pi

@jit
def standard_map(q,p,K):
    '''
    One iteration of standard map.
    (q,p) -standard-map-> (qf,pf)

    Parameters
    ----------
    q : float
        initial position value
    p : float
        initial momentum value
    K : float
        kick amplitude. chaos parameter

    Returns
    -------
    qf : float
        final position value
    pf : float
        final momentum value

    '''
    pf = (p + K/(dpi)*np.sin(dpi*q))%1
    qf = (q + pf)%1
    return qf, pf

@jit
def standard_map_absorption(q,p,K,a=2):
    pf = (p + K*np.sin(q+p/2))*(a*K)-a*K/2
    qf = (q + (p+pf)/2)%dpi
    
    # assert (-a*K/2<=pf and pf<=a*K/2), 'no está en el intervalo permitido [-aK/2,aK/2]'
    
    return qf, pf

@jit
def standard_map_dissipation(q,p,K,eta=0.3):
    pf = (p + K*np.sin(q))%(8*np.pi)-4*np.pi
    qf = (q + pf)%(dpi)
    
    assert (-4*np.pi<=pf and pf<=4*np.pi), 'no está en el intervalo [-4pi,4pi]'
    
    return qf, pf

@jit
def Harper_map(q,p,K):
    pf = (p-K*np.sin(dpi*q))%1
    qf = (q+K*np.sin(dpi*pf))%1
    return qf, pf    

@jit
# probar con K = 0.01, 0.02
def perturbed_cat_map(q,p,K):
    pf = (p + q - dpi*K*np.sin(dpi*q))%1
    qf = (q + pf + dpi*K*np.sin(dpi*pf))%1
    return qf, pf

@jit
def CI(qj,pj,paso):
    '''
    Take random (q,p) values in cell of width "paso"

    Parameters
    ----------
    qj : float
        minimum value of position in cell
    pj : float
        minimum value of momentum in cell
    paso : float
        width of cell. In gral, paso=1/N

    Returns
    -------
    q : float
        random position in cell
    p : float
        random momentum in cell

    '''
    rq = np.random.uniform()*paso
    rp = np.random.uniform()*paso
    
    q = qj+rq
    p = pj+rp
    
    # print('CI', q,p)
    return q,p

@jit
def cae_en_celda(qf,pf,qi,pi,paso):
    '''
    Check if (qf,pf) is in cell or not

    Parameters
    ----------
    qf : float
        position to check
    pf : float
        momentum to check
    qi : float
        mininum position of cell
    pi : float
        mininum momentum of cell
    paso : float
        width of cell. In gral, paso=1/N

    Returns
    -------
    cond : bool
        True if (qf,pf) is in cell. False if not

    '''
    
    cond1 = (0<(qf-qi) and (qf-qi)<paso)
    cond2 = (0<(pf-pi) and (pf-pi)<paso)
    
    cond = (cond1 and cond2)
    # if cond:
    #     print('cond1', qf, pf, 'dif', (qf-qi), (pf-pi))
    return cond

@jit
def CI_and_evolution(qj,pj,paso,K,mapa='normal'):
    '''
    Take a random initial condition in cell and iterate the map ones

    Parameters
    ----------
    qj : float
        mininum position of cell
    pj : float
        mininum momentum of cell
    paso : float
        width of cell. In gral, paso=1/N
    K : float
        kick amplitude. chaos parameter
    mapa : string, optional
        name of map of interest. The default is 'normal'.


    Returns
    -------
    qf : float
        final position 
    pf : float
        final momentum
    nqi : integer
        index of q cell
    npi : integer
        index of p cell

    '''
    
    # tomo una CI dentro de la celda j
    q, p = CI(qj, pj, paso)
    # q, p = gaussian_kernel(q, p, paso, ruido)
    
    # evoluciono un paso 
    qf, pf, nqi, npi = evolution(q, p, K, mapa)
    
    return qf, pf, nqi, npi

@jit
def gaussian_kernel(q,p,paso,ruido,modulo=1):
    # rv = norm(0, ruido*paso)
    qf = (q + np.random.normal(0,ruido))%modulo
    pf = (p + np.random.normal(0,ruido))%modulo
    
    return qf, pf

@jit
def evolution(q,p,K,mapa='normal'):
    '''
    Iterate the map ones

    Parameters
    ----------
    q : float
        initial position value
    p : float
        initial momentum value
    K : float
        kick amplitude. chaos parameter
    mapa : string, optional
        name of map of interest. The default is 'normal'.

    Returns
    -------
    qf : float
        final position
    pf : float
        final momentum
    nqi : integer
        index of q cell
    npi : integer
        index of p cell

    '''
    # evoluciono un paso 
    if mapa=='Harper':
        qf, pf = Harper_map(q,p,K)
        
        nqi = qf//paso
        npi = pf//paso
        
    if mapa=='cat':
        qf, pf = perturbed_cat_map(q,p,K)
        
        nqi = qf//paso
        npi = pf//paso
        
    if mapa=='normal':
        qf, pf = standard_map(q,p,K)
        
        nqi = qf//paso
        npi = pf//paso
        
    elif mapa=='absortion':
        a=2
        qf, pf = standard_map_absorption(q,p,K,a)
        
        if not(-a*K/2<=pf and pf<=a*K/2):
            q, p = CI(q, p, paso)
            qf, pf = evolution(q,p,K,mapa='absortion')
        
        nqi = qf/dpi//paso
        npi = (pf+a*K/2)/(a*K)//paso
                
    elif mapa=='dissipation':
        eta=0.3
        qf, pf = standard_map_dissipation(q,p,K,eta)
        
        nqi = qf/dpi//paso
        npi = (pf+4*np.pi)/(8*np.pi)//paso
        
    return qf, pf, nqi, npi

@jit
def n_from_qp(qf,pf,paso,mapa='normal'):
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
    mapa : string, optional
        Map of interest. The default is 'normal'.

    Returns
    -------
    nqi : integer
        index of q cell
    npi : integer
        index of p cell

    '''
    
    if (mapa=='normal' or mapa=='cat' or mapa=='Harper'):
                
        # print(qf, pf, paso)
        nqi = qf//paso
        npi = pf//paso
        
    elif mapa=='absortion':
        a=2
        
        nqi = qf/dpi//paso
        npi = (pf+a*K/2)/(a*K)//paso
                
    elif mapa=='dissipation':
        
        nqi = qf/dpi//paso
        npi = (pf+4*np.pi)/(8*np.pi)//paso
        
    return nqi, npi

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

@jit
def Ulam_one_trayectory(N,Nx,paso,Nc,K,mapa,symmetry=True):
    Nmitad = N#//2
    
    # S = np.zeros((Nmitad,Nmitad))
    S = np.zeros((Nmitad, Nmitad), dtype=np.float64)
    # inicializo en una condición inicial (cercana al atractor)
    # celda j (en lo posible cerca del atractor)
    qj = 0.1/(2*np.pi); pj = 0.1/(2*np.pi)
    # tomo una CI random en la celda j
    q, p = CI(qj,pj,paso)

    qf, pf = q, p
    nqi, npi = n_from_qp(qf, pf, paso, mapa)
    # evoluciono hasta t=100
    # for nj in range(100):
    #     # evoluciono un tiempo
    #     qf, pf, nqi, npi = evolution(qf, pf, K, mapa)   
    normaj = np.zeros((Nmitad))
    # cont=0
    for nj in tqdm(range(Nc), desc='loop evolution'):
        
        # número de celda
        j = int(nqi+npi*Nx)
        
        assert j < Nx**2, f'number of cell j={j} out of bounds. Con nqi={nqi} y npi={npi}.'
        
        # evoluciono un tiempo
        qf, pf, nqi, npi = evolution(qf,pf,K,mapa)
        qf, pf = gaussian_kernel(qf, pf, paso,ruido,modulo=1)
        
        nqi, npi = n_from_qp(qf, pf, paso, mapa)
        
        # número de celda
        i = int(nqi+npi*Nx)
        
        assert i < Nx**2, f'number of cell i={i} out of bounds. Con nqi={nqi} y npi={npi}.'
        
        
        # if pf>1/2:
        #     cont+=1
        
        # print(qf, pf, nqi, npi, i)
        
        S[i,j] += 1
        normaj[j] += 1
    for j in range(Nmitad):
        if normaj[j] != 0:
            S[:,j] /= normaj[j]
            # print(np.sum(S[:,j]))
        assert  (np.sum(S[:,j]) - 1.0) < 1e-3, f'ojo con sum_i Sij = {np.sum(S[:,j])} para j={j}'
    # print(cont/Nc)
    return S

import matplotlib as mpl
# import seaborn as sb
from sklearn.linear_model import LinearRegression #Regresión Lineal con scikit-learn
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

#%% defino parametros variando N

Ns = np.arange(55,63) #np.arange(124,128,2)#np.concatenate((np.arange(20,71,1),np.arange(120,131,2)))#2**np.arange(5,8)
es = [1/2**8]#0.00390625]#1/2**np.arange(1,3,2)*110 # abs
resonancias = np.zeros((len(Ns),len(es)))
        
mapa = 'normal'#'absortion'#'dissipation'#'normal'#'cat'#'Harper'#
method = 'Ulam'#'one_trayectory'#
eta = 0.3
a = 2
cx = 1

K = 6
Nc = int(1e3)

#%% criterio del overlap
ri=0
e_arr = np.zeros((len(Ns)), dtype=np.complex_)
indexes = np.zeros((len(Ns)))
for ni in range(len(Ns)):
    
    Neff = Ns[ni]
    ruido = es[0]
    flag = f'Ulam_approximation_method{method}_mapa{mapa}_Sij_eigenvals_N{Neff}_ruido_abs{ruido}_grilla{cx}N_K{K:.6f}_Nc{Nc}'
    archives = np.load(flag+'.npz')
    e = archives['e']
    evec = archives['evec']
    
    e1 = np.abs(evec[:,1])
    overlaps=np.zeros((evec.shape[1]))
    for i in range(evec.shape[1]):
        ei = np.abs(evec[:,i])
        overlap = np.vdot(e1,ei)**2
        # print(overlap)
        overlaps[i] = overlap
    
    e_arr[ni] = e[np.argmin(overlaps)]
    indexes[ni] = np.argmin(overlaps)
    
    # if ni==5:
    #     plt.figure(figsize=(10,6))
    #     plt.title(f'Criterio del overlap. N={Ns[ni]}, K={K}')
    #     plt.plot(np.arange(evec.shape[1]), overlaps, 'r.')
    #     plt.vlines(np.arange(evec.shape[1]), 0, overlaps, color='red', alpha=0.8)
    #     plt.xticks(np.arange(evec.shape[1]), rotation=45, fontsize=12)
    #     plt.ylabel(r'overlap $|\langle 1 | i \rangle|^2$')
    #     plt.xlabel(r'autoestado $i$')
    #     # plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()
    #     plt.savefig(f'overlaps_evec'+flag+'.png', dpi=80)
#%% convergence: e(overlap) vs N
xN = 1/Ns
y_train = np.abs(e_arr)

# lista_df = []
# for ni in range(len(Ns)):
#     lista_df.append([np.real(e_arr[ni]), np.imag(e_arr[ni]), np.abs(e_arr[ni]), indexes[ni], K,Ns[ni]])#

# df = pd.DataFrame(lista_df,columns=['Real e', 'Imag e', 'Abs e', 'e index', 'K','N'])#

modelo = LinearRegression()
modelo.fit(X = xN.reshape(-1, 1), y = y_train)

print("Intercept:", modelo.intercept_)
# print("Coeficiente:", list(zip(xN.columns, modelo.coef_.flatten(), )))
# print("Coeficiente de determinación R^2:", modelo.score(xN, y_train))

# Podemos predecir usando el modelo
dom = np.linspace(0,max(xN),1000)
y_pred = modelo.predict(dom.reshape(-1,1))


plt.figure(figsize=(10,6))
plt.plot(xN, y_train, 'r.-', label='Ulam')
plt.plot(dom, y_pred, 'b-', lw=2, alpha=0.8, label='linear fit')
plt.hlines(0.752, 0, max(xN), color='black', alpha=0.8, label=r'$\alpha_{O_1}$')

ni=0
for xitem,yitem in np.nditer([xN,y_train]):
        etiqueta = f"{indexes[ni]:.0f}"
        plt.annotate(etiqueta, (xitem,yitem), textcoords="offset points",xytext=(0,10),ha="center", fontsize=8)
        ni+=1

# plt.xticks(np.arange(evec.shape[1]), rotation=45, fontsize=12)
plt.ylabel(r'$|\epsilon|$')
plt.xlabel(r'$N^{-1}$')
# plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()
plt.show()
plt.savefig(f'Ulam_vs_N'+flag+'.png', dpi=80)
#%% import os
import imageio
ni = 3

Neff = Ns[ni]
ruido = es[0]
flag = f'Ulam_approximation_method{method}_mapa{mapa}_Sij_eigenvals_N{Neff}_ruido_abs{ruido}_grilla{cx}N_K{K:.0f}_Nc{Nc}'
# flag = 'Ulam_approximation_methodUlam_mapanormal_Sij_eigenvals_N110_ruido0.00390625_grilla1N_K7_Nc1000'
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
    if i == 36:
        hus = np.abs(eigenvec_j_to_qp(evec[:,i]))#**2
        plt.figure(figsize=(16,8))
        plt.title(f'Standard Map. N={Ns[ni]}, e={es[ri]:.0e}, K={K}, i={i}, eval={np.abs(e[i]):.3f}')
        sns.heatmap(hus)
        plt.tight_layout()
        plt.savefig(f'autoestado_n{i}'+flag+'.png', dpi=80)
        plt.close()
#%% #%% plot eigenstate i=1
for ki in range(len(Ks)):
    K = Ks[ki]
    flag = f'Ulam_approximation_method{method}_mapa{mapa}_Sij_eigenvals_N{Neff}_ruido_abs{ruido}_grilla{cx}N_K{K:.6f}_Nc{Nc}'

    archives = np.load(flag+'.npz')
    e = archives['e']
    evec = archives['evec']
    if K == 12.5:
        i = 38
        hus = np.abs(eigenvec_j_to_qp(evec[:,i]))#**2
        plt.figure(figsize=(16,8))
        plt.title(f'Standard Map. N={N}, e={ruido:.0e}, K={K}, i={i}, eval={np.abs(e[i]):.3f}')
        sns.heatmap(hus)
        plt.tight_layout()
        plt.savefig(f'autoestado_n{i}'+flag+'.png', dpi=80)
        plt.close()
#%% overlap vs K
Ns = [90,128]
# Neff = 128
ruido = 1/2**8#[0.00390625]#1/2**np.arange(1,3,2)*110 # abs
        
mapa = 'normal'#'absortion'#'dissipation'#'normal'#'cat'#'Harper'#
method = 'Ulam'#'one_trayectory'#
eta = 0.3
# a = 2
cx = 1

Kpaso = .25
Ks = np.arange(0,20.1,Kpaso)#0.971635406

evals_K = np.zeros((len(Ks), len(Ns))); indexes = np.zeros((len(Ks), len(Ns)))

overlap_lim = 5e-3
Nc = int(1e3)#int(2.688e7)#int(1e8)#
#
k = 45
for ni in range(len(Ns)):
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')
    for ki in tqdm(range(len(Ks)), desc='loop K'):
        Neff = Ns[ni]
        K = Ks[ki]
        Nx = int(cx*Neff)
        N = Nx**2
        flag = f'Ulam_approximation_method{method}_mapa{mapa}_Sij_eigenvals_N{Neff}_ruido_abs{ruido}_grilla{cx}N_K{K:.6f}_Nc{Nc}'
    
        archives = np.load(flag+'.npz')
        e = archives['e']
        evec = archives['evec']
        
        e1 = np.abs(evec[:,1])
        overlaps=np.zeros((evec.shape[1]))
        for i in range(evec.shape[1]):
            ei = np.abs(evec[:,i])
            overlap = np.vdot(e1,ei)**2
            # print(overlap)
            overlaps[i] = overlap
        
        cont=0
        i=1
        
        while (cont==0 and i<evec.shape[1]):
    
            # print(f'IPR evec{i} = ', IPRs[i])
            if ((overlaps[i]-min(overlaps))<overlap_lim):# and (IPRs[i]<IPR_treshold):
                cont=1
            else:
                i+=1
        print(Ns[ni], Ks[ki], i, np.abs(e[i]))
        indexes[ki,ni] = i
        evals_K[ki,ni] = np.abs(e[i])
        i = 0

#%% corrijo algunos a mano
# cor_ind = [8, 8.5, 11, 11.5, 14, 14.5, 15.5, 16, 16.5, 17, 17.5, 18, 19.5, 20]

# for ind in cor_ind:
#     ki = round(ind/.5)
#     K = Ks[ki] 
    
#     flag = f'Ulam_approximation_method{method}_mapa{mapa}_Sij_eigenvals_N{Neff}_ruido_abs{ruido}_grilla{cx}N_K{K:.6f}_Nc{Nc}'

#     archives = np.load(flag+'.npz')
#     e = archives['e']
        
#     indexes[ki] = 1
#     evals_K[ki] = np.abs(e[1])
#%% cargo Agam

Agam_archivo = np.array([[ 0.2       ,  0.99008465,  0.98628311],
       [ 0.4       ,  0.9898204 ,  0.98239649],
       [ 0.6       ,  0.98940457,  0.97739039],
       [ 0.8       ,  0.98881599,  0.97020105],
       [ 1.        ,  0.98797987,  0.96592535],
       [ 1.2       ,  0.98679681,  0.96214898],
       [ 1.4       ,  0.98515938,  0.95564731],
       [ 1.6       ,  0.98289978,  0.94665441],
       [ 1.8       ,  0.98036816,  0.93450966],
       [ 2.        ,  0.97824085,  0.91856504],
       [ 2.2       ,  0.97570089,  0.90304041],
       [ 2.4       ,  0.96986259,  0.9030563 ],
       [ 2.6       ,  0.95957671,  0.90803738],
       [ 2.8       ,  0.95090726,  0.89610993],
       [ 3.        ,  0.94905812,  0.86315306],
       [ 3.2       ,  0.94923273,  0.84326165],
       [ 3.4       ,  0.94917123,  0.82556552],
       [ 3.6       ,  0.94789772,  0.80101979],
       [ 3.8       ,  0.9448457 ,  0.75713859],
       [ 4.        ,  0.93932105,  0.73652685],
       [ 4.2       ,  0.93144426,  0.70375668],
       [ 4.4       ,  0.92260203,  0.67890895],
       [ 4.6       ,  0.91337322,  0.73849823],
       [ 4.8       ,  0.90189935,  0.80268764],
       [ 5.        ,  0.88527977,  0.83408279], 
       [ 5.2       ,  0.86410547,  0.84218101],
       [ 5.4       ,  0.84089212,  0.83534343],
       [ 5.6       ,  0.8177247 ,  0.81470074],
       [ 5.8       ,  0.79002054,  0.78028657],
       [ 6.        ,  0.75797302,  0.74335136], 
       [ 6.        ,  0.75797302,  0.74335136],
       [ 6.2       ,  0.7323376 ,  0.71146727],
       [ 6.4       ,  0.72970347,  0.72472119],
       [ 6.6       ,  0.79296025,  0.78255186],
       [ 6.8       ,  0.78923485,  0.77275379],
       [ 7.        ,  0.76113182,  0.7350251 ], 
       [ 7.2       ,  0.71855546,  0.68011619],
       [ 7.4       ,  0.66067582,  0.62796231],
       [ 7.6       ,  0.60957403,  0.60317646],
       [ 7.8       ,  0.60104454,  0.58552043],
       [ 8.        ,  0.59118273,  0.5600842 ], 
       [ 8.2       ,  0.58242286,  0.58242286],
       [ 8.4       ,  0.57650897,  0.57650897],
       [ 8.6       ,  0.56337397,  0.56337397],
       [ 8.8       ,  0.5565611 ,  0.55087264],
       [ 9.        ,  0.53760805,  0.53760805],
       [ 9.2       ,  0.63358958,  0.63358958],
       [ 9.4       ,  0.73156675,  0.73156675],
       [ 9.6       ,  0.67619662,  0.67619662],
       [ 9.8       ,  0.59014398,  0.55900365],
       [10.        ,  0.57820559,  0.52885735],
       [10.2       ,  0.56588387,  0.50939805],
       [10.4       ,  0.54596302,  0.47059897],
       [10.6       ,  0.51881527,  0.45568069],
       [10.8       ,  0.51156501,  0.45796041],
       [11.        ,  0.51132091,  0.49758454],
       [11.2       ,  0.51220442,  0.4996588 ],
       [11.4       ,  0.47803335,  0.47803335],
       [11.6       ,  0.47895462,  0.47895462],
       [11.8       ,  0.48059366,  0.47336772],
       [12.        ,  0.61720967,  0.6075464 ],
       [12.2       ,  0.67707983,  0.67364555],
       [12.4       ,  0.67541944,  0.66408497],
       [12.6       ,  0.64221332,  0.6203312 ],
       [12.8       ,  0.65777468,  0.64573523],
       [13.        ,  0.64829548,  0.61764157],
       [13.2       ,  0.61397579,  0.5505278 ],
       [13.4       ,  0.57044458,  0.52129359],
       [13.6       ,  0.53949803,  0.50775636],
       [13.8       ,  0.49370899,  0.47434986],
       [14.        ,  0.45947742,  0.45947742],
       [14.2       ,  0.46296583,  0.46296583],
       [14.4       ,  0.45887062,  0.4388266 ],
       [14.6       ,  0.48520089,  0.426412  ],
       [14.8       ,  0.49827048,  0.40101597],
       [15.        ,  0.48354679,  0.38821192],
       [15.2       ,  0.47140208,  0.42023722],
       [15.4       ,  0.44895293,  0.4442015 ],
       [15.6       ,  0.55769646,  0.55769646],
       [15.8       ,  0.62293468,  0.62293468],
       [16.        ,  0.56056279,  0.56056279],
       [16.2       ,  0.49882391,  0.49882391],
       [16.4       ,  0.46746261,  0.46352421],
       [16.6       ,  0.45748065,  0.41715617],
       [16.8       ,  0.46908386,  0.39327775],
       [17.        ,  0.46646993,  0.39580034],
       [17.2       ,  0.4611999 ,  0.41720077],
       [17.4       ,  0.45224197,  0.41820463],
       [17.6       ,  0.46202086,  0.43950669],
       [17.8       ,  0.44612345,  0.43252012]])

x2 = Agam_archivo[:,0]; y2 = Agam_archivo[:,1]

# load pendientes.txt
# pendientes_archivo = np.array([[0.0, 1.0, 0],
# [0.2, 1.0, 0],
# [0.4, 1.0, 0],
# [0.6, 0.9995777230703548, 0],
# [0.8, 0.9985610771769128, 0],
# [1.0, 0.9980072265946655, 0],
# [1.2, 0.996443149645892, 0],
# [1.4, 0.992269573611514, 0],
# [1.6, 0.985194217113963, 0],
# [1.8, 0.9742642578476657, 0],
# [2.0, 0.9691394281009377, 0],
# [2.2, 0.9625941481124255, 0],
# [2.4, 0.9586829090707926, 0],
# [2.6, 0.9518577401526696, 0],
# [2.8, 0.9485945203344941, 0],
# [3.0, 0.9415941373933686, 0],
# [3.2, 0.9263050121366043, 0],
# [3.4, 0.9268109551399025, 0],
# [3.6, 0.9058882124101635, 0],
# [3.8, 0.9130627238218614, 0.017],
# [4.0, 0.9017773577051491, 0.016],
# [4.2, 0.878, 0.035],
# [4.4, 0.873, 0.042],
# [4.6, 0.870, 0.037],
# [4.8, 0.8556, 0.048 ],
# [5.0, 0.841, 0.039],
# [5.2, 0.807, 0.05],
# [5.4, 0.811, 0.014],
# [5.6, 0.789,0.092],
# [5.8, 0.781, 0.03],
# [6.0, 0.75, 0.049],
# [6.2, 0.708, 0],
# [6.4, 0.699, 0.094],
# [6.6, 0.694, 0.135],
# [6.8, 0.804, 0.115],
# [7.0, 0.698, 0.16],
# [7.2, 0.6890928290888447,0],
# [7.4, 0.6643128179757942,0],
# [7.6, 0.6418740879898527,0],
# [7.8, 0.6640985310735583, 0.094],
# [8.0, 0.6410121866125829,0],
# [8.2, 0.6297957492912091, 0.086],
# [8.4, 0.5966030069369422,0],
# [8.6, 0.6595749831030804,0.125],
# [8.8, 0.6287752362840034,0.074],
# [9.0, 0.666931589529349, 0.105],
# [9.2, 0.6520130466882241,0],
# [9.4, 0.887,0.28],
# [9.6, 0.759, 0.148],
# [9.8, 0.711, 0.071],
# [10.0, 0.6576862082279084,0],
# [10.2, 0.636625155058811,0],
# [10.4, 0.5925197492879645,0],
# [10.6, 0.558746858078737,0],
# [10.8, 0.5682069685406644,0],
# [11.0, 0.5363034051023226,0],
# [11.2, 0.4969124112327961,0],
# [11.4, 0.5506891372519315, 0.098],
# [11.6, 0.473, 0.091],
# [11.8, 0.560973373775162, 0.17],
# [12.0, 0.604,0.117],
# [12.2, 0.646,0.158],
# [12.4, 0.607,0.109],
# [12.6, 0.631,0.131],
# [12.8, 0.653, 0.255],
# [13.0, 0.658,0.305],
# [13.2, 0.583, 0.173],
# [13.4, 0.577,0.191],
# [13.6, 0.554, 0.1],
# [13.8, 0.501110797563448, 0.024],
# [14.0, 0.480416522565377,0],
# [14.2, 0.474940551339353,0],
# [14.4, 0.449287200924191,0],
# [14.6, 0.439860745967237,0],
# [14.8, 0.478366345485303,0],
# [15.0, 0.482483598240196,0],
# [15.2, 0.489418842123907,0],
# [15.4, 0.512312215780331,0],
# [15.6, 0.554513512285739,0],
# [15.8, 0.520285216011930,0],
# [16.0, 0.501508302913157,0],
# [16.2, 0.511737997587165,0],
# [16.4, 0.551339892636643,0],
# [16.6, 0.478744966247957,0],
# [16.8, 0.477898760451666,0],
# [17.0, 0.475452173381912,0],
# [17.2, 0.405875290874922,0],
# [17.4, 0.417952080926956,0],
# [17.6, 0.415942514762718,0],
# [17.8, 0.461708811268827,0],
# [18.0, 0.454765286457298,0],
# [18.2, 0.488488389034261,0],
# [18.4, 0.497152829855067,0],
# [18.6, 0.514101995099865,0],
# [18.8, 0.544624642083803,0],
# [19.0, 0.465426230140068,0],
# [19.2, 0.489653770249319,0],
# [19.4, 0.506236512035130,0],
# [19.6, 0.479928184183485,0],
# [19.8, 0.439765301492974,0]])
# x = pendientes_archivo[:,0]; y = pendientes_archivo[:,1]
pendientes_archivo = np.genfromtxt('pendientes_RAFA.txt', delimiter=',')
x = pendientes_archivo[:,0]; y = pendientes_archivo[:,1]
#%% plot 
# pendientes_archivo = np.genfromtxt('pendientes.txt',delimiter=',',usecols=(0,1))
# x = pendientes_archivo[:,0]; y = pendientes_archivo[:,1]
plt.figure(figsize=(12,8))
plt.plot(x,y, '.-', label=r'OTOC $\alpha_{O_1}$ N=5000')
plt.plot(x2,y2, '.-', label=r'Agam $|\epsilon_1|$'+f' N=90')
plt.plot(Ks,evals_K[:,0], '.-', label=r'Ulam $|\epsilon_{overlap}|$'+f' N={Ns[0]}')
# plt.plot(Ks,evals_K[:,1], '.-', label=r'$|\epsilon_{overlap}|$'+f'N={Ns[1]}')
plt.xlabel(r'$K$')
plt.ylabel(r'$|\epsilon|$')#', $\alpha_{O_1}$')
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(f'Agam_y_overlap_vs_K_N{Ns[0]}_ruido{ruido}.png', dpi=80)
