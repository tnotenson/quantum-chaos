#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 19:48:46 2022

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

delta=8

font_size=20+delta
letter_size=22+delta
label_size=25+delta
title_font=28+delta
legend_size=23+delta

from matplotlib import rc
rc('font', family='serif', size=font_size)
rc('text', usetex=True)


delta2=3

mpl.rcParams['lines.linewidth'] = 2+0.25*delta2
mpl.rcParams['axes.labelsize'] = label_size
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['xtick.major.size']=6+0.1*delta2
mpl.rcParams['xtick.minor.size']=3+0.1*delta2
mpl.rcParams['xtick.major.width']=1.4+0.5*delta2
mpl.rcParams['xtick.minor.width']=0.9+0.5*delta2
mpl.rcParams['xtick.direction']='in'

mpl.rcParams['ytick.minor.visible']=True
mpl.rcParams['ytick.major.size']=6+0.1*delta2
mpl.rcParams['ytick.minor.size']=3+0.1*delta2
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

#%% try it. Create Perron-Frobenius matrix
Ns = [30]#np.arange(14,31,2)
# Kpaso = 0.1
# Ks = np.arange(0,20,Kpaso)

# sigmas = [0]#np.linspace(0,1,100)

# for si in tqdm(range(len(sigmas)), desc='sigmas loop'):
#     sigma = sigmas[si]


#     for ni in range(len(Ns)):
#         N = Ns[ni]
#         es = np.zeros((len(Ks)), dtype=np.complex_)
        
#         for ki in range(len(Ks)):
#             K = Ks[ki]
#             U = matrix_U_Fourier(N, K, sigma=sigma)
#             # Diagonalize it
#             t0=time()
#             e, evec = np.linalg.eig(U)
#             # e, evec = eigs(U, k=4)
#             eabs = np.abs(e)
#             evec=evec[:,eabs.argsort()[::-1]]
#             e = e[eabs.argsort()][::-1]
#             t1=time()
#             print(f'Diagonalization: {t1-t0} seg')
#             print(f'K={K}',f'|e|={np.abs(e)[1]}')
#             es[ki] = e[1]
#         filename = f'Fishman_N{N}_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_sigma{sigma}'
#         np.savez(filename+'.npz', Ks=Ks, es=es)

# es_sin_ruido = es

#%%

# sigma=0#0202020202020204

# N = 30
Kpaso = 0.1
Ks = np.arange(0,20,Kpaso)
N = Ns[0]
sigma = 0.2#sigmas[0]
filename = f'Fishman_N{N}_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_sigma{sigma}'

archives = np.load(filename+'.npz')
Ks = archives['Ks']
es = archives['es']

es_Fishman = es

pendientes_archivo = np.genfromtxt('pendientes_exponential2.txt', delimiter=',')
Kx = pendientes_archivo[:,0]; pendientes = pendientes_archivo[:,1]; yerr = pendientes_archivo[:,2];
linf = pendientes_archivo[:,3]; lsup = pendientes_archivo[:,4];
# pend_normed = (pendientes - min(pendientes))/(max(pendientes) - min(pendientes))

tamanios=lsup-linf

for i,tam in enumerate(tamanios):
    if tam < 3:
        if i<6:
            pendientes[i] = 1
            yerr[i] = 0
        else:
            pendientes[i] = (pendientes[i+1]+pendientes[i-1])/2
            yerr[i] =  np.sqrt(yerr[i-1]**2 + yerr[i+1]**2)
# cuando el error da infinito sólo usé 2 ptos para ajustarla exponencial
# con lo cual reemplazo ese punto promediando los vecinos

# threep_arg = []
# for k in range(len(lsup)):
#     tam = lsup[k]-linf[k]
#     if tam == 4:
#         threep_arg.append(k)
        
# err_arg = np.where(yerr==np.infty)
# for i in err_arg[0]:
#     if i==0:
#         value = 1
#         error = yerr[1]
#     elif i==23:
#         value = y[22]
#         error = yerr[22]
#     elif i==24:
#         value = y[25]
#         error = yerr[25]
#     else:    
#         value = (y[i-1]+y[i+1])/2
#         error = np.sqrt( yerr[i-1]**2 + yerr[i+1]**2 )
#     y[i] = value
#     yerr[i] = error

# es_sin_ruido = np.abs(es)
#%% plot es vs Ks

# Agam_archivo = np.array([[ 0.2       ,  0.99008465,  0.98628311],
#         [ 0.4       ,  0.9898204 ,  0.98239649],
#         [ 0.6       ,  0.98940457,  0.97739039],
#         [ 0.8       ,  0.98881599,  0.97020105],
#         [ 1.        ,  0.98797987,  0.96592535],
#         [ 1.2       ,  0.98679681,  0.96214898],
#         [ 1.4       ,  0.98515938,  0.95564731],
#         [ 1.6       ,  0.98289978,  0.94665441],
#         [ 1.8       ,  0.98036816,  0.93450966],
#         [ 2.        ,  0.97824085,  0.91856504],
#         [ 2.2       ,  0.97570089,  0.90304041],
#         [ 2.4       ,  0.96986259,  0.9030563 ],
#         [ 2.6       ,  0.95957671,  0.90803738],
#         [ 2.8       ,  0.95090726,  0.89610993],
#         [ 3.        ,  0.94905812,  0.86315306],
#         [ 3.2       ,  0.94923273,  0.84326165],
#         [ 3.4       ,  0.94917123,  0.82556552],
#         [ 3.6       ,  0.94789772,  0.80101979],
#         [ 3.8       ,  0.9448457 ,  0.75713859],
#         [ 4.        ,  0.93932105,  0.73652685],
#         [ 4.2       ,  0.93144426,  0.70375668],
#         [ 4.4       ,  0.92260203,  0.67890895],
#         [ 4.6       ,  0.91337322,  0.73849823],
#         [ 4.8       ,  0.90189935,  0.80268764],
#         [ 5.        ,  0.88527977,  0.83408279], 
#         [ 5.2       ,  0.86410547,  0.84218101],
#         [ 5.4       ,  0.84089212,  0.83534343],
#         [ 5.6       ,  0.8177247 ,  0.81470074],
#         [ 5.8       ,  0.79002054,  0.78028657],
#         [ 6.        ,  0.75797302,  0.74335136], 
#         [ 6.        ,  0.75797302,  0.74335136],
#         [ 6.2       ,  0.7323376 ,  0.71146727],
#         [ 6.4       ,  0.72970347,  0.72472119],
#         [ 6.6       ,  0.79296025,  0.78255186],
#         [ 6.8       ,  0.78923485,  0.77275379],
#         [ 7.        ,  0.76113182,  0.7350251 ], 
#         [ 7.2       ,  0.71855546,  0.68011619],
#         [ 7.4       ,  0.66067582,  0.62796231],
#         [ 7.6       ,  0.60957403,  0.60317646],
#         [ 7.8       ,  0.60104454,  0.58552043],
#         [ 8.        ,  0.59118273,  0.5600842 ], 
#         [ 8.2       ,  0.58242286,  0.58242286],
#         [ 8.4       ,  0.57650897,  0.57650897],
#         [ 8.6       ,  0.56337397,  0.56337397],
#         [ 8.8       ,  0.5565611 ,  0.55087264],
#         [ 9.        ,  0.53760805,  0.53760805],
#         [ 9.2       ,  0.63358958,  0.63358958],
#         [ 9.4       ,  0.73156675,  0.73156675],
#         [ 9.6       ,  0.67619662,  0.67619662],
#         [ 9.8       ,  0.59014398,  0.55900365],
#         [10.        ,  0.57820559,  0.52885735],
#         [10.2       ,  0.56588387,  0.50939805],
#         [10.4       ,  0.54596302,  0.47059897],
#         [10.6       ,  0.51881527,  0.45568069],
#         [10.8       ,  0.51156501,  0.45796041],
#         [11.        ,  0.51132091,  0.49758454],
#         [11.2       ,  0.51220442,  0.4996588 ],
#         [11.4       ,  0.47803335,  0.47803335],
#         [11.6       ,  0.47895462,  0.47895462],
#         [11.8       ,  0.48059366,  0.47336772],
#         [12.        ,  0.61720967,  0.6075464 ],
#         [12.2       ,  0.67707983,  0.67364555],
#         [12.4       ,  0.67541944,  0.66408497],
#         [12.6       ,  0.64221332,  0.6203312 ],
#         [12.8       ,  0.65777468,  0.64573523],
#         [13.        ,  0.64829548,  0.61764157],
#         [13.2       ,  0.61397579,  0.5505278 ],
#         [13.4       ,  0.57044458,  0.52129359],
#         [13.6       ,  0.53949803,  0.50775636],
#         [13.8       ,  0.49370899,  0.47434986],
#         [14.        ,  0.45947742,  0.45947742],
#         [14.2       ,  0.46296583,  0.46296583],
#         [14.4       ,  0.45887062,  0.4388266 ],
#         [14.6       ,  0.48520089,  0.426412  ],
#         [14.8       ,  0.49827048,  0.40101597],
#         [15.        ,  0.48354679,  0.38821192],
#         [15.2       ,  0.47140208,  0.42023722],
#         [15.4       ,  0.44895293,  0.4442015 ],
#         [15.6       ,  0.55769646,  0.55769646],
#         [15.8       ,  0.62293468,  0.62293468],
#         [16.        ,  0.56056279,  0.56056279],
#         [16.2       ,  0.49882391,  0.49882391],
#         [16.4       ,  0.46746261,  0.46352421],
#         [16.6       ,  0.45748065,  0.41715617],
#         [16.8       ,  0.46908386,  0.39327775],
#         [17.        ,  0.46646993,  0.39580034],
#         [17.2       ,  0.4611999 ,  0.41720077],
#         [17.4       ,  0.45224197,  0.41820463],
#         [17.6       ,  0.46202086,  0.43950669],
#         [17.8       ,  0.44612345,  0.43252012]])

# y2 = Agam_archivo[:,1]
# x2 = Agam_archivo[:,0]
# y2 = np.zeros(len(Ks))


# for i,K in enumerate(Ks):
    
#     archives = np.load(f'Blum_Agam_evals_K{K:.1f}_Nsfijo_N90_smin0.0010_smax0.0010.npz')
#     e1 = archives['es'][0][1]
#     y2[i] = np.abs(e1)

Ks = np.arange(18,20.1,0.2)

Agam_18_to_20 = np.zeros(len(Ks))

for k in range(len(Ks)):
    Agam_archive = np.load(f'Blum_Agam_evals_K{Ks[k]:.1f}_Nsfijo_N90_smin0.0010_smax0.0010.npz')
    es = Agam_archive['es']
    # ss = Agam_archive['ss']
    Agam_18_to_20[k] = np.abs(es[0][1])
    
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

xfinal = np.concatenate((x2,Ks))

yfinal = np.concatenate((y2,Agam_18_to_20))
    
#%%
Ks = np.arange(0,20,0.1)
# KUlam = 
alpha = 0.75
# plt.style.use('seaborn-whitegrid')

plt.figure(figsize=(12,8))
# plt.title(f'K={13}')
# plt.plot(x[46],y[46], '.', ms=14,color='cyan')
# plt.plot(x[err_arg],normalize(y[err_arg]), '-r', ms=14,color='blue')
plt.plot(Kx,pendientes, 'ok', ms=10, alpha=alpha)
plt.errorbar(Kx,pendientes,yerr, fmt='-k', label=r'$O_1$ decay', alpha=alpha)
# plt.plot(xfinal,yfinal, color='red', label=r'Agam', alpha=alpha)#, ls='dashdot'
# plt.plot(np.arange(21), y_normed, '.-', label='regular area')
# plt.plot(KUlam,evals_K[:-8,0], '.-', label=r'Ulam', alpha=alpha)# $1^{er}$ NL $|\epsilon_{overlap}|$'+f' N={90}'
plt.plot(Ks,np.abs(es_Fishman), color='grey', label=f'Fishman', alpha=alpha)#, ls='dashed'
plt.plot(Ks,np.abs(es_sin_ruido), color='blue', ls='dashed', label=f'Fishman sin ruido', alpha=alpha)
# plt.vlines(9.31, 0.3,1.2, ls='dashed', color='grey', alpha=0.9)
# plt.vlines(11.4, 0.3,1.2, ls='dashed', color='grey', alpha=0.9)
plt.xticks(list(Ks[::20])+[20])
plt.yticks(np.arange(0.4,1.1,0.2))
plt.ylim(0.38,1.08)
plt.xlabel(r'$K$')
plt.ylabel(r'$|\lambda|$')
# plt.legend(loc='best',fancybox=True, framealpha=1, shadow=True, borderpad=.35)
plt.tight_layout()
plt.savefig(f'pendientes_Fishman_vs_K'+filename+'.pdf',dpi=100)
#%% Comparación archivos pendientes
# pendientes_archivo = np.genfromtxt('pendientes_exponential.txt', delimiter=',')
# x = pendientes_archivo[:,0]; y = pendientes_archivo[:,1]; yerr = pendientes_archivo[:,2];
# linf = pendientes_archivo[:,3]; lsup = pendientes_archivo[:,4];

# err_arg = np.where(yerr==np.infty)
# for i in err_arg[0]:
#     if i==0:
#         value = 1
#         error = yerr[1]
#     elif i==23:
#         value = y[22]
#         error = yerr[22]
#     elif i==24:
#         value = y[25]
#         error = yerr[25]
#     else:    
#         value = (y[i-1]+y[i+1])/2
#         error = np.sqrt( yerr[i-1]**2 + yerr[i+1]**2 )
#     y[i] = value
#     yerr[i] = error
    
# pendientes_archivo = np.genfromtxt('pendientes_RAFA.txt', delimiter=',')
# x2 = pendientes_archivo[:,0]; y2 = pendientes_archivo[:,1];

# plt.figure(figsize=(16,8))
# plt.plot(x[err_arg],y[err_arg], '.', ms=14,color='blue')
# plt.errorbar(x,y,yerr, label=r'nuevo $\alpha_{O_1}$', alpha=alpha)
# plt.plot(x2,y2, label=r'RAFA $\alpha_{O_1}$', alpha=alpha)
# plt.xticks(Ks[::10])
# plt.xlabel(r'$K$')
# plt.ylabel(r'$|\epsilon|$')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.savefig(f'comparacion_pendientes.png',dpi=80)
#%% intervalo de tiempo vs K
# linf = pendientes_archivo[:,3]; lsup = pendientes_archivo[:,4];
# tamaños = lsup - linf 
# plt.figure(figsize=(16,8))
# # plt.plot(x[err_arg],y[err_arg], '.', ms=14,color='blue')
# plt.plot(x,tamaños)
# # plt.plot(x2,y2, label=r'RAFA $\alpha_{O_1}$', alpha=alpha)
# plt.xticks(Ks[::10])
# plt.xlabel(r'$K$')
# plt.ylabel(r'$\Delta t$')
# # plt.legend(loc='best')
# plt.tight_layout()
# plt.savefig(f'intervalos_exponenciales_vs_K.png',dpi=80)

#%% add sigma value to filename
# import os

# for N in [10,18,20]:
#     filename = f'Fishman_N{N}_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}'
    
#     old_name = filename+'_sigma{sigma}'+'.npz'
#     new_name = filename+f'_sigma{sigma}'+'.npz'
    
#     os.rename(old_name, new_name)
#%% plot df form (view npz_to_dat_DataFrame.py)
# df_fild = df[df['e index']==1]

# df_plot = df.pivot('N','sigma','Abs e')
# # sb.lineplot(x="K",
# #                 y="Abs e", hue='N',
# #                 data=df)
# plt.figure(figsize=(10,10))
# ax = sb.heatmap(df_plot,cbar_kws={'label': r'$|\epsilon_1|$'})
# ax.set_ylabel(r'$N$')
# ax.set_yticks([ni for ni in range(0,len(Ns))])
# ax.set_yticklabels([f'{Ns[ni]:.0f}' for ni in range(0,len(Ns))])
# ax.set_xlabel(r'$\sigma$')
# ax.set_xticks([si for si in range(0,len(sigmas),10)])
# ax.set_xticklabels([f'{sigmas[si]:.2f}' for si in range(0,len(sigmas),10)])
# plt.vlines(0.2020202,min(Ns),max(Ns), lw=2, color='black')
# plt.tight_layout()
# plt.savefig(f'Fishman_heatmap_sigma_vs_N_color_e'+filename+'.png',dpi=80)
#%% another plot
# from matplotlib import pylab as pl
# colors = pl.cm.jet(np.linspace(0,1,len(Ns)))

# df_plot = pd.pivot_table(df, values='Abs e', 
#                                 index=['N'], 
#                                 columns=['sigma'], 
#                                 fill_value=0)
# plt.figure(figsize=(12,8))
# for i in range(len(Ns)):
#     plt.plot(sigmas, df_plot.iloc[i,:], color=colors[i], label=r'$N$='+f'{Ns[i]:.0f}', lw=1, alpha=1)#+f'{df_plot[]}'
# plt.title(f'K={Ks[0]}')
# plt.xlabel(r'$\sigma$')
# plt.ylabel(r'$|\epsilon_1|$')
# plt.hlines(0.48041652256537754,min(sigmas),max(sigmas),ls='dashed', color='k',label=r'$1^{er}\,\alpha_{O_1}$')
# # plt.hlines(0.8458086183080821,min(sigmas),max(sigmas),ls='dashdot', color='k', label=r'$2^{do}\,\alpha_{O_1}$')
# # plt.hlines(0.8986145290885205,min(sigmas),max(sigmas),ls='solid', color='k', label=r'$3^{er}\,\alpha_{O_1}$')
# plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
# plt.tight_layout()
# # plt.show()
# plt.savefig(f'Fishman_Nconvergence_vs_sigma_K{K}'+filename+'.png',dpi=80)
# #%% same plot for N sigmas and eig
# df_plot = pd.pivot_table(df, values='Abs e', 
#                                 index=['K'], 
#                                 columns=['N', 'sigma'], 
#                                 fill_value=0)

# # Ks = 

# colors = pl.cm.jet(np.linspace(0,1,4))

# plt.figure(figsize=(12,8))
# for i,col in enumerate([14,22,30]):
#     if not col == 'K':
#         plt.plot(Ks, df_plot[col], color=colors[i], label=f'D={col}', lw=1+0.5*delta2, alpha=.75)

# plt.xlabel(r'$K$')
# plt.xticks(list(Ks[::40])+[20])
# plt.yticks(np.arange(0.4,1.1,0.2))
# plt.ylabel(r'$|\lambda|$')
# leg = plt.legend(loc='best', framealpha=.2)#, shadow=False, borderpad=1, borderaxespad=0.)#,bbox_to_anchor=(1.05,1)
# leg.set_alpha(0.01)
# plt.tight_layout()
# # plt.show()
# plt.savefig(f'Fishman_Nconvergence_vs_K'+filename+'.pdf',dpi=100)