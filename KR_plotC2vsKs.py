#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 13:32:20 2022

@author: tomasnotenson
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
# from qutip import *
from time import time
from tqdm import tqdm # customisable progressbar decorator for iterators
from sklearn.linear_model import LinearRegression #Regresión Lineal con scikit-learn

plt.rcParams['text.usetex'] = True

delta = 0
font_size=20-delta
letter_size=22-delta
label_size=25-delta
title_font=28-delta
legend_size=23-delta

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

# parameters
N = 5000
Kpaso = 1/5
Ks = np.arange(15,20.1,Kpaso)
# some filename definitions
opA = 'X'
opB = 'P'
operatorss = 'A'+opA+'_B'+opB
time_lim = int(3e1+5) # number of kicks
phi = []#np.identity(N)#-sustate
# modif = ''#'_sust_evec0'
state = '_Tinf_state'
flag = '2pC_FFT_with'+state

# cargo el archivo (con correrlo 1 vez, alcanza)
file = flag+f'_Kmin{min(Ks):.1f}_Kmax{max(Ks):.1f}_Kpaso{Kpaso}_basis_size{N}_time_lim{time_lim}'+operatorss+'.npz'#+'_evolucion_al_reves' _centro{n0}
archives = np.load(file)
C2_Ks = archives['C2_Ks']
Ks = archives['Ks']
#%% Desprolijo, calculo <XP>
from qutip import *

def fV(N):
    V = 0
    for j in range(N):
        V += basis(N,(j+1)%N)*basis(N,j).dag()
    return V

def fU(N, qs):
    U = 0
    tau = np.exp(2j*np.pi/N)
    for j in range(N):
        U += basis(N,j)*basis(N,j).dag()*tau**(qs[j])
    return U


qs = np.arange(0, N) #take qs in [0;N) with qs integer

t0 = time()
# Define Schwinger operators
Us = fU(N, qs)
Vs = fV(N)
    
# # # Define momentum and position operatorsÇ
P = (Vs - Vs.dag())/(2j)
X = (Us - Us.dag())/(2j)
t1 = time()
print(f'Operator creation: {t1-t0} seg')
saturation = X.tr()*P.tr()

#%% ploteo para un K fijo

K = 20

k = round((K-15)/0.2)

times = np.arange(time_lim)

x = times

y = np.abs(C2_Ks[k,:])/N

# create plot
plt.figure(figsize=(16,8))
plt.title(f'N={N}, K={K}')
plt.plot(x, y, '.-', ms=10, lw=1.5,  color='blue', label=f'K={K:.1f}')
plt.hlines(())
plt.ylabel(r'$C_2$')
plt.xlabel(r'$t$')
plt.ylim(-0.0001,y[2]*1.2) 
plt.xlim(2,np.max(x)) 
plt.xticks(x[::2])
# plt.yscale('log')

plt.grid()
plt.legend(loc = 'best')
# plt.show()
# plt.tight_layout()
file = flag+f'_K{K:.1f}_basis_size{N}_time_lim{time_lim}.png'
plt.savefig(file, dpi=80)