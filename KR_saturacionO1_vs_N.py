#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 17:28:56 2022

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
Ns = [1000,2000,3000,5000,10000]

Kss = []

Kpaso = 4#1/5
Ks = [15,19]#np.arange(0,20.1,Kpaso)
Kss.append(Ks)
Kss.append(Ks)
Kss.append(Ks)
Kpaso = 1/5
Ks = np.arange(0,20.1,Kpaso)
Kss.append(Ks)
Kpaso = 0
Ks = [15]
Kss.append(Ks)
# some filename definitions
opA = 'X'
opB = 'P'
operatorss = 'A'+opA+'_B'+opB
time_lim = int(5e1+1) # number of kicks
# time_lim = int(3e1+5) # number of kicks
phi = []#np.identity(N)#-sustate
modif = ''#'_sust_evec0'
state = '_Tinf_state'
flag = '4pC_FFT_with'+state
#%%
colors = mpl.pyplot.cm.jet(np.linspace(0,1,len(Ns)))


times = np.arange(time_lim)


plt.figure(figsize=(16,8))

i = 0
for N,Ks in zip(Ns,Kss):
    # print(N,Ks)
        
    if len(np.diff(Ks)):
        Kpaso = round(np.mean(np.diff(Ks)),1)
        Kp = 15-min(Ks)
        k = round(Kp/0.2)
    else:
        Kpaso = 0 
        k = 0
    
    print(len(np.diff(Ks)), Kpaso)
    
    file = flag+f'_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_basis_size{N}_time_lim{time_lim}'+operatorss+modif+'.npz'#+'_evolucion_al_reves' _centro{n0}
    archives = np.load(file)
    O1_Ks = archives['O1']
    Ks = archives['Ks']
    
    plt.title(f'N={N}, K={Ks[k]}')
    x = times
    y = np.abs(O1_Ks[k,:])/N*4
    plt.plot(x, y, '.-', ms=10, lw=1.5, color=colors[i], label=f'N={N}')
    plt.hlines(np.mean(y[40:]),min(x),max(x), color=colors[i])
    i+=1
plt.ylabel(r'$O_1$')
plt.xlabel(r'$t$')
# plt.ylim(-0.0001,y[2]*1.2) 
plt.xlim(2,np.max(x)) 
plt.xticks(x[::2])
plt.yscale('log')

plt.grid()
plt.legend(loc = 'best')
# plt.show()
# plt.tight_layout()
file = 'comparacionNs_'+flag+f'_K{Ks[k]:.1f}_basis_size{N}_time_lim{time_lim}.png'
plt.savefig(file, dpi=80)

