#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 18:51:07 2022

@author: tomasnotenson
"""
# import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm # customisable progressbar decorator for iterators
# import seaborn as sb
# from sklearn.linear_model import LinearRegression #Regresión Lineal con scikit-learn
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

def normalize(x):
    return (x-min(x))/(max(x)-min(x))
def normalize_IPR_O1(x,interval=-10):
    return (x-min(x))/(np.mean(x[interval:])-min(x))
#%% set parameters 

# nqubits = 8;
N = 200#2**nqubits#11
# hbar = 1/(2*np.pi*N)
nr=2
if(N<80): nr=4
elif(N<40): nr=6
elif(N<20): nr=8
elif(N<10): nr=10
elif(N<6): nr=16
elif(N<4): nr=20
  
Nstates= np.int32(nr*N)#N/2)  #% numero de estados coherentes de la base
paso = N/Nstates

Kpaso = .1
Ks = np.arange(0,20.1,Kpaso)#

file = f'IPR_vs_Ks_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_N{N}_coherent_basis_grid{Nstates}x{Nstates}_numpy.npz'
archives = np.load(file)

IPR_means = archives['IPR_means']
rs = archives['rs']
#%%
desde = 1
x = Ks[desde:]
y = IPR_means[desde:]

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

Kpaso1 = .05
K1 = np.arange(0,20,Kpaso1)
archives = np.load(f'r_vs_Ks_Kmin{min(K1)}_Kmax{max(K1):.2f}_Kpaso{Kpaso1}_basis_size{1000}.npz')
r_1 = archives['r_Ks']
x2 = K1; yr2 = r_1

pendientes_archivo = np.genfromtxt('pendientes_RAFA.txt', delimiter=',')
aux = pendientes_archivo[:,1:3]
y3 = np.zeros((len(aux)))
for i in range(len(aux)):
    if i <= round(6/0.2) and aux[i,1] != 0:
        y3[i] = aux[i,1]
    else:
        y3[i] = aux[i,0]
x3 = pendientes_archivo[:,0]; #y3 = pendientes_archivo[:,1]

r_Nacho = np.genfromtxt('r_std_n1000.dat', delimiter='\t')
x4 = r_Nacho[:,0]; y4 = r_Nacho[:,1]

file = 'IPR_O1_from_t_sat500_Kmin0.0_Kmax20.0_Kpaso0.1_time_lim6000_basis_size200AX_BP.npz'
archives = np.load(file)
IPR_O1 = archives['IPRs']
y2 = normalize_IPR_O1(IPR_O1)
Kticks = Ks[::10]

plt.figure(figsize=(12,8))
plt.plot(Ks,y2, '.-', label=r'IPR $O_1$')
# plt.plot(x2[1:],normalize(yr2)[1:], '.-', label=r'$\langle r \rangle$')
plt.plot(x,normalize(y), '.-', label=r'$\langle$IPR$\rangle_{CS}$')
plt.plot(x3,normalize_IPR_O1(1-y3, interval=-34), '.-', label=r'$\alpha_{O_1}$')
plt.plot(x4,normalize_IPR_O1(y4), '.-', label=r'$\langle r \rangle$')
# plt.plot(x2,y2, '.-', label=r'$|\epsilon_1|$ N=90 s=1e-3')
plt.xlabel(r'$K$')
plt.ylabel('IPR (normed)')
plt.xticks(Kticks)
plt.xlim(0,13)
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('IPR_todos_vs_K_Kpaso0.1.png', dpi=80)
#%%
plt.figure(figsize=(12,8))
plt.plot(x3,y3, '.-', label=r'$\alpha_{O_1}$')
# plt.plot(x2,y2, '.-', label=r'$|\epsilon_1|$ N=90 s=1e-3')
plt.xlabel(r'$K$')
plt.ylabel('IPR')
plt.xticks(Kticks)
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()