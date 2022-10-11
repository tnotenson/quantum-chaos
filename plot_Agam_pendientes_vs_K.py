#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:10:52 2022

@author: tomasnotenson
"""

# import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
from tqdm import tqdm # customisable progressbar decorator for iterators
import pandas as pd
# import seaborn as sb
# from sklearn.linear_model import LinearRegression #Regresión Lineal con scikit-learn
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

def normalize(x):
    return (x-min(x))/(max(x)-min(x))
#%% load Blum Agam simulation 
Kpaso = 1/5
Ks = np.concatenate((np.arange(0,5.4,Kpaso),np.arange(6.4,20,Kpaso)))
# ss = [1/5000]
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

# resonancias = np.zeros((len(Ks)))
# for k,K in enumerate(Ks):
    
#     file = f'Blum_Agam_evals_K{K:.1f}_Nsfijo_N80_smin0.0002_smax0.0002.npz'
#     archives = np.load(file)
#     es = archives['es']    
#     e = es[0,:]
#     resonancia = np.abs(e[1])
#     if resonancia <= 1:
#         resonancias[k] = resonancia
#     else: resonancias[k] = 1

#
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
#
plt.figure(figsize=(12,8))
plt.plot(x,y, '.-', label=r'$\alpha_{O_1}$')
plt.plot(x2,y2, '.-', label=r'$|\epsilon_1|$ N=90 s=1e-3')
plt.xlabel(r'$K$')
plt.ylabel(r'$|\epsilon|$')
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('Agam_vs_K_N90.png', dpi=80)

#%% load and plot Agam data

cte = 90*0.001
Npaso = 5
Ns = np.arange(50,81,Npaso)
K = 13 
ss = cte/Ns##np.arange(1,5)*1e-3

Kpaso = 0
Ks = [13]#np.arange(4,20.1,Kpaso)

N = Ns[0]
nvec = N**2

filename = f'df_Blum_Agam_evals_K{K:.1f}_Nsfijo_Nmin{min(Ns):.0f}_Nmax{max(Ns):.0f}_smin{min(ss):.4f}_smax{max(ss):.4f}' 

df = pd.read_csv(filename+'.dat', header=None, delimiter=r"\s+")
df = df.set_axis(['s','Real e', 'Imag e', 'Abs e', 'e index', 'K', 'N'], axis=1, inplace=False)
print(df.head()) 

df_fild = df[df['e index']==1]
df_fild = df_fild[df_fild['N']==80]
#%% e vs s for different Ns
plt.figure(figsize=(16,8))
sb.lineplot(data=df_fild,x='s', y='Abs e',
                alpha=0.7, hue='N', palette='viridis')#, style='N', markers=True)
plt.ylabel(r'$|\epsilon|$')
plt.hlines(0.609, df_fild['s'].min(), df_fild['s'].max(), color='black', label=r'$\alpha_{O_1}$')

plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0., title = "N")
plt.tight_layout()
plt.savefig(f'Agam_vs_s_forNs_K{K:.1f}_Nsfijo_Nmin{min(Ns):.0f}_Nmax{max(Ns):.0f}_smin{min(ss):.4f}_smax{max(ss):.4f}.png', dpi=80)
#%% e vs N for different ss
plt.figure(figsize=(16,8))
g = sb.lineplot(data=df_fild,x='N', y='Abs e',
                alpha=0.95, hue='s', style='s', markers=True, palette='viridis')
plt.ylabel(r'$|\epsilon|$')
plt.hlines(0.609, df_fild['N'].min(), df_fild['N'].max(), color='black', label=r'$\alpha_{O_1}$')

# # check axes and find which is have legend
# for ax in g.axes.flat:
#     leg = g.axes.flat[0].get_legend()
#     if not leg is None: break
# # or legend may be on a figure
# if leg is None: leg = g._legend

# # change legend texts
# new_title = 's'
# leg.set_title(new_title)
# new_labels = [f'{ss[i]:.2f}' for i in range(len(ss))]
# for t, l in zip(leg.texts, new_labels):
#     t.set_text(l)
norm = plt.Normalize(df_fild['s'].min(), df_fild['s'].max())
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])

g.get_legend().remove()
g.figure.colorbar(sm)#, title='s')
# plt.legend(loc='best', title = "s")
# plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0., title='s')
plt.tight_layout()
plt.savefig(f'Agam_vs_N_forss_K{K:.1f}_Nsfijo_Nmin{min(Ns):.0f}_Nmax{max(Ns):.0f}_smin{min(ss):.4f}_smax{max(ss):.4f}.png', dpi=80)
#%% e in complex plane
plt.figure(figsize=(13,10))


# df_fild.plot.scatter(x='Real e', y='Imag e', c='s', s='N', colormap='YlOrRd', colorbar=True, alpha=0.9)
sb.scatterplot(data=df_fild,x='Real e', y='Imag e',
                alpha=0.7, hue='s', size='N',palette='viridis')#, legend=False)

# plt.ylabel()
# g._legend.remove()

r = 1
thetas = np.linspace(-1,1,1000)*np.pi

plt.plot(r*np.cos(thetas), r*np.sin(thetas), 'b-', lw=1, alpha=0.5)#, label=r'$|z|=1$')
plt.plot(0.609,0,'k*', alpha=0.25, ms=20, label=r'$\alpha_{O_1}$')
# # cbar = plt.colorbar(orientation='vertical')
# # cbar.set_label(label=r'$K$')
plt.xlabel(r'$\Re{(\epsilon)}$')
plt.ylabel(r'$\Im{(\epsilon)}$')
plt.xlim(0.585,0.640)
plt.ylim(-0.01,0.01)
# # plt.xscale('log')
plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
plt.tight_layout()
# plt.grid(True)
plt.savefig(f'Agam_complex_plane_vs_N_forss_K{K:.1f}_Nsfijo_Nmin{min(Ns):.0f}_Nmax{max(Ns):.0f}_smin{min(ss):.4f}_smax{max(ss):.4f}.png', dpi=80)
#%% ploteo con matplotlib basado en fisplot
# import matplotlib as mpl
# fig, [ax, ax_s] = plt.subplots(1, 2, dpi=175, constrained_layout=True)
# fig.set_size_inches(8, 5)

# df_ss = df['s'].values
# df_Re_es = df['Real e'].values
# df_Im_es = df['Imag e'].values
# df_Ks = df['K'].values
# df_Ns = df['N'].values

# cmap = 'viridis'

# norm = mpl.colors.Normalize(df_ss.min(), df_ss.max())  # Un normalizador que asocia valores del intervalo de n al rango [0:1].
# colormap = plt.cm.ScalarMappable(norm, cmap)  # Asocio los valores del rango [0:1] a una secuencia de colores.
# colors = colormap.get_cmap()  # El mapa de color que se utilizará al plotear todo.

# re = df_Re_es#np.real(z_n)
# im = df_Im_es#np.imag(z_n)
# for i in range(len(df_ss)):
#     ax.plot(re[i-1:i+1], im[i-1:i+1], '.-', lw=0.5,
#             ms=7.5*(1-i/df_ss.max()), c=colors(norm(i)))
# ax.set_ylabel(r'$\Im(z)$')
# ax.set_xlabel(r'$\Re(z)$')
# ax.set_aspect('equal', 'datalim')
# ax.grid()