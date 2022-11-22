# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:29:46 2013

@author: delande
"""

# Copyright Dominique Delande
# Provided "as is" without any warranty
#
# Computes several "trajectories" of the standard map
# Initial conditions are generated randomly
# Prints the points in the (I,theta) plane (modulo 2*pi)
# Consecutive trajectories are separated by an empty line
# Output is printed in the "several_trajectories.dat" file
# Only points in a restricted area are actually printed. This makes
# it possible to zoom in a limited region of phase space

from math import *
from time import time
import socket
import random
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams['text.usetex'] = True

font_size=20+10
letter_size=22+10
label_size=25+10
title_font=28+10
legend_size=23+10

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
#%%

dpi=2.0*pi
# The following 6 lines for the primary plot
# Execution time is insignificant
K=12.2
# Ks = np.arange(0, 20.1, .2)#np.array([0.5, 3, 6, 8, 10, 12])#[0.25, 0.5, 2, 3, 6, 10]
number_of_trajectories=int(2e4)
number_of_points=800
theta_min_for_plot=0.
theta_max_for_plot=1.#dpi
I_min_for_plot=0.
I_max_for_plot=1.#dpi

# The following 6 lines good for a secondary island
# Execution time is insignificant
# K=1.0
# number_of_trajectories=200
# number_of_points=1000
# theta_min_for_plot=1.7
# theta_max_for_plot=4.6
# I_min_for_plot=2.5
# I_max_for_plot=3.8

# The following 6 lines good for a ternary island
# Execution time: 28seconds
# K=1.0
# number_of_trajectories=1000
# number_of_points=10000
# theta_min_for_plot=4.05
# theta_max_for_plot=4.35
# I_min_for_plot=3.6
# I_max_for_plot=3.75
# N = number_of_points*number_of_trajectories

# data_Ks = np.zeros((N, 2, len(Ks)))

# for k, K in enumerate(Ks):
        
#     thetas, Is = [[], []]
#     Keff = K/(2*np.pi)
#     for i_traj in range(number_of_trajectories):
#       theta=random.uniform(0,1)
#       I=random.uniform(0,1)
#       for i in range(number_of_points):
#         if ((theta>=theta_min_for_plot)&(theta<=theta_max_for_plot)&(I>=I_min_for_plot)&(I<=I_max_for_plot)):
#           # print (theta,I)
#           thetas.append(theta)
#           Is.append(I)
#         I=(I+Keff*sin(dpi*theta))%1
#         theta=(theta+I)%1
#       # print(' ')
#     data_Ks[:,0,k] = thetas
#     data_Ks[:,1,k] = Is
    
#%% 
# i = 0
# for i in range(len(Ks)):
    
#     thetas = data_Ks[:,0,i]
#     Is = data_Ks[:,1,i]
#     plt.figure(figsize=(10,10))
#     plt.title(f'K={Ks[i]:.2f}')
#     plt.plot(thetas, Is, 'b.', ms=0.2)
#     plt.xlabel(r'$q$')
#     plt.xlim(0,1)
#     # plt.xlim(theta_min_for_plot,theta_max_for_plot)
#     plt.ylabel(r'$p$')
#     plt.ylim(0,1)
#     # plt.ylim(I_min_for_plot,I_max_for_plot)
#     plt.tight_layout()
#     plt.savefig(f'phase_space_K{Ks[i]:.2F}.png')
#%% Compute area ratio between regular and chaotic region
from copy import deepcopy

def initial_region(theta,I,theta0,I0,delta=0.001):
    modulo = np.sqrt((theta-theta0)**2+(I-I0)**2)
    if modulo < delta:
        return True
    else: 
        return False

def Metropolis_chaotic_area(K,number_of_trajectories=int(1.5e4), number_of_points=int(1e4)):
    n_tmax=0
    ts = np.zeros(number_of_trajectories); 
    thetas = np.zeros(number_of_trajectories); Is = np.zeros(number_of_trajectories)
    Keff = K/(2*np.pi)
    for i_traj in tqdm(range(number_of_trajectories), desc='loop trajectories'):
        theta=random.uniform(0,1)
        thetas[i_traj] = theta
        I=random.uniform(0,1)
        Is[i_traj] = I
        for i in range(number_of_points):
             # if ((theta>=theta_min_for_plot)&(theta<=theta_max_for_plot)&(I>=I_min_for_plot)&(I<=I_max_for_plot)):
                 # print (theta,I)
                 # thetas.append(theta)
                 # Is.append(I)
             I=(I+Keff*sin(dpi*theta))%1
             theta=(theta+I)%1
             decision = initial_region(theta,I,thetas[i_traj],Is[i_traj])
             if decision:
                 n_tmax+=1 
                 ts[i_traj] = i
                 break
    return n_tmax/number_of_trajectories, ts, np.array([thetas,Is])

K = 3

r_reg, ts, qp = Metropolis_chaotic_area(K)

print(r_reg)
#%%
import matplotlib as mpl

colors = mpl.pyplot.cm.jet(np.linspace(0,1,len(ts)))

thetas = qp[0]
Is = qp[1]
plt.figure(figsize=(10,10))
plt.title(f'K={K:.2f}')
plt.scatter(thetas, Is, c=ts, vmin=0, vmax=number_of_points, s=5)
plt.xlabel(r'$q$')
plt.xlim(0,1)
# plt.xlim(theta_min_for_plot,theta_max_for_plot)
plt.ylabel(r'$p$')
plt.ylim(0,1)
# plt.ylim(I_min_for_plot,I_max_for_plot)
plt.tight_layout()
plt.savefig(f'regular_area_phase_space_K{K:.2f}.png')

#%% r_reg vs K
Kpaso = 1/5
Ks = np.arange(0,20.1,Kpaso)
delta = 0.005
rs = np.zeros(len(Ks))
for k,K in tqdm(enumerate(Ks), desc='K loop'):
    r_reg, _, _ = Metropolis_chaotic_area(K,delta=delta)
    rs[k] = r_reg
file = f'regular_area_vs_K__Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_time_lim{number_of_points}_ntraj{number_of_trajectories}_delta{delta}'
np.savez(file+'.npz', rs=rs, Ks=Ks)
#%%
y = 1-rs**(-1)
y_normed = (y-min(y))/(max(y)-min(y))

plt.figure(figsize=(10,10))
plt.title(r'$r_{reg}$ Regular Area')
plt.plot(Ks, y_normed, '.-')
plt.xlabel(r'$K$')
# plt.xlim(0,1)
# plt.xlim(theta_min_for_plot,theta_max_for_plot)
plt.ylabel(r'$1-r_{reg}^{-1}$')
# plt.ylim(0,1)
# plt.ylim(I_min_for_plot,I_max_for_plot)
plt.tight_layout()
plt.savefig(file+'.png')

#%% make a gif with phase spaces vs K
# import os 
# import imageio

# # Build GIF
# with imageio.get_writer('gif_phase_space.gif', mode='I', fps=1) as writer:
#     for i in range(len(Ks)):
#         filename = f'phase_space_K{Ks[i]:.2F}.png'
#         image = imageio.imread(filename)
#         writer.append_data(image)
#%%
# #%% extra: r chaometer "figura de analisis"
# Ks = np.linspace(0,15,101)

# def f(x,m,b):
#     return np.exp(b+m*x)/(1+np.exp(b+m*x))
# rp = 0.386
# rgoe = 0.531

# length = rgoe - rp

# m = 1
# b = -5#np.log(0.3)

# rs = f(Ks,m,b)*length + rp

# plt.figure(figsize=(12,8))

# plt.plot(Ks, rs, 'b.-', lw=2)
# plt.hlines(rp, min(Ks), max(Ks), ls='dashed', lw=2, alpha=0.5)
# plt.hlines(rgoe, min(Ks), max(Ks), ls='dashed', lw=2, alpha=0.5)
# plt.xlabel(r'$K$ chaos parameter')
# plt.xlim(min(Ks),max(Ks))
# # plt.xlim(theta_min_for_plot,theta_max_for_plot)
# plt.ylabel(r'$r$')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f'r_figura_analisis.png')
# #%% extra: cij vs ei (figura de analisis)
# from scipy.stats import norm

# j = np.linspace(-5,5,1000)

# cij = norm.pdf(j,loc=0,scale=2)
# histo = norm.rvs(loc=0,scale=2,size=1000)

# plt.figure(figsize=(12,8))
# plt.plot(j, cij, 'r-', lw=2)
# plt.hist(histo,density=True, bins='auto')
# plt.xlabel(r'$E_i$ energy')
# plt.xlim(min(j),max(j))
# # plt.xlim(theta_min_for_plot,theta_max_for_plot)
# plt.ylabel(r'$|c_{ij}|^2$')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f'cij_vs_ei_figura_analisis.png')
# #%% extra: Aij vs wi (figura de analisis)
# from scipy.stats import norm

# j = np.linspace(-5,5,1000)

# cij = norm.pdf(j,loc=0,scale=2)
# histo = norm.rvs(loc=0,scale=2,size=1000)

# plt.figure(figsize=(12,8))
# plt.plot(j, cij, 'r-', lw=2)
# plt.hist(histo,density=True, bins='auto')
# plt.xlabel(r'$\omega_i$ frecuency')
# plt.xlim(min(j),max(j))
# # plt.xlim(theta_min_for_plot,theta_max_for_plot)
# plt.ylabel(r'$|A_{i}|$')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f'Ai_vs_wi_figura_analisis.png')
# #%%

# fig, ax = plt.subplots(2,3, figsize=(16,8))

# for fila in range(2):
#     for columna in range(3):
#         i = 3*fila+columna
#         thetas = data_Ks[:,0,i]
#         Is = data_Ks[:,1,i]
#         ax[fila,columna].set_title(f'K={Ks[i]:.1f}')
#         ax[fila,columna].plot(thetas, Is, 'b.', ms=0.2)
#         ax[fila,columna].set_xlabel(r'$\theta$')
#         ax[fila,columna].set_xlim(0,1)
#         # plt.xlim(theta_min_for_plot,theta_max_for_plot)
#         ax[fila,columna].set_ylabel(r'$p_\theta$')
#         ax[fila,columna].set_ylim(0,1)
#         # plt.ylim(I_min_for_plot,I_max_for_plot)
# plt.tight_layout()
# plt.savefig(f'standar_map_K{Ks}.png',dpi=300)
