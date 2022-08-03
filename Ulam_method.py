#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 16:26:03 2022

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
from scipy.sparse.linalg import eigs
import scipy.sparse as ss

plt.rcParams.update({
"text.usetex": True,
"font.family": "sans-serif",
"font.sans-serif": ["Helvetica"], "font.size": 16})

dpi = 2*np.pi

@jit
def standard_map(q,p,K):
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
    rq = np.random.uniform()*paso
    rp = np.random.uniform()*paso
    
    q = qj+rq
    p = pj+rp
    
    # print('CI', q,p)
    return q,p

@jit
def cae_en_celda(qf,pf,qi,pi,paso):
    cond1 = (0<(qf-qi) and (qf-qi)<paso)
    cond2 = (0<(pf-pi) and (pf-pi)<paso)
    
    cond = (cond1 and cond2)
    # if cond:
    #     print('cond1', qf, pf, 'dif', (qf-qi), (pf-pi))
    return cond

@jit
def CI_and_evolution(qj,pj,paso,K,mapa='normal',ruido=10):
    # tomo una CI dentro de la celda j
    q, p = CI(qj, pj, paso)
    # q, p = gaussian_kernel(q, p, paso, ruido)
    
    # evoluciono un paso 
    qf, pf, nqi, npi = evolution(q, p, K, mapa)
    
    return qf, pf, nqi, npi

@jit
def gaussian_kernel(q,p,paso,ruido,modulo=1):
    # rv = norm(0, ruido*paso)
    qf = (q + np.random.normal(0,paso/ruido))%modulo
    pf = (p + np.random.normal(0,paso/ruido))%modulo
    
    return qf, pf

@jit
def evolution(q,p,K,mapa='normal'):
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
def Ulam(N,Nx,paso,Nc,K,mapa,ruido=10,modulo=1):
    
    S = np.zeros((N, N), dtype=np.float64)
    
    for j in tqdm(range(N), 'Ulam approximation'):
        # celda j fija
        # límites inf. de la celda j fija
        qj, pj = qp_from_j(j, Nx, paso, mapa)
        # repito Nc veces
        for nj in range(Nc):
            # tomo una CI random dentro de la celda j 
            # y luego evoluciono un paso
            qf, pf, nqi, npi = CI_and_evolution(qj,pj,paso,K,mapa,ruido)
            qf, pf = gaussian_kernel(qf, pf, paso, ruido,modulo)
            nqi, npi = n_from_qp(qf, pf, paso,mapa)
            # print('llega')
            i = int(nqi+npi*Nx)
            
            # print(qf, pf, nqi, npi, i)
            
            S[i,j] += 1
        S[:,j] /= Nc
        assert  (np.sum(S[:,j]) - 1.0) < 1e-3, 'ojo con sum_i Sij'
    return S

@jit
def Ulam_one_trayectory(N,Nx,paso,Nc,K,mapa,symmetry=True):
    Nmitad = N//2
    
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
        
        assert j < Nx**2//2, f'number of cell j={j} out of bounds. Con nqi={nqi} y npi={npi}.'
        
        # evoluciono un tiempo
        qf, pf, nqi, npi = evolution(qf,pf,K,mapa)
        qf, pf = gaussian_kernel(qf, pf, paso,ruido,modulo=1)
        
              
        nqi, npi = n_from_qp(qf, pf, paso, mapa)
        
        # número de celda'Ulam'#
        i = int(nqi+npi*Nx)
        
        assert i < Nx**2//2, f'number of cell i={i} out of bounds. Con nqi={nqi} y npi={npi}.'
        
        
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

@jit
def eigenvec_j_to_qp(eigenvector, mapa='normal'):
    N = len(eigenvector)
    Nx = int(np.sqrt(N))
    paso = 1
    eig_res = np.zeros((Nx,Nx))
    for j in range(N):
        q,p = qp_from_j(j, Nx, paso, mapa)
        # print(int(q),int(p))
        eig_res[int(q),int(p)] = eigenvector[j]
    return eig_res
    
#%%
Ns = np.arange(104,122,2) #np.arange(124,128,2)#np.concatenate((np.arange(20,71,1),np.arange(120,131,2)))#2**np.arange(5,8)
es = [1e10]#1/1.5]#np.logspace(5,6,1) 
resonancias = np.zeros((len(Ns),len(es)))
        
mapa = 'normal'#'absortion'#'dissipation'#'normal'#'cat'#'Harper'#
method = 'Ulam'#'one_trayectory'#
eta = 0.3
a = 2
cx = 1

K = 19.74#0.971635406

Nc = int(1e3)
#%%

for ni in tqdm(range(len(Ns)), desc='loop ni'):
    for ri in tqdm(range(len(es)), desc='loop ri'):
                
        Neff = Ns[ni]
        Nx = int(cx*Neff)
        N = Nx**2
        ruido = es[ri]
        
        print(f'N={Neff}d',f'e={ruido}')
        
        paso = 1/Nx
                
        if (mapa=='normal' or mapa=='absortion' or mapa=='cat' or mapa=='Harper'):
            if method=='one_trayectory':
                S = Ulam_one_trayectory(N, Nx, paso, Nc, K, mapa)    
            elif method=='Ulam':
                S = Ulam(N, Nx, paso, Nc, K, mapa, ruido)
            
        elif mapa=='dissipation':
            S = Ulam_one_trayectory(N, Nx, paso, Nc, K, mapa)
            
        # hinton(S)
        # mitad = int(S.shape[0]/2)
        # diagonalize operator
        t0=time()
        e, evec = eigs(S)#np.linalg.eig(S)
        t1=time()
        print(f'\nDiagonalización: {t1-t0} seg')
        flag = f'Ulam_approximation_mapa{mapa}_Sij_eigenvals_N{Neff}_ruido{ruido}_grilla{cx}N_K{K}_Nc{Nc}'
        np.savez(flag+'.npz', e=e, evec=evec[:6])
        del S; del evec; 
        # plot eigenvalues
        # plt.figure(figsize=(16,8))
        # plt.plot(e.real, e.imag, 'r*', ms=10, alpha=0.7)#, label=r'$\epsilon =${}'.format(es[ri]))
        # plt.xlabel(r'$\Re(\lambda)$')
        # plt.ylabel(r'$\Im(\lambda)$')
        # plt.xlim(0.49,1.1)
        # plt.ylim(-0.1,0.1)
        # plt.grid(True)
        del e;
        
        
        # x = w.real
        # # extract imaginary part using numpy
        # y = w.imag
          
        # ############################################################################
        # # fig1 = plt.figure(4)
        # plt.scatter(x, y, s=5, alpha=0.8, label='Diego')
        
        # theta = np.linspace(0, 2*np.pi, 100)
           
        # r= 1
        
        # x = r*np.cos(theta)
        # y = r*np.sin(theta)
           
        # plt.plot(x,y,color='b', lw=1)
        # plt.legend(loc='upper right')
        # plt.savefig(f'autovalores_N{Neff}_Kc_standardmap.png', dpi=100)
        # plt.show()
        # plt.close()
        
        # i = -2
        # while e.real[i]<0:
        #     i+=1
        # resonancias[ni,ri] = np.sort(np.abs(e))[i]
# flag = f'resonancias_method{method}_mapa{mapa}_Nc{Nc}_Nmin{min(Ns)}_Nmax{max(Ns)}_lenN{len(Ns)}_rmin{min(es)}_rmax{max(es)}_lenr{len(es)}'
# np.savez(flag+'.npz', resonancias=resonancias)
# plt.legend(loc='best')
# plt.savefig(f'Comparacion_zoom_Ulam_Diego_Neff{Neff}_Nx{Nx}_K{K}.png', dpi=100)
#%% ploteo un autoestado
# vec = eigenvec_j_to_qp(evec[0])
# sns.heatmap(vec)

#%%
# Ns = np.concatenate((np.arange(50,64,2),np.arange(100,172,2),np.arange(200,208,2),np.arange(212,232,2)))
Ns = np.arange(100,104,2)
resonancias = np.zeros((len(Ns),len(es)))

for ni in tqdm(range(len(Ns)), desc='loop ni'):
    for ri in tqdm(range(len(es)), desc='loop ri'):
        Neff = Ns[ni]
        ruido = es[ri]
        print(ruido)
        flag = f'Ulam_approximation_mapa{mapa}_Sij_eigenvals_N{Neff}_ruido{ruido}_grilla{cx}N_K{K}_Nc{Nc}'
        archives = np.load(flag+'.npz')
        e = archives['e']
        resonancias[ni,ri] = np.abs(e)[1]

# flag = f'resonancias_method{method}_mapa{mapa}_Nc{Nc}_Nmin{min(Ns)}_Nmax{max(Ns)}_lenN{len(Ns)}_rmin{min(es)}_rmax{max(es)}_lenr{len(es)}'
# archives = np.load(flag+'.npz')

# resonancias = archives['resonancias']


# e_abs = np.sort(np.abs(e))[::-1]
# gammas = -2*np.log(e_abs)
# # gammas = -2*np.log(resonancias[:,0])#e_abs)
# js = np.arange(len(e_abs))
# # js = np.arange(len(resonancias))

# hastaj = len(e_abs)-1#20

# dom = np.linspace(0,hastaj,1000)

# plt.figure()
# plt.plot(js[:hastaj], gammas[:hastaj], 'r*', ms=5)
# plt.plot(dom, gammas[1]*(dom)**2, 'g-', lw=1, label=r'$\gamma_1 j^2$')
# plt.xlabel(r'$j$')
# plt.ylabel(r'$\gamma_j$')
# # plt.legend(loc='best')
# plt.ylim(-0.001,5)
# plt.xlim(0,7)
# plt.grid(True)
# # plt.show()
# # plt.savefig('decay_rates_'+flag+'.png', dpi=100)



# %%
import matplotlib as mpl

# Ns = np.arange(21,71,1)#np.concatenate((np.arange(17.6,71,2.6),np.arange(74.4,80.8,1.6))) #2**np.arange(5,8)
# es = np.logspace(-6,-1,6) 

# resonancias = np.zeros((len(Ns),len(es)))

# for ni in tqdm(range(len(Ns)), desc='loop ni'):
#     for ri in tqdm(range(len(es)), desc='loop ri'):
#         Neff = round(Ns[ni],1)
#         ruido = es[ri]
#         # print(Neff, ruido)
#         flag = f'Ulam_approximation_mapa{mapa}_Sij_eigenvals_N{Neff}_ruido{ruido}_grilla{cx}N_K{K}_Nc{Nc}'
#         print(flag)
#         archives = np.load(flag+'.npz')
#         e = archives['e']#['arr_0']#
#         resonancias[ni,ri] = np.abs(e)[1]
# # archives = np.load(flag+'.npz')

# # resonancias = archives['resonancias']

# # resonancias en función del ruído y de N

fit = np.linspace(0,1,len(Ns))

cmap = mpl.cm.get_cmap('viridis')
color_gradients = cmap(fit)  

titulo_plot = 'Standard Map'#'Arnold Perturbed Cat Map'#'Harper map'#

markers = ['-o','--*',':s','-^']
# plt.figure()
plt.title(titulo_plot)
fig, (ax1,ax2) = plt.subplots(1,2, gridspec_kw={'width_ratios': [30, 1]}, figsize=(16,8))



for ni in range(len(Ns)):
# ni = 2
    ax1.plot(es, resonancias[ni,:], markers[(ni)%4], lw=0.5, ms=10, color=color_gradients[ni], alpha=0.8, label=f'N={Ns[ni]}')
ax1.set_xlabel(r'$\epsilon$')
ax1.set_ylabel(r'$\lambda$')
ax1.set_xscale("log")
# ax1.legend(loc='best')
ax1.grid()
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                orientation='vertical',
                                ticks=fit[::5], )
ax2.set_yticklabels([round(N,1) for N in Ns[::5]])
ax2.set_ylabel(r'$N$', rotation=0)


# plt.savefig('resonancias_vs_e_distintos_N_'+flag+'.png', dpi=100)
#%% en funcion de N

# Ns = np.concatenate((np.arange(17.6,71,2.6),np.arange(74.4,80.8,1.6))) #2**np.arange(5,8)
# es = np.logspace(-6,-1,6) 

# resonancias = np.zeros((len(Ns),len(es)))

# for ni in tqdm(range(len(Ns)), desc='loop ni'):
#     for ri in tqdm(range(len(es)), desc='loop ri'):
#         Neff = round(Ns[ni],1)
#         ruido = es[ri]
#         # print(Neff, ruido)
#         flag = f'Ulam_approximation_mapa{mapa}_Sij_eigenvals_N{Neff}_ruido{ruido}_grilla{cx}N_K{K}_Nc{Nc}'
#         print(flag)
#         archives = np.load(flag+'.npz')
#         e = archives['arr_0']
#         resonancias[ni,ri] = np.abs(e)[1]
# # archives = np.load(flag+'.npz')

# # resonancias = archives['resonancias']

# # resonancias en función del ruído y de N

fit = np.linspace(0,1,len(es))

cmap = mpl.cm.get_cmap('viridis')
color_gradients = cmap(fit)  

textstr = r'$\epsilon = $'+f'{es[0]:e}'

res_mean = np.mean(resonancias[resonancias.shape[0]//2:,:])

nc = 2*Nc/(0.42*Ns**2)

markers = ['-o','--*',':s','-^']
# plt.figure()
plt.title(titulo_plot)
fig, (ax1,ax2) = plt.subplots(1,2, gridspec_kw={'width_ratios': [30, 1]}, figsize=(10,6))

rerr = 1/np.sqrt(nc)

for ri in range(len(es)):
# ni = 2
    ax1.plot(Ns, resonancias[:,ri], markers[(ri)%4], lw=0.5, ms=10, color=color_gradients[ri], alpha=0.8)
    ax1.errorbar(Ns, resonancias[:,ri], rerr, marker=None, linestyle='',
         mec=color_gradients[ri], ms=10, mew=3)
    ax1.set_xlabel(r'$N$')
ax1.set_ylabel(r'$\lambda$')
# ax1.set_xscale("log")
# ax1.legend(loc='best')
ax1.grid()
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
ax1.text(np.mean(Ns), min(resonancias)+0.005, textstr, fontsize=14,
        verticalalignment='top', bbox=props)

# cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                # orientation='vertical',
                                # ticks=fit, )
# ax2.set_ylabel(r'$\epsilon$', rotation=0)
# ax2.set_yticklabels([f'{e:.2f}' for e in es])
# ax2.set_label('N')19.74

plt.savefig('resonancias_vs_N_distintos_e_'+flag+'.png', dpi=100)
#%% 
from sklearn.linear_model import LinearRegression #Regresión Lineal con scikit-learn
from scipy.stats import linregress
x = 1/Ns
y = resonancias[:,-1]

yerr = 1/np.sqrt(nc)
# instruimos a la regresión lineal que aprenda de los datos (x,y)
xt = x.reshape(-1,1)

regresion_lineal = LinearRegression() # creamos una instancia de LinearRegression

regresion_lineal.fit(xt, y) 
# vemos los parámetros que ha estimado la regresión lineal
m = regresion_lineal.coef_
b = regresion_lineal.intercept_

merr = np.std(yerr)*np.sqrt(len(yerr)/(len(yerr)-1))/np.sqrt(np.var(x)*len(x))
berr = np.std(yerr)*np.sqrt(1/(len(yerr)-1) + np.mean(x)**2/np.var(x)*len(x))

bp, mp = np.polynomial.polynomial.polyfit(x, y, 1, w = [1.0 / ty for ty in yerr], full=False)
print('mp = ' + str(mp) +f' +- {merr}' + ', bp = ' + str(bp) +f' +- {berr}' )

dom = np.linspace(0,0.05,10000)

print('m = ' + str(m[0]) +f' +- {merr}' + ', b = ' + str(b) +f' +- {berr}' )

plt.figure(figsize=(16,8))

plt.plot(x,y,'r.')
plt.plot(dom,m*dom+b,'-b',label=r'$mN^{-1}+b$')
plt.errorbar(x, y, yerr, marker=None, linestyle='',
     mec=color_gradients[ri], ms=10, mew=3)
plt.xlabel(r'$N^{-1}$')
plt.ylabel(r'$\lambda$')
plt.grid(True)
plt.legend(loc='best')
plt.xlim(0,0.05)
# plt.show()

plt.savefig(f'extrapolacion_resonancia_clasica_Ninfty_ajuste_lineal_method{method}_mapa{mapa}_Nc{Nc}_Nmin{min(Ns)}_Nmax{max(Ns)}_lenN{len(Ns)}.png', dpi=80)
#%% decay rates extrapolation
import statsmodels.api as sm
x = 1/Ns
y = -2*np.log(resonancias[:,0])

# yerr = 1/np.sqrt(Ns)
# instruimos a la regresión lineal que aprenda de los datos (x,y)
xt = x.reshape(-1,1)

regresion_lineal = LinearRegression() # creamos una instancia de LinearRegression

regresion_lineal.fit(xt, y) 
# vemos los parámetros que ha estimado la regresión lineal
m = regresion_lineal.coef_
b = regresion_lineal.intercept_

merr = np.std(yerr)*np.sqrt(len(yerr)/(len(yerr)-1))/np.sqrt(np.var(x)*len(x))
berr = np.std(yerr)*np.sqrt(1/(len(yerr)-1) + np.mean(x)/np.sqrt(np.var(x)*len(x)))

print('m = ' + str(m[0]) + ', b = ' + str(b) +f' +- {berr}' )

lambda_lineal = np.exp(-1/2*b)

plt.figure(figsize=(16,8))

plt.plot(x,y,'r.')
plt.plot(x,m*x+b,'-b',label=r'$mN^{-1}+b$')
plt.xlabel(r'$N^{-1}$')
plt.ylabel(r'$\gamma$')
plt.grid(True)
plt.legend(loc='best')
# plt.show()

plt.savefig(f'extrapolacion_decay_rate_clasica_Ninfty_ajuste_lineal_method{method}_mapa{mapa}_Nc{Nc}_Nmin{min(Ns)}_Nmax{max(Ns)}_lenN{len(Ns)}.png', dpi=80)
print('lambda=',np.exp(-1/2*b))
#%% ajuste más sofisticado a las tasas de decaimiento
from scipy.optimize import curve_fit

def f(x,a,b,c):
    return a*x**2+b*x+c

p0=[1.857,1000,0.089]
popt, pcov = curve_fit(f, x, y, p0=[0.0857,1370,0.389])
# popt = p0
plt.plot(x, y, 'r.')
plt.plot(x,m*x+b,'-b', lw=1,label=r'$mN^{-1}+b$')
plt.plot(x, f(x, *popt), 'g--', lw=1,
         label=r'$a (N^{-1})^2+b N^{-1} +c$')#r'$a(1+bN^{-1})^c$')
plt.xlabel(r'$N^{-1}$')
plt.ylabel(r'$\gamma$')
plt.grid(True)
plt.legend(loc='best')
# plt.show()
plt.savefig(f'extrapolacion_Shepelyansky_decay_rate_clasica_Ninfty_ajuste_lineal_method{method}_mapa{mapa}_Nc{Nc}_Nmin{min(Ns)}_Nmax{max(Ns)}_lenN{len(Ns)}.png', dpi=80)

print('lambda=',np.exp(-1/2*f(0,*popt)))
#%% Linear Regression: Summary 
import statsmodels.api as sm
X = x
X = sm.add_constant(X.ravel())
results = sm.OLS(y,X).fit()
print(results.summary() )
#%% round Ns in name files
# import os

# N = 80.79999999999995
# Nr = round(N,1)

# for ri in tqdm(range(len(es)), desc='loop ri'):
#     ruido = es[ri]

#     # Absolute path of a file
#     old_name = f"Ulam_approximation_mapanormal_Sij_eigenvals_N{N}_ruido{ruido}_grilla1N_K7_Nc1000000.npz"
#     new_name = f"Ulam_approximation_mapanormal_Sij_eigenvals_N{Nr}_ruido{ruido}_grilla1N_K7_Nc1000000.npz"
    
#     # Renaming the file
#     os.rename(old_name, new_name)



# %% plot eigenvalues for each N
# # Ns = 2**np.arange(5,8)
# Ncs = [int(1e3), int(1e4), int(1e5)]
# Nc = Ncs[1]

# markers = ['o','*','s','^']

# plt.figure(figsize=(10,10))

# r= 1
# theta = np.linspace(0, 2*np.pi, 100)

# x = r*np.cos(theta)
# y = r*np.sin(theta)
   
# plt.plot(x,y,color='b', lw=1)

# for i, Neff in enumerate(Ns):
#     flag = f'Ulam_approximation_mapa{mapa}_Sij_eigenvals_N{Neff}_grilla{cx}N_K{K}_Nc{Nc}'
#     archives = np.load(flag+'.npz')
#     e = archives['arr_0']
#     print(f'N={Neff} ', e.real[:10])
#     plt.plot(e.real, e.imag, markers[i], ms=10, alpha=0.6, label=f'N={Neff}')
# plt.xlabel(r'$\Re(\lambda)$')
# plt.ylabel(r'$\Im(\lambda)$')
# # plt.xlim(0.5,1)
# # plt.ylim(-0.025,0.025)
# plt.xlim(-1.1,1.1)
# plt.ylim(-1.1,1.1)
# plt.grid(True)
# plt.legend(loc='best')
# plt.savefig(flag+'.png', dpi=300)
#%%
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

# from math import *
# import time
# import socket
# import random
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# from tqdm import tqdm

# dpi=2.0*pi
# # The following 6 lines for the primary plot
# # Execution time is insignificant
# K=0.97
# # Ks = np.linspace(0.5, 8.5, 6)#np.array([0.5, 3, 6, 8, 10, 12])#[0.25, 0.5, 2, 3, 6, 10]
# number_of_trajectories=100
# number_of_points=1000
# theta_min_for_plot=0.
# theta_max_for_plot=1.#dpi
# I_min_for_plot=0.
# I_max_for_plot=1.#dpi

# # The following 6 lines good for a secondary island
# # Execution time is insignificant
# # K=1.0
# # number_of_trajectories=200
# # number_of_points=1000
# # theta_min_for_plot=1.7
# # theta_max_for_plot=4.6
# # I_min_for_plot=2.5
# # I_max_for_plot=3.8

# # The following 6 lines good for a ternary island
# # Execution time: 28seconds
# # K=1.0
# # number_of_trajectories=1000
# # number_of_points=10000
# # theta_min_for_plot=4.05
# # theta_max_for_plot=4.35
# # I_min_for_plot=3.6
# # I_max_for_plot=3.75
# N = number_of_points*number_of_trajectories

# data_Ks = np.zeros((N, 2, len(Ks)))

        
# thetas, Is = [[], []]
# Keff = K/(2*np.pi)
# for i_traj in range(number_of_trajectories):
#   theta=random.uniform(0,1)
#   I=random.uniform(0,1)
#   for i in range(number_of_points):
#     if ((theta>=theta_min_for_plot)&(theta<=theta_max_for_plot)&(I>=I_min_for_plot)&(I<=I_max_for_plot)):
#       # print (theta,I)
#       thetas.append(theta)
#       Is.append(I)
#     I=(I+Keff*sin(dpi*theta))%1
#     theta=(theta+I)%1
#   # print(' ')
# data_Ks[:,0,k] = thetas
# data_Ks[:,1,k] = Is
