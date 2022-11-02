#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 20:33:21 2022

@author: tomasnotenson
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression #Regresión Lineal con scikit-learn

### Podemos agregar una función para propagar errores 

def SE_par_lm(x,y,b0,b1):
    '''
    Compute standard errors for the linear fit parameters

    Parameters
    ----------
    x : array_like
        x data
    y : array_like
        y data
    b0 : float
        intercept b0 estimation 
    b1 : float
        slope b1 estimation 

    Returns
    -------
    SE0 : float
        intercept b0 standard error 
    SE1 : float
        slope b1 standard error

    '''
    n = len(x)
    df = n-2
    xbar = np.mean(x)
    Sx = np.sum((x-xbar)**2)/(n-1)
    sigma_squared = np.sum((y-b0-b1*x)**2)/df
    Var0 = sigma_squared*(1/n + xbar**2/((n-1)*Sx))
    SE0 = np.sqrt(Var0)
    Var1 = sigma_squared/((n-1)*Sx)
    SE1 = np.sqrt(Var1)
    # print(f'n={n}, df={df}, xbar={xbar}, sigma_squared={sigma_squared}, Sx={Sx}')
    # print(f'var0={Var0}, var1={Var1}, SE0={SE0}, SE1={SE1}')
    return SE0,SE1

def ajuste(x,y,linf,lsup,tipo='power law'):
    
    # variables for fit
    if tipo=='exponential':
        xt = x[linf:lsup].reshape(-1,1)
    elif tipo=='power law':
        xt = np.log(x[linf:lsup]).reshape(-1,1)
    yt = np.log(y[linf:lsup])
    # fit
    regresion_lineal = LinearRegression() # creamos una instancia de LinearRegression
    regresion_lineal.fit(xt, yt) 

    # estimated parameters
    b0 = regresion_lineal.intercept_
    b1 = regresion_lineal.coef_
    
    SE0, SE1 = SE_par_lm(x[linf:lsup],y[linf:lsup],b0,b1)
    return b0,b1,SE0,SE1

def power_law_by_exp(t, resonancia):
    return np.exp(-resonancia*t)

points = 5
resonancias = np.linspace(0,1,points)

t = np.arange(11)

fx = 0 
for r in range(len(resonancias)):
    fx += power_law_by_exp(resonancias[r],t)

linf = 3
lsup = 6

b,m,SEb,SEm = ajuste(t,fx,linf,lsup)

plt.figure(figsize=(10,10))
plt.title(f'cant de resonancias = {points}. límites: [{min(resonancias):.1f},{max(resonancias):.1f}]. Espaciado lineal')
plt.plot(t, fx, '.-', label=r'$O_1$')
plt.plot(t, np.exp(m*np.log(t)+b), '-r', lw=1.5, label=f'ajuste', alpha=0.6)
# plt.fill_between(t, np.exp((m-SEm)*np.log(t)+b+SEb), np.exp((m+SEm)*np.log(t)+b-SEb),
          # color='red', alpha=0.2)
plt.xlabel(r'$t$')
# plt.xlim(theta_min_for_plot,theta_max_for_plot)
plt.ylabel(r'$\sum_i \exp(-\epsilon_i.t) $')
plt.yscale('log')
plt.xscale('log')
# plt.xlim(0,50)
# plt.ylim(0,1)
# plt.ylim(I_min_for_plot,I_max_for_plot)
plt.tight_layout()
plt.savefig('power_law_by_exponential.png', dpi=80)
#%% 

ps = np.logspace(0,3,100)

ms = np.zeros(len(ps))
errms = np.zeros(len(ps))

# t = np.arange(101)

for i,points in enumerate(ps):
    points = round(points)
    
    resonancias = np.linspace(0,1,points)

    fx = 0 
    for r in range(len(resonancias)):
        fx += power_law_by_exp(resonancias[r],t)
    
    _,b1,_,SE1 = ajuste(t,fx,linf,lsup)
    
    ms[i] = b1; errms[i] = SE1
    
plt.figure(figsize=(10,10))
plt.title(r'$t_{max}$'+f' = {max(t)}.\nlímites: [{min(resonancias):.1f},{max(resonancias):.1f}]. Espaciado lineal')
# plt.errorbar(ps, ms, errms)
plt.plot(ps, ms)
plt.hlines(np.mean(ms[-10:]), min(ps), max(ps), color='red', ls='dashed', alpha=0.4)
plt.xlabel('points')
# plt.xlim(0,1)
# plt.xlim(theta_min_for_plot,theta_max_for_plot)
plt.ylabel(r'$m$')
# plt.yscale('log')
# plt.xscale('log')
# plt.ylim(0,1)
# plt.ylim(I_min_for_plot,I_max_for_plot)
plt.tight_layout()
plt.savefig('m_vs_points_power_law_by_exponential.png', dpi=80)