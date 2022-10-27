#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 20:38:28 2022

@author: tomasnotenson
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from qutip import *
from time import time
from tqdm import tqdm # customisable progressbar decorator for iterators
from sklearn.linear_model import LinearRegression #Regresión Lineal con scikit-learn

# parameters
N = 5000
Kpaso = 20/100
Ks = np.arange(0,20,Kpaso)
# some filename definitions
opA = 'X'
opB = 'P'
operatorss = 'A'+opA+'_B'+opB
time_lim = int(2e1+1) # number of kicks
phi = []#np.identity(N)#-sustate
modif = ''#'_sust_evec0'
state = '_Tinf_state'
flag = '4pC_FFT_with'+state
#%% open file for writing

# f = open("pendientes_RAFA.txt", "a")
#%% load file .npz
file = flag+f'_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_basis_size{N}_time_lim{time_lim}'+operatorss+modif+'.npz'#+'_evolucion_al_reves' _centro{n0}
archives = np.load(file)
O1_Ks = archives['O1']

K = 7.6

k = round(K/Kpaso)

# create plot
plt.figure(figsize=(16,8))
plt.title(f'A={opA}, B={opB}. N={N}')

# variables for plot
O1s = O1_Ks[k]
y = np.abs(O1s)/N#np.log10(O1s)

times = np.arange(0,time_lim)
x = times

shift = 0

# fit limits
linf=14+shift
lsup=21+shift
# variables for fit
xt = x[linf:lsup].reshape(-1,1)
yt = np.log(y[linf:lsup])
# fit
regresion_lineal = LinearRegression() # creamos una instancia de LinearRegression
regresion_lineal.fit(xt, yt) 

# estimated parameters
m = regresion_lineal.coef_
b = regresion_lineal.intercept_

# # fit limits
# linf=lsup-1
# lsup=12
# variables for fit
# xt = x[linf:lsup].reshape(-1,1)
# yt = np.log(y[linf:lsup])
# # fit
# regresion_lineal = LinearRegression() # creamos una instancia de LinearRegression
# regresion_lineal.fit(xt, yt) 

# # estimated parameters
# m2 = regresion_lineal.coef_
# b2 = regresion_lineal.intercept_

# # # fit limits
# linf=lsup-1
# lsup=11
# # variables for fit
# xt = x[linf:lsup].reshape(-1,1)
# yt = np.log(y[linf:lsup])
# # fit
# regresion_lineal = LinearRegression() # creamos una instancia de LinearRegression
# regresion_lineal.fit(xt, yt) 

# # estimated parameters
# m3 = regresion_lineal.coef_
# b3 = regresion_lineal.intercept_

# # # fit limits
# linf=11
# lsup=16
# # variables for fit
# xt = x[linf:lsup].reshape(-1,1)
# yt = np.log(y[linf:lsup])
# # fit
# regresion_lineal = LinearRegression() # creamos una instancia de LinearRegression
# regresion_lineal.fit(xt, yt) 

# # estimated parameters
# m4 = regresion_lineal.coef_
# b4 = regresion_lineal.intercept_


# f.write(f'{Ks[k]:.1f}, {np.exp(m[0]/2)}')#, {np.exp(m2[0]/2)}')#', {np.exp(m3[0]/2)}')#', {np.exp(m4[0]/2)}')
# f.write('\n')

print(f'K={Ks[k]}, eval = {np.exp(m[0]/2)}')#',  {np.exp(m2[0]/2):.4f}')

plt.plot(x, y, '.-', ms=10, lw=1.5,  color='blue', label=f'K={Ks[k]:.2f}')
plt.plot(x, np.exp(m*x+b), '-r', lw=1.5, label=f'{np.exp(m[0]/2):.3f}**(2*t) ajuste', alpha=0.6)
plt.fill_between(x[linf:lsup],0*np.ones(len(xt)),1*np.ones(len(xt)), alpha=0.3)
# plt.plot(x, np.exp(m2*x+b2), '-g', lw=1.5, label=f'{np.exp(m2[0]/2):.3f}**(2*t) ajuste', alpha=0.6)
# plt.plot(x, np.exp(m3*x+b3), '-c', lw=1.5, label=f'{np.exp(m3[0]/2):.3f}**(2*t) ajuste', alpha=0.6)
# plt.plot(x, np.exp(m4*x+b4), '-k', lw=1.5, label=f'{np.exp(m4[0]/2):.3f}**(2*t) ajuste', alpha=0.6)
plt.ylabel(r'$O_1$')
plt.xlabel(r'$t$')
plt.xticks(times)
plt.yscale('log')
plt.ylim(1e-6,1)
# plt.xlim(-0.01,15)
plt.grid()
plt.legend(loc = 'best')

file = flag+f'_K{Ks[k]:.1f}_basis_size{N}_time_lim{time_lim}'+operatorss+modif+'.png'
plt.savefig(file, dpi=80)
#%% cargo el archivo (con correrlo 1 vez, alcanza)
file = flag+f'_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_basis_size{N}_time_lim{time_lim}'+operatorss+modif+'.npz'#+'_evolucion_al_reves' _centro{n0}
archives = np.load(file)
O1_Ks = archives['O1']

#%% OJO! ajuste de pendientes automático REVISAR!!!
from scipy.optimize import curve_fit
def lin_mod(x,b0,b1):
    '''
    Linear function

    Parameters
    ----------
    x : array_like or float
        domain value/s
    b0 : float
        intercept parameter
    b1 : float
        slope parameter

    Returns
    -------
    float 
    f(x) value of linear model
    '''
    return b0 + b1*x

def ajuste_curve_fit(x,y,linf,lsup,tipo='exponential'):
    '''
    Linear fit permormed by curve fit xD

    Parameters
    ----------
    x : array_like 
        domain values
    x : array_like 
        image values
    linf : integer
        initial time for fit interval
    lsup : integer
        final time for fit interval

    Returns
    -------
    pest : array_like
        parameter values
    pcov : array_like
        covariance matrix

    '''        
    # variables for fit
    if tipo=='exponential':
        xt = x[linf:lsup]#.reshape(-1,1)
    elif tipo=='power law':
        xt = np.log(x[linf:lsup])#.reshape(-1,1)
    # print(tipo, xt)
    yt = np.log(y[linf:lsup])
    pest, pcov = curve_fit(lin_mod, xt, yt)
    return pest,pcov


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
    print(f'n={n}, df={df}, xbar={xbar}, sigma_squared={sigma_squared}, Sx={Sx}')
    print(f'var0={Var0}, var1={Var1}, SE0={SE0}, SE1={SE1}')
    return SE0,SE1

def ajuste(x,y,linf,lsup,tipo='expontential'):
    
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

def plotO1vst(x,y,solo=False,*args,**kwargs):
        
    # create plot
    plt.figure(figsize=(16,8))
    plt.title(f'N={N}')
    plt.plot(x, y, '.-', ms=10, lw=1.5,  color='blue', label=f'K={Ks[k]:.2f}')
    if not solo:
        tamanio = lsup-linf
        plt.fill_between(x[linf:lsup],0*np.ones(tamanio),1*np.ones(tamanio), alpha=0.3)
        if tipo == 'exponential':
            plt.plot(x, np.exp(m*x+b), '-r', lw=1.5, label=f'{np.exp(m/2):.3f}**(2*t) ajuste', alpha=0.6)
            plt.fill_between(x, np.exp((m-SEm)*x+b+SEb), np.exp((m+SEm)*x+b-SEb),
                      color='red', alpha=0.2)
        elif tipo == 'power law':
            plt.plot(x, np.exp(m*np.log(x)+b), '-r', lw=1.5, label=f'slope = {m}+-{SEm}', alpha=0.6)
            plt.fill_between(x, np.exp((m-SEm)*np.log(x)+b+SEb), np.exp((m+SEm)*np.log(x)+b-SEb),
                      color='red', alpha=0.2)
        
    # plt.plot(x, np.exp(m2*x+b2), '-g', lw=1.5, label=f'{np.exp(m2[0]/2):.3f}**(2*t) ajuste', alpha=0.6)
    # plt.plot(x, np.exp(m3*x+b3), '-c', lw=1.5, label=f'{np.exp(m3[0]/2):.3f}**(2*t) ajuste', alpha=0.6)
    # plt.plot(x, np.exp(m4*x+b4), '-k', lw=1.5, label=f'{np.exp(m4[0]/2):.3f}**(2*t) ajuste', alpha=0.6)
    plt.ylabel(r'$O_1$')
    plt.xlabel(r'$t$')
    plt.xticks(times)
    plt.yscale('log')
    plt.ylim(1e-6,1)
    # plt.xlim(-0.01,15)
    plt.grid()
    plt.legend(loc = 'best')
    plt.show()
    file = flag+f'_K{Ks[k]:.1f}_basis_size{N}_time_lim{time_lim}.png'
    plt.savefig(file, dpi=80)
    return 


stand_by = input('Desea cambiar el K?\nPresione Enter si NO\nIngrese el nuevo valor de K si SI\n: ')

if stand_by!='':
    K = float(stand_by)    

k = round(K/Kpaso)

# variables for plot
O1s = O1_Ks[k]
y = np.abs(O1s/N)#np.log10(O1s)

times = np.arange(0,time_lim)
x = times

# plotO1vst(x,y,solo=True)

if K<12:
    tipo = 'power law'
    linf = int(input("límite inferior del ajuste:"))
    lsup = int(input("límite superior del ajuste:"))+1
    
    # b,m,errb,errm = ajuste(x,y,linf,lsup)
    # m = m[0]
    pest,pcov = ajuste_curve_fit(x,y,linf,lsup,tipo=tipo)
    b, m = pest
    SEb,SEm = np.sqrt(np.diag(pcov))
    # open file
    # f = open("pendientes_RAFA.txt", "a")
    # f.write(f'{Ks[k]:.1f}, {np.exp(m/2)}, {linf}, {lsup}')#, {np.exp(m2[0]/2)}')#', {np.exp(m3[0]/2)}')#', {np.exp(m4[0]/2)}')
    # f.write('\n')
    # f.close()
    print(f'K={Ks[k]}, slope = {m}+-{SEm}, intercept = {b}+-{SEm}')
    # print(f'K={Ks[k]}, eval = {np.exp(m/2)}')
    
    plotO1vst(x,y,m=m,b=b,SEb=SEb,SEm=SEm)
    
    plt.xscale('log')    
else: 
    # tipo = 'exponential'
    linf = int(input("límite inferior del ajuste:"))
    lsup = int(input("límite superior del ajuste:"))+1
    
    # b,m,errb,errm = ajuste(x,y,linf,lsup)
    # m = m[0]
    pest,pcov = ajuste_curve_fit(x,y,linf,lsup)
    b, m = pest
    SEb,SEm = np.sqrt(np.diag(pcov))
    # open file
    # f = open("pendientes_RAFA.txt", "a")
    # f.write(f'{Ks[k]:.1f}, {np.exp(m/2)}, {linf}, {lsup}')#, {np.exp(m2[0]/2)}')#', {np.exp(m3[0]/2)}')#', {np.exp(m4[0]/2)}')
    # f.write('\n')
    # f.close()
    print(f'K={Ks[k]}, eval = {np.exp(m/2)}')
    # print(f'K={Ks[k]}, eval = {np.exp(m/2)}')
    
    plotO1vst(x,y,m=m,b=b,SEb=SEb,SEm=SEm)

#%% pendientes vs K

listas = np.genfromtxt('pendientes.txt', usecols=(0,1), delimiter=',')#, delimiter=',')
Ks = listas[:,0]
pendientes = listas[:,1]
pend_normed = (pendientes - min(pendientes))/(max(pendientes) - min(pendientes))

plt.figure(figsize=(16,8))
plt.title(f'A={opA}, B={opB}. N={N}')
plt.plot(Ks, pend_normed, '.-b', lw=1.5, label='normed')
plt.plot(Ks, pendientes, '.-r', lw=1.5, label='not normed')
plt.ylabel(r'$\lambda$')
plt.xlabel(r'$K$')
plt.xticks(Ks[::5], rotation=0, fontsize=12)
plt.ylim(-0.01,1.01)
# plt.xlim(-0.2,max(times)+0.2)
plt.grid(True)
plt.legend(loc = 'best')
plt.savefig(f'pendientes_vs_K_N5000_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_basis_size{N}_time_lim{time_lim}.png', dpi=80)
