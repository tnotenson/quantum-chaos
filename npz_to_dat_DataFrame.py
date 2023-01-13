#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 13:08:12 2022

Pasar de .npz a .dat en formato DataFrame

@author: tomasnotenson
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
#%% O1 vs t for many K values
filename = '4pC_FFT_with_Tinf_state_Kmin0.0_Kmax19.8_Kpaso0.2_basis_size5000_time_lim21AX_BP.npz'

data = np.load(filename)
O1 = data['O1']
Ks = data['Ks']

lista_df = []
for ki in tqdm(range(O1.shape[0]), desc='k loop'):
    K = Ks[ki]
    for ti in range(O1.shape[1]):
        lista_df.append([ti,np.abs(O1[ki,ti]),K])
    lista_df.append([None,None,None])
    


df = pd.DataFrame(lista_df,columns=['t','O1','K'])
df.to_csv('O1_KR_Kmin0.0_Kmax19.8_Kpaso0.2_basis_size5000_time_lim21' + ".dat", header=None, index=None, sep=' ', mode='a')
#%% epsilon vs ss for many K values

cte = 90*0.001
Npaso = 5
Ns = np.arange(50,81,Npaso)
K = 13 
ss = cte/Ns##np.arange(1,5)*1e-3

Kpaso = 0
Ks = [13]#np.arange(4,20.1,Kpaso)

# N = Ns[-1]

lista_df = []
for ni in tqdm(range(len(Ns)), desc='N loop'):
    if Ns[ni] != 80:
        N = Ns[ni]
        nvec = N**2
        filename = f'Blum_Agam_evals_K{K:.1f}_Nsfijo_N{N}_smin{min(ss):.4f}_smax{max(ss):.4f}.npz'
    
        archives = np.load(filename)
        es = archives['es']     
        ss = archives['ss']

    for ki in tqdm(range(len(Ks)), desc='k loop'):
        K = Ks[ki]
        for si in range(es.shape[0]):
            for ei in range(es.shape[1]):
                lista_df.append([ss[si],np.real(es[si,ei]),np.imag(es[si,ei]),np.abs(es[si,ei]),ei,K,N])#
    ### no dejo espacio en el .dat

df = pd.DataFrame(lista_df,columns=['s','Real e', 'Imag e', 'Abs e', 'e index', 'K','N'])#
df.to_csv(f'df_Blum_Agam_evals_K{K:.1f}_Nsfijo_Nmin{min(Ns):.0f}_Nmax{max(Ns):.0f}_smin{min(ss):.4f}_smax{max(ss):.4f}' + ".dat", header=None, index=None, sep=' ', mode='a')
#%% Fishman method
Ns = np.arange(14,31,2)
sigmas = [0.2]#np.linspace(0,1,100)
Kpaso = 0.1
Ks = np.arange(0,20,Kpaso)

lista_df = []
for si in tqdm(range(len(sigmas)), desc='sigmas loop'):
    sigma = sigmas[si]
    for ni in tqdm(range(len(Ns)), desc='N loop'):
        N = Ns[ni]
        filename = f'Fishman_N{N}_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_sigma{sigma}'
        archives = np.load(filename+'.npz')
        es = archives['es']
        Ks = archives['Ks']
        for ki in range(len(Ks)):
            K = Ks[ki]
            lista_df.append([np.real(es[ki]),np.imag(es[ki]),np.abs(es[ki]),K,N,sigma])#
df = pd.DataFrame(lista_df,columns=['Real e', 'Imag e', 'Abs e', 'K','N', 'sigma'])#
df.to_csv(f'df_Fishman_evals_K{K:.1f}_Nsfijo_Nmin{min(Ns):.0f}_Nmax{max(Ns):.0f}' + ".dat", header=None, index=None, sep=' ', mode='a')
#%% O1 vs t para varios Ns

Ns = [i for i in range(1000,5100,500)]+[10000]
time_lim = 51
Kpaso = 0
Ks = [17]#np.arange(0,20,Kpaso)

lista_df = []
for ki in tqdm(range(len(Ks)), desc='K loop'):
    K = Ks[ki]
    for ni in tqdm(range(len(Ns)), desc='N loop'):
        N = Ns[ni]
        filename = f'4pC_FFT_with_Tinf_state_Kmin{min(Ks)}_Kmax{max(Ks)}_Kpaso{Kpaso}_basis_size{N}_time_lim{time_lim}AX_BP'
        archives = np.load(filename+'.npz')
        O1_Ks = archives['O1']
        o1 = np.abs(O1_Ks[0])/N*4
        # times = np.arange(0,len(o1))
        for ti in range(len(o1)):
            lista_df.append([ti,o1[ti],K,N])#
df = pd.DataFrame(lista_df,columns=['t', 'O1/N*4', 'K', 'N'])#
df.to_csv(f'df_O1_vs_t_with_Tinf_state_K6_basis_size{N}_time_lim{time_lim}AX_BP_N' + ".dat", header=None, index=None, sep=' ', mode='a')
