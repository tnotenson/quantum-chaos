# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:29:46 2013

@author: delande
"""
# Copyright Dominique Delande
# Provided "as is" without any warranty
#
# Computes the Floquet eigenstates of the periodically kicked rotor
# by diagonalization of the evolution operator over one period
# The latter is computed directly in momentum space, using Bessel functions
# (hopefully with the correct phase factors)
# It could also be computed using a "split" technique
# where the kick operator is computed in configuration space (where it is diagonal)
# and the free propagation is computed in momentum space (where it is diagonal)
# Passing from configuration to momentum representation is done a Fast Fourier Transform

# Because of truncation of the momentum eigenbasis, the matrix is not exactly unitary
# and some eigenvalues are not on the unit circle, but inside (worse for large K/hbar)
# This script prints the spectrum of the evolution operator truncated at -basis_size/2 and basis_size/2-1
# thus basis_size eigenvalues.
# It also prints few eigenstates (more precisely |psi[p)|**2) in the momentum basis
# States are chosen completely randomly and can thus be "unconverged" (eigenvalue of modulus
# smaller than unity) if they touch one of the boundaries.
# Execution time is dominated by the diagonalization itself, it scales as basis_size**3.
# It is roughly 10 seconds for basis_size=512
 
from math import *
import time
import socket
import random
import os
import numpy
from scipy import special
from scipy import linalg
import sys

# for generating reproducible random sequences
# comment out for irreproducible sequences  
random.seed(123)  
dpi=2.0*pi
# Execution time: 13 seconds for the following set
K=11.6
hbar=1.0
basis_size=512
n=basis_size
# Avoid quasi_momentum=0 because of discrete symmetry p <-> -p
quasi_momentum=0.1
number_of_printed_eigenstates=4
# Put random_kicked_rotor to True to generate the dynamics of the random kicked rotor
random_kicked_rotor=False

K_over_hbar=K/hbar
dpi_over_n=dpi/n
tab_u=numpy.zeros((n,n),dtype=complex)
tab_bessel=numpy.zeros(2*n)
for i in range(2*n):
  tab_bessel[i]=special.jn(i-n,K_over_hbar)
#print tab_bessel  
for i in range(n):
  if (random_kicked_rotor):
# This is for the random kicked rotor      
    phase=random.uniform(0.0,dpi)
  else:  
# and this is for the regular kicked rotor      
    phase=0.5*hbar*(i+quasi_momentum-n/2)**2-0.5*pi*i
  complex_phase_factor=complex(cos(phase),-sin(phase))
  for j in range(n):
    tab_u[j,i]=complex_phase_factor*tab_bessel[j+n-i]
    complex_phase_factor*=1j 
(w,vr)=linalg.eig(tab_u)
#print w

# output the final result
orig_stdout = sys.stdout
eig_file=open('Floquet_eigenvalues.dat','w')
sys.stdout = eig_file
print ('# Data generated on',time.asctime(),' using Python script',os.path.abspath( __file__ ),' on computer:',socket.gethostname())
print ('# K =',K)
print ('# hbar =',hbar)
print ('# basis_size =',n)
print ('# quasi_momentum =',quasi_momentum)
if (random_kicked_rotor):   
  print ('# Random kicked rotor')
else:
  print ('# Regular kicked rotor')
for i in range(n):
  print (w[i].real,w[i].imag)
sys.stdout = orig_stdout
eig_file.close()

# output few eigenstates
orig_stdout = sys.stdout
floquet_file=open('Floquet_eigenstates.dat','w')
sys.stdout = floquet_file
print ('# Data generated on',time.asctime(),' using Python script',os.path.abspath( __file__ ),' on computer:',socket.gethostname())
print ('# K =',K)
print ('# hbar =',hbar)
print ('# basis_size =',n)
print ('# quasi_momentum =',quasi_momentum)
if (random_kicked_rotor):   
  print ('# Random kicked rotor')
else:
  print ('# Regular kicked rotor')
print ('# number_of_printed_eigenstates =',number_of_printed_eigenstates)
for j in range(number_of_printed_eigenstates):
  j_plot=int(n*random.uniform(0.,1.))
  print ('# State',j_plot)
  for i in range(n):
    print (i-n/2,abs(vr[i,j_plot])**2)
  print (' ')
  sys.stdout = orig_stdout
floquet_file.close()
#%%

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

momentum_data = numpy.genfromtxt('Floquet_eigenvalues.dat')
density_data = numpy.genfromtxt('Floquet_eigenstates.dat')

fig, ax = plt.subplots(2)

ax[0].plot(momentum_data[:,0], momentum_data[:,1], '.')
ax[0].plot(momentum_data[25,0], momentum_data[25,1], 'r.')
ax[0].set_xlabel(r'$\mathtt{Re}(\lambda)$')
ax[0].set_ylabel(r'$\mathtt{Im}(\lambda)$')
ax[0].grid()

# plot |psi(p)|^2


ax[1].plot(density_data[:,0], density_data[:,1], 'r-', label='n=26')
ax[1].set_xlabel(r'Momentum $p$')
ax[1].set_ylabel(r'Floquet Density $|\psi(p)|^2$')
ax[1].legend()
ax[1].grid()
