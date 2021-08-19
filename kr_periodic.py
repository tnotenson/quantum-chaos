# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:29:46 2013

@author: delande
"""
# Copyright Dominique Delande
# Provided "as is" without any warranty
#
# Computes the quantum dynamics of the periodically kicked rotor
# by iteration of the evolution operator over one period
# The latter is computed using a "split" technique
# where the kick operator is computed in configuration space (where it is diagonal)
# and the free propagation is computed in momentum space (where it is diagonal)
# Passing from configuration to momentum representation is done a Fast Fourier Transform
# For this reason, it is more efficient to use a basis size which is a power of 2
# The initial state is chosen randomly near zero momentum, in a band defined
# by the half_width_initial_state parameter
# Averaging can be performed over the initial state and (optionally) the quasi-momentum
# The expectation value <p**2> is printed at each kick
# Also the (average) density in momentum space is printed at the final time 

from math import *
import time
import socket
import random
import os
import numpy 
from scipy import fftpack
import sys
import matplotlib.pyplot as plt

def compute_p2(wave_function,tab_momentum):
#  n=wave_function.size
#  p2=0.0  
#  norm=0.0
#  for i in range(n):
#    p2+=((i+quasi_momentum-n/2)*abs(wave_function[i]))**2
#    norm+=abs(wave_function[i])**2
  p2=numpy.sum((abs(wave_function)*tab_momentum)**2)
  norm=numpy.sum(abs(wave_function)**2)
#  norm=1.0
#  print norm,norm2 
  return p2/norm

# for generating reproducible random sequences
# comment out for irreproducible sequences  
random.seed(123)  
dpi=2.0*pi
K=11.6
# Execution time: 16 seconds for the following set
#hbar=0.5
#number_of_initial_states=50
#basis_size=2048
#final_time=800
# Execution time: 10 seconds for the following set with 100 initial states
hbar=1.0
number_of_initial_states=100
basis_size=512
final_time=400
# The following corresponds to M. Raizen's experiment
#hbar=2.0
#number_of_initial_states=500
#basis_size=128
#final_time=50

#
n=basis_size
# put half_width_initial_state to 0 for no averaging over initial state
half_width_initial_state=2
fixed_quasi_momentum=False
quasi_momentum_0=0.0
# Put random_kicked_rotor to True to generate the dynamics of the random kicked rotor
random_kicked_rotor=False
#hbar=2.89

K_over_hbar=K/hbar
dpi_over_n=dpi/n
tab_p2=numpy.zeros(final_time+1)
tab_density=numpy.zeros(n)
tab_momentum=numpy.zeros(n)
tab_spatial_phase=numpy.zeros(n)
tab_local_phase=numpy.zeros(n)
tab_kick_phase_factor=numpy.zeros(n,dtype=complex)
wave_function=numpy.zeros(basis_size,dtype=complex)
tab_free_prop_phase_factor=numpy.zeros(basis_size,dtype=complex)
for i in range(n):
  tab_spatial_phase[i]=cos(i*dpi_over_n)
  tab_kick_phase_factor[i]=complex(cos(K_over_hbar*tab_spatial_phase[i]),-sin(K_over_hbar*tab_spatial_phase[i]))
for i_state in range(number_of_initial_states):
# prepare initial state
# choose the quasi-momentum and populate only few momentum states near p=0 with random phases
  quasi_momentum=random.uniform(-0.5,0.5)
  if (fixed_quasi_momentum):
    quasi_momentum=quasi_momentum_0
  else:
    quasi_momentum=random.uniform(-0.5,0.5)
#  print 'Now starts initial state',i_state,quasi_momentum  
  for i in range(n):
    tab_momentum[i]=i+quasi_momentum-n/2 
    if (random_kicked_rotor):
# This is for the random kicked rotor      
      phase=random.uniform(0.0,dpi)
    else:  
# and this is for the regular kicked rotor      
      phase=0.5*hbar*(i+quasi_momentum-n/2)**2
    tab_free_prop_phase_factor[i]=complex(cos(phase),-sin(phase))
  wave_function[0:n]=0.0
  for i in range(int(n/2-half_width_initial_state),int(n/2+half_width_initial_state+1)):
    phase=random.uniform(0,dpi)
    wave_function[i]=complex(cos(phase),sin(phase))/sqrt(2*half_width_initial_state+1)
#  tab_p2[0]+=compute_p2(wave_function,tab_momentum)
  tab_p2[0]+=numpy.sum((numpy.real(wave_function)*tab_momentum)**2+(numpy.imag(wave_function)*tab_momentum)**2)
# now start time propagation
  for my_time in range(1,final_time+1):  
# first free propagation
    wave_function*=tab_free_prop_phase_factor
# then apply the kick
# this requires first to convert to configuration space by a FFT
    wave_function=fftpack.ifft(wave_function,overwrite_x=True)
# then multiply by the kick operator
    wave_function*=tab_kick_phase_factor   
# back in momentum space by a FFT      
    wave_function=fftpack.fft(wave_function,overwrite_x=True)
#    tab_p2[my_time]+=compute_p2(wave_function,tab_momentum)
    tab_p2[my_time]+=numpy.sum((numpy.real(wave_function)*tab_momentum)**2+(numpy.imag(wave_function)*tab_momentum)**2)
# save the final (squared) wavefunction 
  tab_density+=numpy.real(wave_function)**2+numpy.imag(wave_function)**2
# output the final result
orig_stdout = sys.stdout
p2_file=open('p2_vs_time_quantum.dat','w')
sys.stdout = p2_file
print ('# Data generated on',time.asctime(),' using Python script',os.path.abspath( __file__ ),' on computer:',socket.gethostname())
if (random_kicked_rotor):   
  print ('# Random kicked rotor')
else:
  print ('# Regular kicked rotor')
if (fixed_quasi_momentum):
  print ('# Fixed quasi-momentum =',quasi_momentum_0)
else:
  print ('# With averaging over quasi-momentum')
print ('# K =',K)
print ('# number_of_initial_states =',number_of_initial_states)
print ('# half_width_initial_state =',half_width_initial_state)
print ('# final_time =',final_time)
print ('# basis_size =',n)
for my_time in range(final_time+1):  
  print (my_time,hbar**2*tab_p2[my_time]/number_of_initial_states)
sys.stdout = orig_stdout
p2_file.close()

orig_stdout = sys.stdout
density_file=open('density_momentum.dat','w')
sys.stdout = density_file

print ('# Data generated on',time.asctime(),' using Python script',os.path.abspath( __file__ ),' on computer:',socket.gethostname())
if (random_kicked_rotor):   
  print ('# Random kicked rotor')
else:
  print ('# Regular kicked rotor')
if (fixed_quasi_momentum):
  print ('# Fixed quasi-momentum =',quasi_momentum_0)
else:
  print ('# With averaging over quasi-momentum')
print ('# K =',K)
print ('# number_of_initial_states =',number_of_initial_states)
print ('# half_width_initial_state =',half_width_initial_state)
print ('# final_time =',final_time)
print ('# basis_size =',n)
for i in range(n):  
  print (i-n/2,tab_density[i]/number_of_initial_states)
sys.stdout = orig_stdout
density_file.close()
#%% plot <p**2>
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

momentum_data = numpy.genfromtxt('p2_vs_time_quantum.dat')
density_data = numpy.genfromtxt('density_momentum.dat')

fig, ax = plt.subplots(2)

ax[0].plot(momentum_data[:,0], momentum_data[:,1])
ax[0].set_xlabel(r'Time $t$')
ax[0].set_ylabel(r'Kinetic Energy Average $\langle p^2 \rangle$')
ax[0].grid()

# plot |psi(p)|^2


ax[1].plot(density_data[:,0], density_data[:,1])
ax[1].set_xlabel(r'Momentum $p$')
ax[1].set_ylabel(r'Density $|\psi(p)|^2$')
ax[1].grid()