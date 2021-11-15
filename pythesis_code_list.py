import numpy as np
import scipy
from scipy.linalg import blas as FB
import math
from scipy.optimize import curve_fit
import timeit
import matplotlib.pyplot as plt
from scipy.special import erfc
import scipy.signal

#INDEX - Use ctrl+F to browse faster
#(1) - BINARY OPERATIONS
#(2) - SPIN OPERATIONS
##(2.1) - SPIN SITE OPERATIONS
##(2.2) - CHAIN MODELS
##(2.3) - SYMMETRIES
###(2.3.1) - PARITY
###(2.3.2) = S_z CONSERVATION
#(3) - OUT-OF-TIME-ORDERED CORRELATORS
##(3.1) - WITH TEMPERATURE CHOICES (IF T=INFTY USE INFTY TEMPERATURE OTOCS FOR EXTRA SPEED)
##(3.2) - INFTY TEMPERATURE OTOCS
#(4) - CHAOS OPERATIONS
#(5) - MISCELLANEA CODE

#----------------  (1) BINARY OPERATIONS  ----------------#
#Translations from fortran --> python

#btest (https://gnu.huihoo.org/gcc/gcc-7.1.0/gfortran/BTEST.html)
def btest(i, pos): 
    return bool(i & (1 << pos))

#ibclr/ibset (https://gcc.gnu.org/onlinedocs/gcc-4.6.1/gfortran/IBCLR.html / http://www.lahey.com/docs/lfpro78help/F95ARIBSETFn.htm)
def set_bit(v, index, x): #if x=0 --> ibclr / if x=1 --> ibset 
  """Set the index:th bit of v to 1 if x is truthy, else to 0, and return the new value."""
  mask = 1 << index   
  v &= ~mask
  if x:
    v |= mask
  return v

#ibits
def ibits(num,pos,lens):
    binary = bin(num) # convert number into binary first 
    binary = binary[2:][::-1] # remove first two characters
    for n in range(pos+2):
        binary +=str(0)
        print(binary)
    kBitSubStr = binary[pos:pos+lens][::-1]
    print(kBitSubStr)
    return (int(kBitSubStr,2)) 

def mvbits(frm,frompos,leng,to,topos):
    binary = bin(frm)[2:][::-1]
    binary2 = bin(to)[2:][::-1]
    lentot = frompos+topos+leng+max(len(binary),len(binary2))
    for m in range(lentot):
        binary += str(0)
        binary2 += str(0)
    zz = list(binary2)
    zz2 = list(binary)	 
    zz[topos:topos+leng] = zz2[frompos:frompos+leng]
    return int(''.join(zz[::-1]),2)

#----------------  (2) SPIN OPERATIONS  ----------------#

##(2.1) SPIN SITE OPERATIONS

# Pauli at site operators
# Pauli at site operator (x,i) - The opt version is faster than the non opt version if sites>10
def S_xi_opt(pos_i,sites):
    dim = 2**sites
    S = np.zeros((dim,dim),dtype=complex)
    for i in range(dim):
        t1 = (i)^(set_bit(0,pos_i,1))
        S[i,t1]+=1
    return S

def S_xi(pos_i,sites):
    dim = 2**sites
    S = np.zeros((dim,dim),dtype=complex)
    estados2 = np.zeros(dim,dtype=np.int64)
    for i in range(dim):
        if btest(i,pos_i) == True:
            estados2[i] = set_bit(i,pos_i,0)
        else:
            estados2[i] = set_bit(i,pos_i,1)
    for i in range(dim):
        for j in range(dim):
            if i == estados2[j]:
                S[i,j] = S[i,j]+1
    return S

# Pauli at site operator (y,i) - The opt version is faster than the non opt version if sites>10
def S_yi_opt(pos_i,sites):
    dim = 2**sites
    S = np.zeros((dim,dim),dtype=complex)
    for i in range(dim):
        t1 = (i)^(set_bit(0,pos_i,1))
        S[i,t1]+= -1j*((-1)**(ibits(i,pos_i,1)))
    return S

def S_yi(pos_i,sites):
    dim = 2**sites
    S = np.zeros((dim,dim),dtype=complex)
    estados2 = np.zeros(dim,dtype=np.int64)
    a = np.zeros(dim,dtype=complex)
    for i in range(dim):
        if btest(i,pos_i) == True:
            estados2[i] = set_bit(i,pos_i,0)
            a[i] = 1j
        else:
            estados2[i] = set_bit(i,pos_i,1)
            a[i] = -1j
    for i in range(dim):
        for j in range(dim):
            if i == estados2[j]:
                S[i,j] = S[i,j]+a[i]
    return S

# Pauli at site operator (z,i) - The opt version is faster than the non opt version if sites>10
def S_zi_opt(pos_i,sites):
    dim = 2**sites
    S = np.zeros((dim,dim),dtype=complex)
    for i in range(dim):
        t1 = (i)^(set_bit(0,pos_i,1))
        S[i,i]+= (-1)**(ibits(i,pos_i,1))
    return S

def S_zi(pos_i,sites):
    dim = 2**sites
    S = np.zeros((dim,dim),dtype=complex)
    for i in range(dim):
        if btest(i,pos_i) == False:
            S[i,i] = 1
        else:
            S[i,i] = -1
    return S

# Neighbor i-j site interactions
# Neighbor interaction (x,i)(x,j) - The opt version is faster if
def S_xxij_opt(pos_i,pos_j,sites):  
    dim = 2**sites
    Sx = np.zeros((dim,dim),dtype=complex)
    for i in range(dim):
        if (pos_i<=sites-1-np.abs(pos_j-pos_i)):
            t1 = (i)^(set_bit(0,pos_i,1))
            t1 = t1^(set_bit(0,(pos_i+np.abs(pos_j-pos_i) % sites), 1))
            Sx[i,t1]+= 1
    return Sx

def S_xxij(pos_i,pos_j,sites):
    dim = 2**sites
    S = np.zeros((dim,dim),dtype=complex)
    estados2 = np.zeros(dim,dtype=np.int64)
    for i in range(dim):
        if btest(i,pos_i) == True:
            estados2[i] = set_bit(i,pos_i,0)
        else:
            estados2[i] = set_bit(i,pos_i,1)
    for i in range(dim):
        if btest(estados2[i],pos_j) == True:
            estados2[i] = set_bit(estados2[i],pos_j,0)
        else:
            estados2[i] = set_bit(estados2[i],pos_j,1)
    for i in range(dim):
        for j in range(dim):
            if i == estados2[j]:
                S[i,j] = S[i,j]+1
    return S

# Neighbor interaction (y,i)(y,j) - The opt version is faster if
def S_yyij_opt(pos_i,pos_j,sites):  
    dim = 2**sites
    S = np.zeros((dim,dim),dtype=complex)
    for i in range(dim):
        if (pos_i<=sites-1-np.abs(pos_j-pos_i)):
            t1 = (i)^(set_bit(0,pos_i,1))
            t1 = t1^(set_bit(0,(pos_i+np.abs(pos_j-pos_i) % sites), 1))
            S[i,t1]+= -1*((-1)**(ibits(i,pos_i,1) + ibits(i,pos_i+np.abs(pos_j-pos_i) % sites,1)))
    return S

def S_yyij(pos_i,pos_j,sites):
    dim = 2**sites
    S = np.zeros((dim,dim),dtype=complex)
    estados2 = np.zeros(dim,dtype=np.int64)
    a = np.zeros(dim,dtype=complex)
    for i in range(dim):
        if btest(i,pos_i) == True:
            estados2[i] = set_bit(i,pos_i,0)
            a[i] = 1j
        else:
            estados2[i] = set_bit(i,pos_i,1)
            a[i] = -1j
    #estados2 = estados2.astype(np.int64)
    for i in range(dim):
        if btest(estados2[i],pos_j) == True:
            estados2[i] = set_bit(estados2[i],pos_j,0)
            a[i] = a[i]*1j
        else:
            estados2[i] = set_bit(estados2[i],pos_j,1)
            a[i] = -a[i]*1j
    for i in range(dim):
        for j in range(dim):
            if i == estados2[j]:
                S[i,j] = S[i,j]+a[i]
    return S

# Neighbor interactino (z,i)(z,j) - The opt version is faster if
def S_zzij_opt(pos_i,pos_j,sites):  
    dim = 2**sites
    S = np.zeros((dim,dim),dtype=complex)
    for i in range(dim):
         if (pos_i<=sites-1-np.abs(pos_j-pos_i)):
            S[i,i]+= (-1)**(ibits(i,pos_i,1) + ibits(i,pos_i+np.abs(pos_j-pos_i) % sites,1))
    return S

def S_zzij(pos_i,pos_j,sites):
    dim = 2**sites
    S = np.zeros((dim,dim),dtype=complex)
    for i in range(dim):
        if btest(i,pos_i) == True:
            if btest(i,pos_j) == True:
                S[i,i] = 1
            else:
                S[i,i] = -1
        else:
            if btest(i,pos_j) == True:
                S[i,i] = -1
            else:
                S[i,i] = 1
    return S
    
# Entire spin direction operations

def S_x(sites): # Entire S_x operator
    """ Defines the operator S_x = sum S_x_i where i is the position between (0,N-1) and N = number of sites."""
    dimension = 2 ** sites 
    H = np.zeros((dimension,dimension),dtype=complex)
    for l in range(1,dimension+1):
        for j in range(0,sites):
            if btest(l-1,j) == True:
                hh = set_bit(l-1,j,0) + 1
            if btest(l-1,j) == False:
                hh = set_bit(l-1,j,1) + 1
            H[l-1,hh-1]=H[l-1,hh-1]+1
    return H

def S_y(sites): # Entire S_y operator
    """ Defines the operator S_y = sum S_y_i where i is the position between (0,N-1) and N = number of sites."""
    dimension = 2 ** sites 
    H = np.zeros((dimension,dimension),dtype=complex)
    for l in range(1,dimension+1):
        for u in range(0,sites):
            if btest(l-1,u) == True:
                hh = set_bit(l-1,u,0) + 1
                H[l-1,hh-1]=H[l-1,hh-1] + 1j
            if btest(l-1,u) == False:
                hh = set_bit(l-1,u,1) + 1
                H[l-1,hh-1]=H[l-1,hh-1] - 1j
    return H

def S_z(sites): # Entire S_z operator
    """ Defines the operator S_z = sum S_z_i where i is the position between (0,N-1) and N = number of sites."""
    dimension = 2 ** sites 
    H = np.zeros((dimension,dimension),dtype=complex)
    for l in range(1,dimension+1):
        for j in range(0,sites):
            if btest(l-1,j) == True:
                hh = set_bit(l-1,j,1) + 1
                H[l-1,hh-1]=H[l-1,hh-1] - 1
            if btest(l-1,j) == False:
                hh = set_bit(l-1,j,0) + 1
                H[l-1,hh-1]=H[l-1,hh-1] + 1
    return H

# Neighbour interactionss xx,yy,zz all in one for speed

def spin_interactions(sites,neig,BC,Cxx,Cyy,Czz):  
    dim = 2**sites
    Sx = np.zeros((dim,dim),dtype=complex)
    Sy = np.zeros((dim,dim),dtype=complex)
    Sz = np.zeros((dim,dim),dtype=complex)
    t1 = 0
    kk = 0
    if (type(Cxx) == float) | (type(Cxx) == int):
        Cx = np.zeros(dim,dtype=float)+Cxx
    else:
        Cx = Cxx
    if (type(Cyy) == float) | (type(Cyy) == int):
        Cy = np.zeros(dim,dtype=float)+Cyy
    else:
        Cy = Cyy
    if (type(Czz) == float) | (type(Czz) == int):
        Cz = np.zeros(dim,dtype=float)+Czz
    else:
        Cz = Czz
    for i in range(dim):
        for n in range(sites-1):
            if ((n<=sites-1-neig) | (BC=="perodic")):
                kk = ibits(i,n,1) + ibits(i,n+neig % sites,1)
                t1 = (i)^(set_bit(0,n,1))
                t1 = t1^(set_bit(0, n+neig % sites, 1))
		#print(t1)
                Sy[i,t1]+= -Cy[i] * (-1)**(kk)
                Sx[i,t1]+= Cx[i]
                Sz[i,i]+= Cz[i] * (-1)**(kk)
    return Sx + Sy + Sz

##(2.3) SYMMETRIES

###(2.3.1) PARITY

# "EVEN" and "ODD" for either desired subspace. If "BOTH" (or anything else) returns the basis and energies ordered from even to odd parity and smaller to higher energies
def parity_subspace(sites,basis,ener,pariti):
    dim = 2**sites
    basis2 = np.zeros((dim,dim),dtype=complex)
    if sites % 2 == 0:
        rangoz = int(sites/2)
    else:
        rangoz = int((sites-1)/2)
    #print(rangoz)
    for i in range(dim):
        q = i
        p = i
        for j in range(rangoz):
            q = mvbits(p,j,1,q,sites-1-j)
            #print(q)
            q = mvbits(p,sites-1-j,1,q,j)
            #print(q)
        basis2[i,:] = basis[q,:]
    dimparity = -1
    enerparity = []
    basisparity = np.zeros((dim,dim),dtype=complex)
    if pariti == "EVEN":
        for i in range(dim):
            #print(np.vdot(basis[:,i],basis2[:,i]).real)
            if np.vdot(basis[:,i],basis2[:,i]).real > 0:
                dimparity += 1
                enerparity.append(ener[i])
                basisparity[:,dimparity]=basis[:,i]
    if pariti == "ODD":
        for i in range(dim):
            #print(np.vdot(basis[:,i],basis2[:,i]).real)
            if np.vdot(basis[:,i],basis2[:,i]).real < 0:
                dimparity += 1
                enerparity.append(ener[i])
                basisparity[:,dimparity]=basis[:,i]
    if pariti == "BOTH":
        for i in range(dim):
            #print(np.vdot(basis[:,i],basis2[:,i]).real)
            if np.vdot(basis[:,i],basis2[:,i]).real > 0:
                dimparity += 1
                enerparity.append(ener[i])
                basisparity[:,dimparity]=basis[:,i]
        for i in range(dim):
            #print(np.vdot(basis[:,i],basis2[:,i]).real)
            if np.vdot(basis[:,i],basis2[:,i]).real < 0:
                dimparity += 1
                enerparity.append(ener[i])
                basisparity[:,dimparity]=basis[:,i]
    return enerparity,basisparity[:,0:dimparity+1]

# Reorders the eigenkets and eigenstates in the first even block and second odd block	
def parity_subspace_both(sites,basis,ener):
    dim = len(ener)
    basis2 = np.zeros((dim,dim),dtype=complex)
    if sites % 2 == 0:
        rangoz = int(sites/2)
    else:
        rangoz = int((sites-1)/2)
    #print(rangoz)
    for i in range(dim):
        q = i
        p = i
        for j in range(rangoz):
            q = mvbits(p,j,1,q,sites-1-j)
            q = mvbits(p,sites-1-j,1,q,j)
        basis2[i,:] = basis[q,:]
    dimparity = -1
    enerparity = []
    basisparity = np.zeros((dim,dim),dtype=complex)
    for i in range(dim):
        if np.vdot(basis[:,i],basis2[:,i]).real > 0:
            dimparity += 1
            enerparity.append(ener[i])
            basisparity[:,dimparity]=basis[:,i]
    dimeven = len(enerparity)
    for i in range(dim):
        if np.vdot(basis[:,i],basis2[:,i]).real < 0:
            dimparity += 1
            enerparity.append(ener[i])
            basisparity[:,dimparity]=basis[:,i]
    return enerparity,basisparity[:,0:dimparity+1],dimeven


	
# Operator 1/2 (S_zi+S_z(sites-i)) which commutes with parity!
def S_zi_parity(pos_i,sites,basisparity):
    dim = 2**sites
    dim2 = min(basisparity.shape)
    S = np.zeros((dim2,dim2),dtype=complex)
    if (sites % 2 != 0 and pos_i == int(sites/2 - 0.5) ):
        for i in range(dim2):
            for j in range(dim2):
                for m in range(dim):
                    S[i,j] += basisparity[m,i]*basisparity[m,j]* ((-1)**(ibits(m,pos_i,1))) 
    else:
        for i in range(dim2):
            for j in range(dim2):
                for m in range(dim):
                    S[i,j] += 0.5*basisparity[m,i]*basisparity[m,j]*  ( ((-1)**(ibits(m,pos_i,1))) + ((-1)**(ibits(m,sites-1-pos_i,1))) )    
    return S

###(2.3.2) S_z CONSERVATION

#S_z conservation spin up subspaces
def sz_subspace(sites,n_partic):
    label_state = 0
    dim = int(math.factorial(sites)/(math.factorial(sites-n_partic)*math.factorial(n_partic)))
    dim2 = 2**sites
    states = np.zeros(dim,dtype=int)
    flag = np.zeros(dim2)
    for i in range(dim2-1):
        k=0
        j=0
        while k<=n_partic and j<=sites-1:
            k+=ibits(i,j,1) # +=0 si i-j == impar y +=1 si i-j = par
            j+=1
            #print(k)
        #print("-----")
        if k==n_partic:
            #print("funco en k,j=",k,j)
            label_state+=1
            states[label_state-1] = i
            flag[i] = label_state
    return states, flag

def sz_subspace_parity_subspace_both(sites,n_partic,basis,ener):
    dim = len(ener)
    states, flag = sz_subspace(sites,n_partic)
    basis2 = np.zeros((dim,dim),dtype=complex)
    if sites % 2 == 0:
        rangoz = int(sites/2)
    else:
        rangoz = int((sites-1)/2)
    #print(rangoz)
    for i in range(dim):
        q = states[i]
        p = states[i]
        for j in range(rangoz):
            q = mvbits(p,j,1,q,sites-1-j)
            q = mvbits(p,sites-1-j,1,q,j)
        basis2[i,:] = basis[flag[q-1],:]
    dimparity = -1
    enerparity = []
    basisparity = np.zeros((dim,dim),dtype=complex)
    for i in range(dim):
        if np.vdot(basis[:,i],basis2[:,i]).real > 0:
            dimparity += 1
            enerparity.append(ener[i])
            basisparity[:,dimparity]=basis[:,i]
    dimeven = len(enerparity)
    for i in range(dim):
        if np.vdot(basis[:,i],basis2[:,i]).real < 0:
            dimparity += 1
            enerparity.append(ener[i])
            basisparity[:,dimparity]=basis[:,i]
    return enerparity,basisparity[:,0:dimparity+1],dimeven


#Neighbor (z,i)(z,j) interaction - The opt version is faster if
def sz_subspace_S_zi(pos_i,sites,n_partic):  
    dim = int(math.factorial(sites)/(math.factorial(sites-n_partic)*math.factorial(n_partic)))
    S = np.zeros((dim,dim),dtype=complex)
    states, flag = sz_subspace(sites,n_partic)
    for i in range(dim):
        if btest(states[i],pos_i) == False:
            S[i,i] += 1
        else:
            S[i,i] += -1
    return S

#S_z subspace neighbor i,j interactions
#Neighbor (x,i)(x,j) interaction - The opt version is faster if
def sz_subspace_S_xxij_opt(pos_i,pos_j,sites,n_partic):  
    dim = int(math.factorial(sites)/(math.factorial(sites-n_partic)*math.factorial(n_partic)))
    S = np.zeros((dim,dim),dtype=complex)
    states, flag = sz_subspace(sites,n_partic)
    for i in range(dim):
        if (pos_i<=sites-1-np.abs(pos_j-pos_i)):
            stepi = ibits(states[i],pos_i,1) + ibits(states[i], (pos_i+np.abs(pos_j-pos_i))%(sites),1)
            if stepi == 1:
                t1 = (states[i])^(set_bit(0,pos_i,1))
                print(t1)
                t1 = int(flag[(t1)^(set_bit(0, pos_i+np.abs(pos_j-pos_i) % sites, 1))] )
                S[i,t1-1]+= 1
    return S

def sz_subspace_S_xxij(pos_i,pos_j,sites,n_partic):
    dim = int(math.factorial(sites)/(math.factorial(sites-n_partic)*math.factorial(n_partic)))
    S = np.zeros((dim,dim),dtype=complex)
    states, flag = sz_subspace(sites,n_partic)
    estados2 = np.zeros(dim,dtype=int)
    for i in range(dim):
        if btest(states[i],pos_i) == True:
            estados2[i] = set_bit(states[i],pos_i,0)
        else:
            estados2[i] = set_bit(states[i],pos_i,1)
    for i in range(dim):
        if btest(estados2[i],pos_j) == True:
            estados2[i] = set_bit(estados2[i],pos_j,0)
        else:
            estados2[i] = set_bit(estados2[i],pos_j,1)
    for i in range(dim):
        for j in range(dim):
            if states[i] == estados2[j]:
                S[i,j] = S[i,j]+1
    return S

# Neighbor (y,i)(y,j) interaction - The opt version is faster if
def sz_subspace_S_yyij_opt(pos_i,pos_j,sites,n_partic):  
    dim = int(math.factorial(sites)/(math.factorial(sites-n_partic)*math.factorial(n_partic)))
    S = np.zeros((dim,dim),dtype=complex)
    states, flag = sz_subspace(sites,n_partic)
    for i in range(dim):
        if (pos_i<=sites-1-np.abs(pos_j-pos_i)):
            stepi = ibits(states[i],pos_i,1) + ibits(states[i], (pos_i+np.abs(pos_j-pos_i))%(sites),1)
            if stepi == 1:
                t1 = (states[i])^(set_bit(0,pos_i,1))
                print(t1)
                t1 = int(flag[(t1)^(set_bit(0, pos_i+np.abs(pos_j-pos_i) % sites, 1))] )
                S[i,t1-1]+= -((-1)**(ibits(states[i],pos_i,1) + ibits(states[i],pos_i+np.abs(pos_j-pos_i) % sites,1)))
    return S

def sz_subspace_S_yyij(pos_i,pos_j,sites,n_partic):
    dim = int(math.factorial(sites)/(math.factorial(sites-n_partic)*math.factorial(n_partic)))
    S = np.zeros((dim,dim),dtype=complex)
    states, flag = sz_subspace(sites,n_partic)
    estados2 = np.zeros(dim,dtype=int)
    a = np.zeros(dim,dtype=complex)
    for i in range(dim):
        if btest(states[i],pos_i) == True:
            estados2[i] = set_bit(states[i],pos_i,0)
            a[i] = 1j
        else:
            estados2[i] = set_bit(states[i],pos_i,1)
            a[i] = -1j
    for i in range(dim):
        if btest(estados2[i],pos_j) == True:
            estados2[i] = set_bit(estados2[i],pos_j,0)
            a[i] = a[i]*1j
        else:
            estados2[i] = set_bit(estados2[i],pos_j,1)
            a[i] = -a[i]*1j
    for i in range(dim):
        for j in range(dim):
            if states[i] == estados2[j]:
                S[i,j] = S[i,j] + a[i]
    return S

#Neighbor (z,i)(z,j) interaction - The opt version is faster if
def sz_subspace_S_zzij_opt(pos_i,pos_j,sites,n_partic):  
    dim = int(math.factorial(sites)/(math.factorial(sites-n_partic)*math.factorial(n_partic)))
    S = np.zeros((dim,dim),dtype=complex)
    states, flag = sz_subspace(sites,n_partic)
    for i in range(dim):
        if (pos_i<=sites-1-np.abs(pos_j-pos_i)):
            S[i,i]+= (-1)**(ibits(states[i],pos_i,1) + ibits(states[i],pos_i+np.abs(pos_j-pos_i) % sites,1))
    return S

def sz_subspace_S_zzij(pos_i,pos_j,sites,n_partic):
    dim = int(math.factorial(sites)/(math.factorial(sites-n_partic)*math.factorial(n_partic)))
    S = np.zeros((dim,dim),dtype=complex)
    states, flag = sz_subspace(sites,n_partic)
    for i in range(dim):
        if btest(states[i],pos_i) == True:
            if btest(states[i],pos_j) == True:            
                S[i,i]+=1
            else:
            	S[i,i]+=-1
        else:
            if btest(states[i],pos_j) == True:            
                S[i,i]+=-1
            else:
            	S[i,i]+=1
    return S

def sz_subspace_spin_interactions(sites,n_partic,neig,BC,Cxx,Cyy,Czz):
    states, flag = sz_subspace(sites,n_partic)    
    dim = int(math.factorial(sites)/(math.factorial(sites-n_partic)*math.factorial(n_partic)))
    Sx = np.zeros((dim,dim),dtype=complex)
    Sy = np.zeros((dim,dim),dtype=complex)
    Sz = np.zeros((dim,dim),dtype=complex)
    t1 = 0
    kk = 0
    if (type(Cxx) == float) | (type(Cxx) == int):
        Cx = np.zeros(dim,dtype=float)+Cxx
    else:
        Cx = Cxx
    if (type(Cyy) == float) | (type(Cyy) == int):
        Cy = np.zeros(dim,dtype=float)+Cyy
    else:
        Cy = Cyy
    if (type(Czz) == float) | (type(Czz) == int):
        Cz = np.zeros(dim,dtype=float)+Czz
    else:
        Cz = Czz
    #print(flag)
    for i in range(1,dim+1):
        for n in range(sites-1):
            if ((n<=sites-1-neig) | (BC=="perodic")):
                stepi = ibits(states[i-1],n,1) + ibits(states[i-1], (n+neig)%(sites),1)
                #print("stepi = ",stepi)
                if (stepi == 1):
                    kk = ibits(states[i-1],n,1) + ibits(states[i-1],n+neig % sites,1)
                    t1 = (states[i-1])^(set_bit(0,n,1))
                    #print(t1)
                    t1 = int(flag[(t1)^(set_bit(0, n+neig % sites, 1))] )
                    #print(t1)
                    Sy[i-1,t1-1]+= -Cy[i-1] * (-1)**(kk)
                    Sx[i-1,t1-1]+= Cx[i-1]
                Sz[i-1,i-1]+= Cz[i-1] * (-1)**(ibits(states[i-1],n,1) + ibits(states[i-1],n+neig % sites,1))
    return Sx + Sy + Sz

def Parity(sites): # Returns the parity operator in the S_z base #### HAY QUE REHACERLO!!!!!! ESTA MAL! (PERO NO MUY MAL)
    dim = 2 ** sites
    identity = np.zeros((dim,dim),dtype=complex)
    zeros = np.zeros((4,4),dtype=complex)
    for n in range(dim):
        identity[n,n] = 1
    Dx = np.matrix([[0,1],[1,0]],dtype=complex)
    Dy = np.matrix([[0,-1j],[1j,0]],dtype=complex)
    Dz = np.matrix([[1,0],[0,-1]],dtype=complex)
    Parity = identity
    if sites % 2 == 0:
        tt = sites/2
    else:
        tt = (sites-1)/2
    for u in range(tt):
        P = 0.5* (identity + np.matmul(S_x_i(u,sites),S_x_i(sites-1-u,sites)) + np.matmul(S_y_i(u,sites),S_y_i(sites-1-u,sites)) + np.matmul(S_z_i(u,sites),S_z_i(sites-1-u,sites)) )
        Parity = np.matmul(Parity,P)
    return Parity
    
#----------------  (3) OUT-OF-TIME-ORDERED CORRELATORS  ----------------#

### (3.1) WITH TEMPERATURE CHOICES (IF T=INFTY USE INFTY TEMPERATURE OTOCS FOR EXTRA SPEED)

#OTOC
def OTOC(V,W,ener,basis,N,dt,t0,beta,ortho):
    dim = len(ener)
    if ortho == True:
        basist = np.transpose(basis)
    else:
        basist = np.linalg.inv(basis)
    ope0 = np.matmul(basist,V)
    ope0 = np.matmul(ope0,basis)
    ope = np.matmul(basist,W)
    ope = np.matmul(ope,basis)
    otoc=[]
    mm = np.zeros((dim,dim),dtype=complex)
    mm1 = np.zeros((dim,dim),dtype=complex)
    ensemblemedio = dim
    ensemble = np.identity(dim,dtype=complex)
    if beta != 0:
        for i in range(0,dim):
            ensemble[i][i] = np.exp(-beta*ener[i])
        ensemblemedio = np.abs(np.matrix.trace(ensemble))
    U = np.zeros((dim,dim),dtype=complex) # U evolucion temporal
    Udagger = np.zeros((dim,dim),dtype=complex) # U*   
    for count0 in range(0,N):
        tie = count0*dt+t0
        for count1 in range(0,dim):
            U[count1][count1] = np.exp(-1j*tie*ener[count1])
            Udagger[count1][count1] = np.exp(1j*tie*ener[count1])
        mm = np.matmul(ope0,U)
        mm1= np.matmul(Udagger,mm)
        mm = np.matmul(mm1,ope) - np.matmul(ope,mm1)
        mm = np.matmul(np.transpose(np.conjugate(mm)),mm)
        mm = np.matmul(ensemble,mm)
        otoc.append(0.5*np.abs(np.matrix.trace(mm)) / ensemblemedio )
    return otoc

# OTOC 1 - Re(F) - USE ONLY IF V,W ARE BOTH UNITARY & HERMITIAN
def OTOCF(V,W,ener,basis,N,dt,t0,beta,ortho):
    dim = len(ener)
    if ortho == True:
        basist = np.transpose(basis)
    else:
        basist = np.linalg.inv(basis)
    ope0 = np.matmul(basist,V)
    ope0 = np.matmul(ope0,basis)
    ope = np.matmul(basist,W)
    ope = np.matmul(ope,basis)
    otoc=[]
    mm = np.zeros((dim,dim),dtype=complex)
    mm1 = np.zeros((dim,dim),dtype=complex)
    ensemblemedio = dim
    ensemble = np.identity(dim,dtype=complex)
    if beta != 0:
        for i in range(0,dim):
            ensemble[i][i] = np.exp(-beta*ener[i])
        ensemblemedio = np.abs(np.matrix.trace(ensemble))
    U = np.zeros((dim,dim),dtype=complex) # U evolucion temporal
    Udagger = np.zeros((dim,dim),dtype=complex) # U*   
    for count0 in range(0,N):
        tie = count0*dt+t0
        for count1 in range(0,dim):
            U[count1][count1] = np.exp(-1j*tie*ener[count1])
            Udagger[count1][count1] = np.exp(1j*tie*ener[count1])
        mm = np.matmul(ope0,U)
        mm1= np.matmul(Udagger,mm)
        mm = np.matmul(mm1,ope)
        mm = np.matmul(mm,mm)
        mm = np.matmul(ensemble,mm)
        otoc.append(1 - (np.matrix.trace(mm)/ensemblemedio).real )
    return otoc

### (3.2) INFTY TEMPERATURE OTOCS

# OTOC OPTMIZED SPEED
def OTOC_opt_infty(V,W,ener,basis,N,dt,t0,ortho):
    basis = np.complex64(basis, order = 'F')
    V = np.complex64(V, order = 'F')
    W = np.complex64(W, order = 'F')
    dim = len(ener)
    if ortho == True:
        basist = np.transpose(basis)
    else:
        basist = np.linalg.inv(basis)
    S0 = FB.cgemm(1,V,basis)
    S0 = FB.cgemm(1,basist,S0)
    S = FB.cgemm(1,W,basis)
    S = FB.cgemm(1,basist,S)
    mm = np.zeros((dim,dim),dtype="complex64",order='F')
    mm1 = np.zeros((dim,dim),dtype="complex64",order='F')
    otok = np.zeros(N,dtype = "float32")
    U = np.zeros((dim,dim),dtype="complex64",order='F') # U evolucion temporal
    Udagger = np.zeros((dim,dim),dtype="complex64",order='F') # U*
    tiempo = np.linspace(t0,t0+N*dt,N)
    for ti in range(0,N):
        for c1 in range(0,dim):
            U[c1][c1] = np.exp(-1j*tiempo[ti]*ener[c1])
            Udagger[c1][c1] = np.exp(1j*tiempo[ti]*ener[c1])
        mm = FB.cgemm(1,S0,U)
        mm1= FB.cgemm(1,Udagger,mm)
        mm = FB.cgemm(1,mm1,S) - FB.cgemm(1,S,mm1)
        mm = FB.cgemm(1,np.transpose(np.conjugate(mm)),mm)
        otok[ti] = 0.5*(np.abs(np.matrix.trace(mm))/dim)
    otok = np.array(otok)
    return otok
    
# OTOC 1-Re(F) OPTMIZED SPEED - USE ONLY IF V,W ARE BOTH UNITARY & HERMITIAN
def OTOCF_opt_infty(V,W,ener,basis,N,dt,t0,ortho):
    basis = np.complex64(basis, order = 'F')
    V = np.complex64(V, order = 'F')
    W = np.complex64(W, order = 'F')
    dim = len(ener)
    if ortho == True:
        basist = np.transpose(basis)
    else:
        basist = np.linalg.inv(basis)
    S0 = FB.cgemm(1,V,basis)
    S0 = FB.cgemm(1,basist,S0)
    S = FB.cgemm(1,W,basis)
    S = FB.cgemm(1,basist,S)
    mm = np.zeros((dim,dim),dtype="complex64",order='F')
    mm1 = np.zeros((dim,dim),dtype="complex64",order='F')
    otok = np.zeros(N,dtype = "float32")
    U = np.zeros((dim,dim),dtype="complex64",order='F') # U evolucion temporal
    Udagger = np.zeros((dim,dim),dtype="complex64",order='F') # U*
    tiempo = np.linspace(t0,t0+N*dt,N)
    for ti in range(0,N):
        for c1 in range(0,dim):
            U[c1][c1] = np.exp(-1j*tiempo[ti]*ener[c1])
            Udagger[c1][c1] = np.exp(1j*tiempo[ti]*ener[c1])
        mm = FB.cgemm(1,S0,U)
        mm1= FB.cgemm(1,Udagger,mm)
        mm = FB.cgemm(1,mm1,S)
        mm = FB.cgemm(1,mm,mm)
        otok[ti] = 1 - (np.matrix.trace(mm)/dim).real 
    otok = np.array(otok)
    return otok

def OTOCF_opt_infty_nobasischange(V,W,ener,N,dt,t0):
    S0 = np.complex64(V, order = 'F')
    S = np.complex64(W, order = 'F')
    dim = len(ener)
    mm = np.zeros((dim,dim),dtype="complex64",order='F')
    mm1 = np.zeros((dim,dim),dtype="complex64",order='F')
    otok = np.zeros(N,dtype = "float32")
    U = np.zeros((dim,dim),dtype="complex64",order='F') # U evolucion temporal
    Udagger = np.zeros((dim,dim),dtype="complex64",order='F') # U*
    tiempo = np.linspace(t0,t0+N*dt,N)
    for ti in range(0,N):
        for c1 in range(0,dim):
            U[c1][c1] = np.exp(-1j*tiempo[ti]*ener[c1])
            Udagger[c1][c1] = np.exp(1j*tiempo[ti]*ener[c1])
        mm = FB.cgemm(1,S0,U)
        mm1= FB.cgemm(1,Udagger,mm)
        mm = FB.cgemm(1,mm1,S)
        mm = FB.cgemm(1,mm,mm)
        otok[ti] = 1 - (np.matrix.trace(mm)/dim).real 
    otok = np.array(otok)
    return otok


# BOTH OTOCS 1 and 2 (Tr(S S0(t) S0(t) S)/D, -Re [Tr(S0(t) S S0(t) S)]/D)
def OTOCS_opt_infty(V,W,ener,basis,N,dt,t0,ortho):
    basis = np.complex64(basis, order = 'F')
    V = np.complex64(V, order = 'F')
    W = np.complex64(W, order = 'F')
    dim = len(ener)
    if ortho == True:
        basist = np.transpose(basis)
    else:
        basist = np.linalg.inv(basis)
    S0 = FB.cgemm(1,V,basis)
    S0 = FB.cgemm(1,basist,S0)
    S = FB.cgemm(1,W,basis)
    S = FB.cgemm(1,basist,S)
    mm = np.zeros((dim,dim),dtype="complex64",order='F')
    mm1 = np.zeros((dim,dim),dtype="complex64",order='F')
    mm2 = np.zeros((dim,dim),dtype="complex64",order='F')
    otok1 = np.zeros(N,dtype = "float32")
    otok2 = np.zeros(N,dtype = "float32")
    tiempo = np.linspace(t0,N*dt+t0,N)
    U = np.zeros((dim,dim),dtype="complex64",order='F') # U evolucion temporal
    Udagger = np.zeros((dim,dim),dtype="complex64",order='F') # U*
    for ti in range(0,N):
        #start_time = timeit.default_timer()
        for c1 in range(0,dim):
            U[c1,c1] = np.exp(-1j*tiempo[ti]*ener[c1])
            Udagger[c1,c1] = np.exp(1j*tiempo[ti]*ener[c1])
        mm = FB.cgemm(1,S0,U)
        mm1= FB.cgemm(1,Udagger,mm) #S0(t)
        mm2 = FB.cgemm(1,mm1,mm1) # S0(t) S0(t)
        mm2 = FB.cgemm(1,mm2,S) # S0(t) S0(t) S
        mm2 = FB.cgemm(1,S,mm2) # S S0(t) S0(t) S
        mm = FB.cgemm(1,mm1,S) #S0(t) S
        mm = FB.cgemm(1,mm,mm) #S0(t) S S0(t) S
        otok1[ti] = -(np.matrix.trace(mm)/dim).real #otok1 = - Re [Tr(S0(t) S S0(t) S)]/D
        otok2[ti] = np.matrix.trace(mm2).real/dim #otok2 = Tr(S S0(t) S0(t) S)/D
        #elapsed = timeit.default_timer() - start_time
        #print(elapsed)   
    otok1 = np.array(otok1)
    otok2 = np.array(otok2)
    return otok1, otok2
    
# OTOC SLOW (IF HIGHER PRECISION REQUIRED)
def OTOC_infty(V,W,ener,basis,N,dt,t0,ortho):
    dim = len(ener)
    if ortho == True:
        basist = np.transpose(basis)
    else:
        basist = np.linalg.inv(basis)
    S0 = np.matmul(V,basis)
    S0 = np.matmul(basist,S0)
    S = np.matmul(W,basis)
    S = np.matmul(basist,S)
    mm = np.zeros((dim,dim),dtype=complex)
    mm1 = np.zeros((dim,dim),dtype=complex)
    otok = np.zeros(N,dtype = float)
    U = np.zeros((dim,dim),dtype=complex) # U evolucion temporal
    Udagger = np.zeros((dim,dim),dtype=complex) # U*
    tiempo = np.linspace(t0,N*dt,N)
    for ti in range(0,N):
        for c1 in range(0,dim):
            U[c1][c1] = np.exp(-1j*tiempo[ti]*ener[c1])
            Udagger[c1][c1] = np.exp(1j*tiempo[ti]*ener[c1])
        mm = np.matmul(S0,U) # S0 * U
        mm1= np.matmul(Udagger,mm) # Udaga * S0 * U = S0(t)
        mm = np.matmul(mm1,S) - np.matmul(S,mm1)
        mm = np.matmul(np.transpose(np.conjugate(mm)),mm)
        otok[ti] = 0.5*(np.abs(np.matrix.trace(mm))/dim)
    otok = np.array(otok)
    return otok

# OTOC 1-Re(F) SLOW (IF HIGHER PRECISION REQUIRED) - USE ONLY IF V,W ARE BOTH UNITARY & HERMITIAN
def OTOCF_infty(V,W,ener,basis,N,dt,t0,ortho):
    dim = len(ener)
    if ortho == True:
        basist = np.transpose(basis)
    else:
        basist = np.linalg.inv(basis)
    S0 = np.matmul(V,basis)
    S0 = np.matmul(basist,S0)
    S = np.matmul(W,basis)
    S = np.matmul(basist,S)
    mm = np.zeros((dim,dim),dtype=complex)
    mm1 = np.zeros((dim,dim),dtype=complex)
    otok = np.zeros(N,dtype = float)
    U = np.zeros((dim,dim),dtype=complex) # U evolucion temporal
    Udagger = np.zeros((dim,dim),dtype=complex) # U*
    for ti in range(0,N):
        for c1 in range(0,dim):
            U[c1][c1] = np.exp(-1j*tiempo[ti]*ener[c1])
            Udagger[c1][c1] = np.exp(1j*tiempo[ti]*ener[c1])
        mm = np.matmul(S0,U)
        mm1= np.matmul(Udagger,mm)
        mm = np.matmul(mm1,S)
        mm = np.matmul(mm,mm)
        otok[ti] = 1 - (np.matrix.trace(mm)/dim).real 
    otok = np.array(otok)
    return otok
    
#----------------  (4) 4 POINT OTOCS  ----------------#

def OTOC4F_infty(A,B,C,D,ener,basis,N,dt,t0,ortho):
    dim = len(ener)
    if ortho == True:
        basist = np.transpose(basis)
    else:
        basist = np.linalg.inv(basis)
    A = np.matmul(A,basis)
    A = np.matmul(basist,A)
    B = np.matmul(B,basis)
    B = np.matmul(basist,B)
    C = np.matmul(C,basis)
    C = np.matmul(basist,C)
    D = np.matmul(D,basis)
    D = np.matmul(basist,D)
    mmc = np.zeros((dim,dim),dtype=complex)
    mmd = np.zeros((dim,dim),dtype=complex)
    mm = np.zeros((dim,dim),dtype=complex)
    mm1 = np.zeros((dim,dim),dtype=complex)
    tiempo = np.linspace(t0,N*dt+t0,N)
    otok = np.zeros(N,dtype=complex)
    U = np.zeros((dim,dim),dtype=complex) # U evolucion temporal
    Udagger = np.zeros((dim,dim),dtype=complex) # U*
    for ti in range(0,N):
        for c1 in range(0,dim):
            U[c1][c1] = np.exp(-1j*tiempo[ti]*ener[c1])
            Udagger[c1][c1] = np.exp(1j*tiempo[ti]*ener[c1])
        mmc = np.matmul(C,U)
        mmc = np.matmul(Udagger,mmc) # C(t)
        mmd = np.matmul(D,U)
        mmd = np.matmul(Udagger,mmd) # D(t)
        mm = np.matmul(A,B) # AB
        mm1 = np.matmul(mmc,mmd) #C(t)D(t) 
        mm = np.matmul(mm,mm1)  #ABC(t)D(t)     
        otok[ti] = np.matrix.trace(mm)/dim #Tr[ ABC(t)D(t)]      
        mm = np.matmul(A,mmc) #AC(t)
        mm1 = np.matmul(B,mmd) #BD(t)
        mm = np.matmul(mm,mm1) #AC(t)BD(t)
        otok[ti] += - (np.matrix.trace(mm)/dim).real #realTr( AC(t)BD(t))
    otok = np.array(otok)
    return otok

def OTOC4F_opt_infty(A,B,C,D,ener,basis,N,dt,t0,ortho):
    basis = np.complex64(basis, order = 'F')
    A = np.complex64(A, order = 'F')
    B = np.complex64(B, order = 'F')
    C = np.complex64(C, order = 'F')
    D = np.complex64(D, order = 'F')
    dim = len(ener)
    if ortho == True:
        basist = np.transpose(basis)
    else:
        basist = np.linalg.inv(basis)
    A = FB.cgemm(1,A,basis)
    A = FB.cgemm(1,basist,A)
    B = FB.cgemm(1,B,basis)
    B = FB.cgemm(1,basist,B)
    C = FB.cgemm(1,C,basis)
    C = FB.cgemm(1,basist,C)
    D = FB.cgemm(1,D,basis)
    D = FB.cgemm(1,basist,D)
    mmc = np.zeros((dim,dim),dtype="complex64",order='F')
    mmd = np.zeros((dim,dim),dtype="complex64",order='F')
    mm = np.zeros((dim,dim),dtype="complex64",order='F')
    mm1 = np.zeros((dim,dim),dtype="complex64",order='F')
    tiempo = np.linspace(t0,N*dt+t0,N)
    otok = np.zeros(N,dtype = "complex64")
    U = np.zeros((dim,dim),dtype="complex64",order='F') # U evolucion temporal
    Udagger = np.zeros((dim,dim),dtype="complex64",order='F') # U*
    for ti in range(0,N):
        for c1 in range(0,dim):
            U[c1][c1] = np.exp(-1j*tiempo[ti]*ener[c1])
            Udagger[c1][c1] = np.exp(1j*tiempo[ti]*ener[c1])
        mmc = FB.cgemm(1,C,U)
        mmc= FB.cgemm(1,Udagger,mmc) # C(t)
        mmd = FB.cgemm(1,D,U)
        mmd= FB.cgemm(1,Udagger,mmd) # D(t)
        mm = FB.cgemm(1,A,B) # AB
        mm1 = FB.cgemm(1,mmc,mmd) #C(t)D(t) 
        mm = FB.cgemm(1,mm,mm1)  #ABC(t)D(t)     
        otok[ti] = np.matrix.trace(mm)/dim #Tr[ ABC(t)D(t)]      
        mm = FB.cgemm(1,A,mmc) #AC(t)
        mm1 = FB.cgemm(1,B,mmd) #BD(t)
        mm = FB.cgemm(1,mm,mm1) #AC(t)BD(t)
        otok[ti] += - (np.matrix.trace(mm)/dim).real #realTr( AC(t)BD(t))
    otok = np.array(otok)
    return otok
#----------------  (4) CHAOS OPERATIONS  ----------------#

#Level statistics/ NN level spacing distribution
def NN_lsd(yyy,beens,named):
    med1=0
    for count2 in range(0,len(yyy)):
        med1=med1+yyy[count2]        
    med1=med1/len(yyy)    
    s1=0
    for count2 in range(0,len(yyy)):
        s1=s1+(yyy[count2]-med1)**2    
    s1=np.sqrt(s1/len(yyy))
    con1=1./(np.sqrt(2.*np.pi)*s1)
    xs3 = np.linspace(yyy[0], yyy[len(yyy)-1], 100)
    fig = plt.figure()
    plt.subplot(211)
    plt.title(r"$Level statistics$")
    plt.hist(yyy,bins=151,normed=1,histtype='step',label="Statistics")
    plt.plot(xs3,con1*np.exp(-(xs3-med1)**2/(2.*s1**2)))    
    nt=100    
    xs = np.linspace(0, 4, nt)
    ys=np.pi*0.5*xs*np.exp(-0.25*np.pi*xs**2)
    ysp=np.exp(-xs)    
    yy=np.linspace(1,len(yyy),len(yyy))    
    de=[]
    nsaco=30
    for count2 in range(0+nsaco,len(yyy)-1-nsaco):
        de.append((yyy[count2+1]-yyy[count2])*(len(yyy)*con1*np.exp(-(yyy[count2]-med1)**2/(2.*s1**2))))
    xs3 = np.linspace(0, 4, 100)
    plt.subplot(212)
    plt.hist(de,bins=beens,normed=1,histtype='step',label="NN Level Spacing")
    plt.plot(xs,ys,label='Wigner')
    plt.plot(xs,ysp,label='Poisson')
    plt.legend(loc='best',fontsize = 'small')
    plt.savefig(str(named)+'.pdf')
    plt.close(fig)
    
# BRODY DISTRIBUTION 
def Brody_distribution(s,B):
    bebi = (math.gamma(((B+2)/(B+1)))) ** (B+1)
    return (B+1)*bebi*(s**B) * np.exp(-bebi*(s**(B+1)))

# BERRY ROBNIK DISTRIBUTION 
def BerryRobnikN2_distribution(s,p):
    p1 = 1 - p
    a = (p1)**2 * np.exp(-p1*s) * erfc(0.5*np.sqrt(np.pi) *p*s)
    b = (2*p1*p+0.5*np.pi*(p**3)*s) * np.exp(-p1*s-0.25*np.pi*(p**2)*(s**2))    
    return a+b


# Adjusts the Brody parameter to a discrete distribution of energy "yyy" separated in an amount bins "beens" (Recommended beens='auto')
def brody_param(yyy,beens):
    med1=0
    for count2 in range(0,len(yyy)):
        med1=med1+yyy[count2]
    med1=med1/len(yyy)
    s1=0
    for count2 in range(0,len(yyy)):
        s1=s1+(yyy[count2]-med1)**2
    
    s1=np.sqrt(s1/len(yyy))
    con1=1./(np.sqrt(2.*np.pi)*s1)
    xs3 = np.linspace(yyy[0], yyy[len(yyy)-1], 100)
    nt=100
    de=[]
    nsaco=30
    for count2 in range(0+nsaco,len(yyy)-1-nsaco):
        de.append((yyy[count2+1]-yyy[count2])*(len(yyy)*con1*np.exp(-(yyy[count2]-med1)**2/(2.*s1**2))))
    datos, binsdata = np.histogram(de,bins=beens,normed=True)
    prueba = binsdata
    deltilla = prueba[1]-prueba[0]
    pruebas = np.zeros(len(prueba)-1)
    for lau in range(1,len(prueba)):
        pruebas[lau-1] = lau * deltilla
    brody_paramet, pcov = curve_fit(Brody_distribution, pruebas, datos)
    return brody_paramet

# Adjusts the BerryRobnik parameter to a discrete distribution of energy "yyy" separated in an amount bins "beens" (Recommended beens='auto')
def BerryRobnikN2_param(yyy,beens):
    med1=0
    for count2 in range(0,len(yyy)):
        med1=med1+yyy[count2]
    med1=med1/len(yyy)
    s1=0
    for count2 in range(0,len(yyy)):
        s1=s1+(yyy[count2]-med1)**2
    
    s1=np.sqrt(s1/len(yyy))
    con1=1./(np.sqrt(2.*np.pi)*s1)
    xs3 = np.linspace(yyy[0], yyy[len(yyy)-1], 100)
    nt=100
    de=[]
    nsaco=30
    for count2 in range(0+nsaco,len(yyy)-1-nsaco):
        de.append((yyy[count2+1]-yyy[count2])*(len(yyy)*con1*np.exp(-(yyy[count2]-med1)**2/(2.*s1**2))))
    datos, binsdata = np.histogram(de,bins=beens,normed=True)
    prueba = binsdata
    deltilla = prueba[1]-prueba[0]
    pruebas = np.zeros(len(prueba)-1)
    for lau in range(1,len(prueba)):
        pruebas[lau-1] = lau * deltilla
    BR_paramet, pcov = curve_fit(BerryRobnikN2_distribution, pruebas, datos)
    return BR_paramet


# Calcultes r parameter in the 10% center of the energy "ener" spectrum. If plotadjusted = True, returns the magnitude adjusted to Poisson = 0 or WD = 1
def r_chaometer(ener,plotadjusted):
    ra = np.zeros(len(ener)-2)
    #center = int(0.5*len(ener))
    #delter = int(0.05*len(ener))
    for ti in range(len(ener)-2):
        ra[ti] = (ener[ti+2]-ener[ti+1])/(ener[ti+1]-ener[ti])
        ra[ti] = min(ra[ti],1.0/ra[ti])
    ra = np.mean(ra)
    if plotadjusted == True:
        ra = (ra -0.3863) / (-0.3863+0.5307)
    return ra
    
# Calculates the NPC of eigenvectors in comparison to the standard spin-site basis
def IPR_eigenstates(basis):
    dim1 = min(basis.shape)
    dim2 = max(basis.shape)
    ipr = np.zeros(dim1)
    for i in range(dim1):
        for j in range(dim2):
            ipr[i]+= (np.abs(basis[j,i]))**4
    ipr = 1.0/ipr
    return ipr

# Returns OTOC's Spectarl IPR and Standard deviation.
def OTOC_chaotic_measures(tiempo,otoks,lentest0,lentest,dt):
    lentest = int(lentest/dt)
    lentest0 = int(lentest0/dt)
    distri = 1
    otok_osc = otoks[lentest0:lentest] - np.mean(otoks[lentest0:lentest])
    std = np.std(otok_osc)
    fft_osc = np.fft.rfft(otok_osc,norm='ortho')
    freq = np.fft.rfftfreq(len(otok_osc), tiempo[1]-tiempo[0])
    dx = freq[1]-freq[0]
    b = 0.5*len(freq)-1
    distri= scipy.integrate.simps(np.abs(fft_osc)**2,freq)
    sipr_simpson= scipy.integrate.simps(np.abs(fft_osc)**4/(distri**2),freq)
    distri = 0
    #sipr_df = 0
    #for i in range(len(freq)):
    #    distri = distri + dx * (abs(fft_osc[i])**2)
    #for i in range(len(freq)):
    #    sipr_df = sipr_df + dx * (abs(fft_osc[i])**4/(distri**2)) 
    #sipr_df = 1/sipr_df
    sipr_simpson = 1/sipr_simpson
    return sipr_simpson, std

# Returns OTOC's Standard deviation.
def OTOC_chaotic_measures_std(tiempo,otoks,lentest0,lentest,dt):
    lentest = int(lentest/dt)
    lentest0 = int(lentest0/dt)
    otok_osc = otoks[lentest0:lentest] - np.mean(otoks[lentest0:lentest])
    std = np.std(otok_osc)
    return std

# Returns OTOC's Spectarl IPR and Standard deviation with a more sofisticated approach.
def OTOC_chaotic_measures_soph(tiempo,otoks,largo,largoz,N,dt,WINDOWTYPE):
    largofinal = int((N - int(N/(largo)))/largoz) 
    std = 0
    sipr_simpson = 0
    for lar in range(largofinal):
        lentest0 = int((lar*largoz))
        lentest = int(N/(largo))+int((lar*largoz))
        distri = 1
        otok_osc = otoks[lentest0:lentest] - np.mean(otoks[lentest0:lentest])
        otok_osc = otok_osc * scipy.signal.windows.get_window(WINDOWTYPE, len(otok_osc))
        std += np.std(otok_osc)
        fft_osc = np.fft.rfft(otok_osc,norm='ortho')
        freq = np.fft.rfftfreq(len(otok_osc), tiempo[1]-tiempo[0])
        dx = freq[1]-freq[0]
        b = 0.5*len(freq)-1
        distri = scipy.integrate.simps(np.abs(fft_osc)**2,freq)
        sipr_simpson += scipy.integrate.simps(np.abs(fft_osc)**4/(distri**2),freq)
    sipr_simpson = 1.0/sipr_simpson
    std = 1.0/std
    return sipr_simpson, std

#---------------- (5) SPIN CHAIN MODELS ----------------#

def Tilted_model(sites,BC,J,B,tita):
    H = .25*J*spin_interactions(sites,1,BC,0,0,1)
    H+= .5*B*(np.sin(tita)*S_x(sites) + np.cos(tita)*S_z(sites))
    e, ev = np.linalg.eigh(H)
    return e, ev

def Ising_XZ_fields(sites,BC,J,hx,hy,hz):
    H = -J*spin_interactions(sites,1,BC,0,0,1)
    H+= (hx*S_x(sites) + hy*S_y(sites) + hz*S_z(sites))
    e, ev = np.linalg.eigh(H)
    return e, ev

def Perturbed_XXZ(sites,BC,alpha,lambd):
    H = .25*spin_interactions(sites,1,BC,1,1,alpha)
    H+= .25*lambd*spin_interactions(sites,2,BC,1,1,alpha)
    e, ev = np.linalg.eigh(H)
    return e, ev
	
def Perturbed_XXZ_sz_subspace(sites,n_partic,BC,alpha,lambd):
    H = .25*sz_subspace_spin_interactions(sites,n_partic,1,BC,1,1,alpha)
    H+= .25*lambd*sz_subspace_spin_interactions(sites,n_partic,2,BC,1,1,alpha)
    e, ev = np.linalg.eigh(H)
    return e, ev

#----------------  (5) MISCELLANEA CODE  ----------------#

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

