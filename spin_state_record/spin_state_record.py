
#%matplotlib inline
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import copy
# change some of the defaults for plots
plt.rcParams['text.usetex'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.figsize'] = [18,6]
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
from IPython.display import display, Markdown, Latex, Math, Pretty

from timeit import timeit
from numpy.random import rand, randint

L = 100          # lattice size is L
J = 1.0          # gauge coupling parameter
kB = 1.0         # the Boltzman constant

def metropolis( s, T ) :
    '''
    This runs the Metropolis algorithm for one unit of "Monte Carlo
    time (MCT)", defined as N steps where N is the number of items in the
    ensemble. In this case this is L*L.
    '''
    L = len( s )
    oldE = energy( s )
    for n in range( L ) : # this loop is 1 MCT

        # flip a random spin and calculate deltaE
        i = randint( L )
        s[i] *= -1    # flip the i-th spin

        newE = energy( s )
        deltaE = newE - oldE

        # these are the Metropolis tests
        if deltaE < 0 :
            oldE = newE
            # keep the flipped spin because it lowers the energy
        elif rand( ) < np.exp(  - deltaE / ( kB * T ) ) :
            oldE = newE
            # keep the spin flip because a random number in [0,1)
            # is less than exp( -dE / k_B T)
        else :
            # the spin flip is rejected
            s[i] *= -1    # unflip the ij spin

    return s

'''
Functions to calculate Energy (E) and Magnetic Moment (M) of the L*L spin lattice
Particles at the edge of the lattice rollover for adjacent calculation using periodic boundary conditions through np.roll
'''

def energy( s ) :
    # this is the energy for each site
    E = -J * ( s * np.roll( s, 1 ) )
    # and this is the avg energy per site
    return np.sum( E ) / L

#Simple sum over spin of all particles
def magnetization( s ) :
    return np.sum( s ) / L

#Creates an LxL lattice of random integer spins with probability (p) to be +1 (and 1-p to be -1)
def randomLattice( L, p ) :

    return ( rand( L ) < p ) * 2 - 1

p = 0.5                           # probability for the initial random lattice
T = 0.01 * J / kB                    # temp of the system in terms of coupling parameter and boltzmann constant
L = 100                             #The length of model
n = 200000                        #number of record Monte Carlo steps
spin_state_record=[]

Ts = np.linspace(30, 100, 60) * J / kB
spinLattice_t_init = randomLattice(L, p)


# use metropolis algo to record the spin state of random 1D lattices under different temperature
for j in range(len(Ts)):  # these are the temperatures  ## <-- for T in Ts:  then replace Ts[j] with T in all the metropolis calls below

    for i in range(100000):  # these are the MCT steps
        if i == 0:
            spinLattice_t = metropolis(spinLattice_t_init, Ts[j])
        spinLattice_t = metropolis(spinLattice_t, Ts[j])

    for i in range(n):  # these are the MCT steps
        spinLattice_t = metropolis(spinLattice_t, Ts[j])
        if (i % 200 == 0):
            temp = copy.copy(spinLattice_t)
            spin_state_record.append(temp)

temp = np.array(spin_state_record,dtype=int)
# np.savetxt('spin_state.txt', temp, fmt='%d')  ## <-- np.save('spin_state.npy', temp)
