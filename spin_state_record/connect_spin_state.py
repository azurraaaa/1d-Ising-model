# %matplotlib inline
import copy
from math import pi
import numpy as np
import matplotlib.pyplot as plt

# change some of the defaults for plots
plt.rcParams['text.usetex'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.figsize'] = [18, 6]
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
from IPython.display import display, Markdown, Latex, Math, Pretty

from timeit import timeit
from numpy.random import rand, randint

# spin_state_total0=np.loadtxt('spin_state_0.001931_different_init.txt',dtype=int)
# print(np.shape(spin_state_total0))

spin_state_total1=np.loadtxt('spin_state_0.004000_different_init.txt',dtype=int)
print(np.shape(spin_state_total1))

spin_state_total2=np.loadtxt('spin_state_0.006000_different_init.txt',dtype=int)
print(np.shape(spin_state_total2))

spin_state_total3=np.loadtxt('spin_state_0.008000_different_init.txt',dtype=int)
print(np.shape(spin_state_total3))

spin_state_total4=np.loadtxt('spin_state_0.010053_different_init.txt',dtype=int)
print(np.shape(spin_state_total4))

randomstate_num0 = 30
n_record_step0 = 1000*randomstate_num0
randomstate_num1 = 20
n_record_step1 = 1000*randomstate_num1
randomstate_num2 = 14
n_record_step2 = 1000*randomstate_num2
randomstate_num = 10
n_record_step = 1000*randomstate_num

spin_state_total = []
J=1
kB=1
Ts0 = np.linspace(0.001, 0.002, 10) * J / kB
Ts1 = np.linspace(0.002, 0.004, 15) * J / kB
Ts2 = np.linspace(0.004, 0.006, 15) * J / kB
Ts3 = np.linspace(0.006, 0.008, 15) * J / kB
Ts4 = np.linspace(0.008, 0.01, 14) * J / kB
# for num in range(len(Ts0)):
#     for step in range(n_record_step0):
#         if(step >= n_record_step):
#             break
#         spin_state_total.append(copy.copy(spin_state_total0[num*n_record_step0+step]))

for num in range(len(Ts1)):
    if (num == 0):
        continue
    for step in range(n_record_step1):
        if(step >= n_record_step):
            break
        spin_state_total.append(copy.copy(spin_state_total1[num*n_record_step1+step]))

for num in range(len(Ts2)):
    for step in range(n_record_step2):
        if(step >= n_record_step):
            break
        spin_state_total.append(copy.copy(spin_state_total2[num*n_record_step2+step]))

for num in range(len(Ts3)):
    for step in range(n_record_step):
        spin_state_total.append(copy.copy(spin_state_total3[num*n_record_step+step]))

for num in range(len(Ts4)):
    for step in range(n_record_step):
        spin_state_total.append(copy.copy(spin_state_total4[num*n_record_step+step]))
# Ts = np.linspace(0.005, 1, 50) * J / kB
# for num in range(len(Ts)):
#     for step in range(n_record):
#         spin_state_total.append(copy.copy(spin_state_total2[num*n_record+step]))

temp = np.array(spin_state_total,dtype=int)
np.savetxt('spin_state_0.002_0.01_different_init.txt', temp, fmt='%d')

