
# %matplotlib inline
from math import pi
import numpy as np
import itertools
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

L = 100  # lattice size is L
J = 1.0  # gauge coupling parameter
kB = 1.0  # the Boltzman constant

spin_state_total=np.loadtxt('spin_state_0.0035_0.005_1000.txt',dtype=int)
print(np.shape(spin_state_total))

n_record_step = 1000
n_total_step = 200000
step_size=int(n_total_step/n_record_step)
p = 0.5                           # probability for the initial random lattice
Ts = np.linspace(0.0035, 0.005, 30) * J / kB
length = range(4,12,2)
L = 100
L_sub=10
sub_length_max=10
combination_num_max=260
conditon_num_max=1026

complexity_total = []
Integration_total=[]
weight = np.logspace(0,L_sub-1,L_sub,endpoint=True,base=2)

for L_sub in length:
    print("sub length is ", L_sub)
    complexity_total = []
    Integration_total = []
    for step_T in range(len(Ts)):
        T= Ts[step_T]
        print("T is ", T)

        H_bisys_total_ave = []
        state_count_bisys = np.zeros((sub_length_max, combination_num_max, conditon_num_max))
        MI = []
        num_total = []
        H_sum_eachspin_total=[]
        count_each_spin= np.zeros((L))
        H_eachspin_total = []
        Integration_linear = np.zeros((L))

        for step in range(n_record_step):
              s = spin_state_total[step_T*n_record_step+step]
              s[s==-1] = 0

              # count the probability of each spin state
              count_each_spin += s

              s_sub = s[0:L_sub]
              s = np.array(s_sub)

              # count the probability for each combination of L_sub number of subsystems
              for k in range(L_sub):
                combination = list(itertools.combinations(s, k+1))
                num = len(combination)
                num_total.append(num)
                for combine_num in range(num):
                    com_value = np.matmul(np.array(combination[combine_num]), np.array(weight[0:k+1].T))
                    state_count_bisys[k,combine_num,int(com_value)] += 1

        #Take the first L_sub length spins as subsystem
        count_each_spin = count_each_spin[0:L_sub]

        #calculate entropy of every subsystem
        for k in range(L_sub):
            H_bisys =0
            for num in range(num_total[k]):
                probability = state_count_bisys[k,num,:]/n_record_step
                probability[probability==0]=1
                #print(probability)
                H_bisys += -sum(probability*np.log2(probability))
            H_bisys_total_ave.append(H_bisys / num_total[k])
        print("H_ave_bisys is ", H_bisys_total_ave)

        #calculate mutual information of every possible bipartition
        for k in range(L_sub-1):
            MI.append(H_bisys_total_ave[k] + H_bisys_total_ave[L_sub-k-2] - H_bisys_total_ave[L_sub-1])
        print("MI is ", MI)

        #calculate entropy of each spin
        probability = np.zeros((2))
        for k in range(L_sub):
            probability[0]=(count_each_spin[k]/n_record_step)
            probability[1]=(1- probability[0])
            probability[probability == 0] = 1
            H_eachspin = -sum(probability * np.log2(probability))
            H_eachspin_total.append(H_eachspin)
        print("H_eachspin is ", H_eachspin_total)

        #calculate the average entropy of sum of k saperate spin
        for k in range(L_sub-1):
            combination = list(itertools.combinations(H_eachspin_total, k + 1))
            num = len(combination)
            H_sum_eachspin=0
            for combine_num in range(num):
                H_sum_eachspin +=  sum(combination[combine_num])
            H_sum_eachspin_total.append(H_sum_eachspin/num)
        print("H_sum_eachspin_total is ", H_sum_eachspin_total)

        #calculate the integration of every possible bipartition
        Integration=np.array(H_sum_eachspin_total[0:L_sub-1])+np.array(H_sum_eachspin_total[L_sub-2::-1])+np.array(MI)\
                    -np.array(H_bisys_total_ave[0:L_sub-1])-np.array(H_bisys_total_ave[L_sub-2::-1])
        print("Integration is ", Integration)

        #calculate the linear integration
        for k in range(L_sub-1):
            Integration_linear[k]=Integration[k]*(k+1)/L_sub
        print("Linear Integration is ", Integration_linear[0:L_sub-1])

        #calculate the average integration of every possible bipartition
        ave_integration=np.array(H_sum_eachspin_total[0:L_sub-1])-np.array(H_bisys_total_ave[0:L_sub-1])
        print("Average Integration is ", ave_integration)
        Integration_total.append(sum(ave_integration)/(L_sub-1))

        #Cauculate the complexity
        Complexity = sum(Integration_linear[0:L_sub-1] - ave_integration)
        print("Complexity is ", Complexity)
        complexity_total.append(Complexity)

        plt.plot(H_bisys_total_ave)
        plt.xlabel('k')
        plt.ylabel('H_bisys_total_ave')
        plt.savefig('./H_bisys_total_ave_%f_%d.jpg'% (T,L_sub))
        plt.close()

        plt.plot(MI)
        plt.xlabel('k')
        plt.ylabel('Mutual information')
        plt.savefig('./Mutual information_%f_%d.jpg'% (T,L_sub))
        plt.close()

        plt.plot(Integration)
        plt.xlabel('k')
        plt.ylabel('Integration')
        plt.savefig('./Integration_%f_%d.jpg'% (T,L_sub))
        plt.close()

        plt.plot(ave_integration,label='Integration average')
        plt.xlabel('k')
        plt.ylabel('Integration sys')
        plt.plot(Integration_linear[0:L_sub-1],label='Linear integration')
        plt.xlabel('k')
        plt.legend()
        plt.savefig('./Integration_linear_ave_%f_%d.jpg'% (T,L_sub))
        plt.close()

    plt.plot(Ts,complexity_total)
    plt.xlabel('Temperature')
    plt.ylabel('complexity')
    plt.savefig('./complexity_total%f.jpg'% (T))
    plt.close()
    plt.plot(Ts,Integration_total)
    plt.xlabel('Temperature')
    plt.ylabel('Integration')
    plt.savefig('./Integration_total%f.jpg'% (T))
    plt.close()
