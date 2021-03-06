
# %matplotlib inline
from math import pi
import numpy as np
import matplotlib.pyplot as plt

# change some of the defaults for plots
import copy

import itertools

spin_state_total=np.loadtxt('spin_state_0.002_0.01_different_init.txt',dtype=int)
print(np.shape(spin_state_total))

randomstate_num = 10
n_record_step = 1000*randomstate_num
n_total_step = 200000
p = 0.5                           # probability for the initial random lattice
Ts = np.linspace(0.002, 0.01, 58) * J / kB
#Ts2 = np.linspace(0.005, 0.04, 2) * J / kB
#Ts = np.concatenate((Ts1,Ts2),axis=0)

distance = range(7,8,1)
L = 100
L_sub=10

complexity_total = []
Integration_total=[]
H_bisys_combination_total = []
weight = np.logspace(0,L_sub-1,L_sub,endpoint=True,base=2)

for L_sub in distance:
    print("sub length is ", L_sub)
    complexity_total = []
    Integration_total = []
    #for step_T in range(len(Ts)):
    step_T = 30
    T= Ts[step_T]
    print("T is ", T)

    H_bisys_total_ave = []
    state_count_bisys = np.zeros((10, 260, 1026))
    MI = []
    num_total = []
    H_sum_eachspin_total=[]
    count_each_spin= np.zeros((100))
    H_eachspin_total = []
    Integration_linear = np.zeros((100))

    for step in range(n_record_step):
      #if(step%200==0):
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
            print(combination[combine_num])
            com_value = np.matmul(np.array(combination[combine_num]), np.array(weight[0:k+1].T))

            state_count_bisys[k,combine_num,int(com_value)] += 1

    #Take the first L_sub length spins as subsystem
    count_each_spin = count_each_spin[0:L_sub]

    #calculate entropy of every subsystem
    for k in range(L_sub):
        H_bisys =0
        H_bisys_combination=[]
        for num in range(num_total[k]):
            print(state_count_bisys[k,num,:])
            probability = state_count_bisys[k,num,:]/n_record_step
            probability[probability==0]=1
            H_bisys_combination.append(-sum(probability*np.log2(probability)))
            H_bisys += -sum(probability*np.log2(probability))
        H_bisys_combination_total.append(copy.copy(H_bisys_combination))
        H_bisys_total_ave.append(H_bisys / num_total[k])
    print("H_ave_bisys is ", H_bisys_total_ave)

    #calculate mutual information of every possible bipartition
    for k in range(L_sub-1):
        MI.append(H_bisys_total_ave[k] + H_bisys_total_ave[L_sub-k-2] - H_bisys_total_ave[L_sub-1])
    print("MI is ", MI)

    #calculate entropy of each spin
    for k in range(L_sub):
        probability = []
        probability.append(count_each_spin[k]/n_record_step)
        probability.append(1- probability[0])
        probability=np.array(probability)
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

    #calculate the total integration
    temp = sum(H_eachspin_total)-H_bisys_total_ave[L_sub-1]
    Integration_total.append(temp)

    #Cauculate the complexity
    Complexity = sum(Integration_linear[0:L_sub-1] - ave_integration)
    print("Complexity is ", Complexity)
    complexity_total.append(Complexity)

    for k in range(L_sub):
        x=np.linspace(0,num_total[k]-1,num_total[k])
        print(np.shape(x))
        print(np.shape(H_bisys_combination_total))
        print(H_bisys_combination_total)
        plt.scatter(x,H_bisys_combination_total[k])
        plt.xlabel('combination num')
        plt.ylabel('Entropy')
        plt.savefig('./H_bisys_entropy_%f_%d.jpg'% (T,k+1))
        plt.close()
