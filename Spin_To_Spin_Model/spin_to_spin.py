from IPython.display import display, Markdown, Latex, Math, Pretty

from timeit import timeit
from numpy.random import rand, randint
from math import pi
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.figsize'] = [18,6]
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

spin_state_total=np.loadtxt('a.txt',dtype=int)
print(np.shape(spin_state_total))

L = 100
ave = L                         #average number of combinations
J = 1
kB = 1
Ts = np.linspace( 0.0035, .005, 30 ) * J / kB
n_record=1000
n_total = 200000

MI_total = []
H_marginal_total = []
H_joint_total = []
Dist = range(1,100,5)

#Calculate MI under the different temperatures and different spin-to-spin distance
for num in range( len( Ts ) ):  # these are the temperatures
    count = np.zeros((2000, 4))  # record the count  ## <-- What is this counting?  Why is the shape [2000, 4]?  Same with all the instantiations below
    probability_joint = [0, 0, 0, 0]  # record the joint probability
    probability = [0, 0, 0, 0]  # record the probability
    probability_joint_show = np.zeros((2, 2))
    MI = np.zeros((20))  # mutual information
    H_marginal = np.zeros((20))
    H_joint = np.zeros((20))

    #count the two spin states among monte carlo process
    for step in range(n_record):
        s = spin_state_total[num*n_total+step*200,:]

        #count spin states under different combinations and also different spin-to-spin distance
        for ave_num in range(ave):
            for r2 in Dist:
                distance_num = int((r2-1)/5)  ## <-- What is this?  If you are trying to keep an index for the distances, try using "for distance_index, r2 in enumerate(Dist):"
                loop = int(100 * distance_num)
                r = ave_num

                if (s[r] == 1 and s[r + r2 if r < L - r2 else r + r2 - L] == 1):  ## <-- replace this complicated if else with (r+r2) % L
                    count[ave_num+loop, 0] += 1
                elif (s[r] == 1 and s[r + r2 if r < L - r2 else r + r2 - L] == -1):
                    count[ave_num+loop, 1] += 1
                elif (s[r] == -1 and s[r + r2 if r < L - r2 else r + r2 - L] == 1):
                    count[ave_num+loop, 2] += 1
                elif (s[r] == -1 and s[r + r2 if r < L - r2 else r + r2 - L] == -1):
                    count[ave_num+loop, 3] += 1

    #Cauculate the MI
    for ave_num in range(ave):
        for r2 in Dist:
            distance_num = int((r2 - 1) / 5)
            loop = int(100 * distance_num)
            for i in range(2):
                for k in range(2):
                    probability_joint_show[i, k] = count[ave_num+loop, i * 2 + k] / n_record
                    probability_joint[i * 2 + k] = count[ave_num+loop, i * 2 + k] / n_record if count[ave_num+loop, i * 2 + k] != 0 else 1
            probability[0] = (count[ave_num+loop, 0] + count[ave_num+loop, 1]) / n_record if (count[ave_num+loop, 0] + count[
                ave_num+loop, 1]) != 0 else 1
            probability[1] = (count[ave_num+loop, 2] + count[ave_num+loop, 3]) / n_record if (count[ave_num+loop, 2] + count[
                ave_num+loop, 3]) != 0 else 1
            probability[2] = (count[ave_num+loop, 0] + count[ave_num+loop, 2]) / n_record if (count[ave_num+loop, 0] + count[
                ave_num+loop, 2]) != 0 else 1
            probability[3] = (count[ave_num+loop, 1] + count[ave_num+loop, 3]) / n_record if (count[ave_num+loop, 1] + count[
                ave_num+loop, 3]) != 0 else 1
            MI[distance_num] += -sum(probability * np.log2(probability)) + sum(probability_joint * np.log2(probability_joint))
            H_marginal[distance_num] += -sum(probability * np.log2(probability))
            H_joint[distance_num] += -sum(probability_joint * np.log2(probability_joint))
    H_marginal = H_marginal / ave
    H_joint = H_joint / ave
    MI = MI / ave

    print("probality is", probability)
    print("H(x)+H(y) is ", H_marginal)
    plt.imshow(probability_joint_show)
    plt.savefig('./probability_joint_show%d.jpg'% (num))
    plt.close()
    #plt.show()

    print("joint probability is ", probability_joint, probability_joint_show)
    print("H(x,y) is ", H_joint)
    print("Mutual information at temperature Ts ", Ts[num], " is ", MI)
    H_marginal_total.append(H_marginal)
    H_joint_total.append(H_joint)
    MI_total.append(MI)

#plot the parameter as a function of distance(under different temperatures)
for num in range( len( Ts ) ):
    plt.plot(Dist, MI_total[num][:])
    plt.xlabel('Distance')
    plt.ylabel('Mutual information')
    plt.savefig('./Mutual information%f.jpg'%(Ts[num]))
    plt.close()
    # plt.show()
    plt.plot(Dist, H_marginal_total[num][:])
    plt.xlabel('Distance')
    plt.ylabel('H_marginal_total')
    plt.savefig('./H_marginal_total%d.jpg'% (Ts[num]))
    plt.close()
    # plt.show()
    plt.plot(Dist, H_joint_total[num][:])
    plt.xlabel('Distance')
    plt.ylabel('H_joint_total')
    plt.savefig('./H_joint_total%d.jpg'% (Ts[num]))
    plt.close()
    # plt.show()

#plot the parameter as a function of temperature(under different distance)
MI_total = np.array(MI_total).T
H_marginal_total = np.array(H_marginal_total).T
H_joint_total=np.array(H_joint_total).T
for distance in range(20):
    plt.plot(Ts, MI_total[distance][:])
    plt.xlabel('Temperature')
    plt.ylabel('Mutual information')
    plt.savefig('./Mutual information%d.jpg'%(distance*5))
    plt.close()
    #plt.show()
    plt.plot(Ts, H_marginal_total[distance][:])
    plt.xlabel('Temperature')
    plt.ylabel('H_marginal_total')
    plt.savefig('./H_marginal_total%d.jpg'%(distance*5))
    plt.close()
    #plt.show()
    plt.plot(Ts, H_joint_total[distance][:])
    plt.xlabel('Temperature')
    plt.ylabel('H_joint_total')
    plt.savefig('./H_joint_total%d.jpg'%(distance*5))
    plt.close()
    #plt.show()

