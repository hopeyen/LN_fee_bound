from simulation_1 import main
import random
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec


alice1, bob1, alice2, bob2, alice3, bob3, alice4, bob4 = [], [], [], [], [], [], [], []

num_trial = 10
time = 10

times = [x* 10.0 for x in range(1, time)]

def getHigherMean(list1, list2):
    avg1 = mean(list1)
    avg2 = mean(list2)

    if avg1 > avg2:
        return avg1
    else:
        return avg2

def getHighestMean(lists):
    highest = lists[0]

    for i in lists:
        highest = getHigherMean(highest, i)

    return highest

# transfer payment values

def runWithPayment(time):
    ps = [x* 1.0 /1000 for x in range(1, 100)]
    # ps = np.arange(0.0, 1.0 + 0.01, 0.01)

    for i in range(len(ps)):
        # trial
        res = [0, 0, 0, 0, 0, 0, 0, 0]
        temp = []
        for k in range(num_trial):
            temp = main(p=ps[i], timeRun = time)
            for j in range(len(temp)):
                res[j] += temp[j]
        for j in range(len(res)):
            res[j] = res[j]/float(num_trial)

        alice1.append(res[0])
        bob1.append(res[1])
        alice2.append(res[2])
        bob2.append(res[3])
        alice3.append(res[4])
        bob3.append(res[5])
        alice4.append(res[6])
        bob4.append(res[7])


    # fig, ax = plt.subplots()
    # ax.plot(ps, alice1)
    # ax.plot(ps, bob1)
    # # ax.plot(ps, alice2)
    # ax.plot(ps, bob2)
    # ax.plot(ps, alice3)
    # ax.plot(ps, bob3)
    # ax.plot(ps, alice4)
    # ax.plot(ps, bob4)
    
    fig, ax = plt.subplots()
    # ax.plot(ps, alice1)
    # ax.plot(ps, bob1)
    ax.plot(ps, alice2)
    ax.plot(ps, bob2)
    # ax.plot(ps, alice3)
    # ax.plot(ps, bob3)
    # ax.plot(ps, alice4)
    # ax.plot(ps, bob4)
    




    # ax.plot(upperbound, 'b-', lowerbound, 'g-', )
    ax.set_title('Transferred payment size vs Cost differences')
    ax.set_xlabel('Payment')
    ax.set_ylabel('Cost differences')

    # plt.axis([0, 6, 0, 100])
    fig.legend(["alice1", "bob1", "alice2", "bob2", "alice3", "bob3", "alice4", "bob4"])
    fig.savefig('networks.png')



    print(len(alice3))
    print(len(bob3))

runWithPayment(50)
# runWithFreq()




