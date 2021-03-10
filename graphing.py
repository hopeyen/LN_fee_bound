from simulation_1 import main
import random
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def Rand(start, end, num): 
    res = [] 
    for j in range(num): 
        res.append(random.randint(start, end)) 
    return res

def getIntersectionPoint(list1, list2):
    # start with list1 being larger than list2

    for i in range(len(list1)):
        if list1[i]<= list2[i]:
            return list2[i]
    return 0



upperbound, lowerbound, intersection = [], [], []
estimatedubd, estimatedlbd = [], []

num_trial = 1000
time = 20

times = [x* 10.0 for x in range(1, time)]


# transfer payment values

def runWithPayment(time):
    ps = [x* 1.0 /10 for x in range(1, 100)]
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

        upperbound.append(res[0])
        lowerbound.append(res[1])
        estimatedlbd.append(res[2])
        estimatedubd.append(res[3])
    # pnw1.append()

    fig, ax = plt.subplots()
    ax.plot(ps, upperbound)
    ax.plot(ps, lowerbound)



    # ax.plot(upperbound, 'b-', lowerbound, 'g-', )
    ax.set_title('Transferred payment size vs Cost differences')
    ax.set_xlabel('Payment')
    ax.set_ylabel('Cost differences')

    # plt.axis([0, 6, 0, 100])
    fig.legend(["upperbound", "lowerbound"])
    fig.savefig('output.png')


    return getIntersectionPoint(upperbound, lowerbound)


def runWithFreq():
    fs = [x* 10.0  for x in range(1,100)]
    for i in range(len(fs)):
        # trial
        res = [0,0, 0,0]
        temp = []
        for k in range(num_trial):
            temp = main(freq=fs[i])
            for j in range(len(temp)):
                res[j] += temp[j]
        for j in range(len(res)):
            res[j] = res[j]/float(num_trial)

        upperbound.append(res[0])
        lowerbound.append(res[1])
        estimatedlbd.append(res[2])
        estimatedubd.append(res[3])

    plt.plot(upperbound, 'b-', lowerbound, 'g-', estimatedlbd, 'b--', estimatedubd, 'g--')

    plt.title('frequency vs costs')
    plt.xlabel("frequnecy \n 1000 trials")
    plt.ylabel("bound")
    plt.legend(["upperbound", "lowerbound", "estimatedlbd", "estimatedubd"])
    plt.savefig('output.png')


def runWithTime():

    interescts = []

    for t in times:
        print(t)
        interescts.append(run(t))
    return interescts


# interescts = runWithTime()

# plt.subplot(211)
# plt.plot(times, interescts)
# plt.subplot(212)

runWithPayment(10)
# runWithFreq()



def simple():
    t = np.arange(0.0, 100.0, 0.1)
    s = np.sin(0.1 * np.pi * t) * np.exp(-t * 0.01)

    fig, ax = plt.subplots()
    ax.plot(t, s)

    # Make a plot with major ticks that are multiples of 20 and minor ticks that
    # are multiples of 5.  Label major ticks with '.0f' formatting but don't label
    # minor ticks.  The string is used directly, the `StrMethodFormatter` is
    # created automatically.
    ax.xaxis.set_major_locator(MultipleLocator(40))
    ax.xaxis.set_major_formatter('{x:.0f}')

    # For the minor ticks, use no labels; default NullFormatter.
    ax.xaxis.set_minor_locator(MultipleLocator(10))

    plt.savefig('simple.png')


# simple()