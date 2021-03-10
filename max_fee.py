from simulation_1 import main
import random
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms

############ Helper functions ################

def getHigherMean(list1, list2):
    avg1 = float(sum(list1))/len(list1)
    avg2 = float(sum(list2))/len(list2)

    if avg1 > avg2:
        return list1
    else:
        return list2

def getLowerMean(list1, list2):
    avg1 = float(sum(list1))/len(list1)
    avg2 = float(sum(list2))/len(list2)

    if avg1 <= avg2:
        return list1
    else:
        return list2

def calculateMax(list1, list2):
    lower = getLowerMean(list1, list2)
    higher = getHigherMean(list1, list2)

    ans = []
    for i in range(len(lower)):
        ans.append(higher[i] -lower[i])
    return ans

def chargeFee(bob, fee):
    ans = []
    for i in range(len(fee)):
        ans.append(bob[i] - fee[i])
    return ans

def payFee(alice, fee):
    ans = []
    for i in range(len(fee)):
        ans.append(alice[i] + fee[i])
    return ans

def getIntersections(list1, list2, ps):
    points = []

    for i in range(1, len(list1)):
        if list1[i] == list2[i]:
            points.append((i, list2[i]))
        elif ((list1[i-1] > list2[i-1]) and (list1[i] < list2[i])): 
            # or ((list1[i-1] < list2[i-1]) and (list1[i] > list2[i]))):
            points.append((ps[i], (list1[i-1]+list1[i])/2))

    return points

def transform(points):
    xs = []
    ys = []

    for i in points:
        xs.append(i[0])
        ys.append(i[1])

    return (xs, ys)


############## Constants and global variables ################

alice0, bob0, alice1, bob1, alice2, bob2 = [], [], [], [], [], []

num_trial = 500

time = 50

givenP = 0.5

freqMean = 1.50

freqSigma = 0.01


############# Main functions #####################

def runWithPayment(time):
    ps = [x* 1.0 /100 for x in range(1, 150)]

    f = np.random.lognormal(freqMean, freqSigma)
    # random.expovariate(givenP)
    # ps = np.arange(0.0, 1.0 + 0.01, 0.01)

    for i in range(len(ps)):
        # trial
        print(str(i))
        res = [0, 0, 0, 0, 0, 0]
        temp = []
        for k in range(num_trial):
            temp = main(p=ps[i], freq=f, timeRun = time)
            for j in range(len(temp)):
                res[j] += temp[j]
        for j in range(len(res)):
            res[j] = res[j]/float(num_trial)

        alice0.append(res[0])
        bob0.append(res[1])
        alice1.append(res[2])
        bob1.append(res[3])
        alice2.append(res[4])
        bob2.append(res[5])

    # if Bob is taking the highest fee he can, then we look at how his costs changes
    # the highest fee Bob can take is Alice's maximum difference of channel costs
    maxFee = calculateMax(alice1, alice2)

    # Alice's cost increase while Bob's cost decrease by the fee charged
    aliceAfter = payFee(alice0, maxFee)
    bobAfter = chargeFee(bob2, maxFee)
    
    

    alicePoints = transform(getIntersections(alice1, aliceAfter, ps))
    bobPoints = transform(getIntersections(bob0, bobAfter, ps))

    bobxs = bobPoints[0]
    bobys = bobPoints[1]
    alicexs = alicePoints[0]
    aliceys = alicePoints[1]

    aliceBenefit = chargeFee(alice1, aliceAfter)
    bobBenefit = chargeFee(bob0, bobAfter)

    
    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(ps, bob0, "k")
    ax.plot(ps, alice1, "r")
    # ax.plot(ps, bob2, "m")
    # ax.plot(ps, bob1, "k--")
    # ax.plot(ps, alice2, "c--")
    ax.plot(ps, aliceAfter, "b-")
    ax.plot(ps, bobAfter, "g-")
    # ax.plot(bobxs, bobys, "go")
    
    ax.plot(ps, aliceBenefit)
    ax.plot(ps, bobBenefit)
    ax.plot(ps, [0 for x in range(len(ps))], "k--")
    ax.plot(alicexs, aliceys, "bo")

    trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
                                       x=0.01, y=-0.30, units='inches')

    for x, y in zip(bobxs, bobys):
        plt.plot(x, y, 'go')
        # plt.text(x, y, '%02f, %02f' % (x, y), transform=trans_offset)

    fig.text(0, 0, 'Alice Points %s; Bob Points %s\n Number of trials: %d; Time of each network: %d' % (str(alicePoints), str(bobPoints), num_trial, time))


    ax.set_title('Transferred payment size vs Cost differences after Max Fee')
    ax.set_xlabel('Payment size')
    ax.set_ylabel('Cost differences')

    # plt.axis([0, 6, 0, 100])
    fig.legend(["bob", "alice", "alice' ", "bob'", "alice benefit", "bob benefit"])
    fig.savefig('paysFeeBenefit.png')


################ Call #####################

runWithPayment(time)
# runWithFreq()




