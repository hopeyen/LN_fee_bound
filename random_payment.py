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

def calculateFee(list1, list2):
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

num_trial = 200

time = 50

paymentMean = 0.65

paymentSigma = 0.1


############# Main functions #####################

def runWithPayment(time):
    ps = np.random.lognormal(paymentMean, paymentSigma)

    fs = [x* 1.0 / 100  for x in range(1,100)]

    for i in range(len(fs)):
        # trial
        print(i)
        res = [0, 0, 0, 0, 0, 0]
        temp = []
        for k in range(num_trial):
            temp = main(p = ps, freq=fs[i], timeRun = time)
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
    maxFee = calculateFee(alice1, alice2)
    aliceAfter_max = payFee(alice0, maxFee)
    bobAfter_max = chargeFee(bob2, maxFee)

    # if Bob is taking the lowest fee he can, it would be the difference between
    # if he transfer for Alice (bob2) and if he does not (bob0=bob1) 
    minFee = calculateFee(bob2, bob0)
    aliceAfter_min = payFee(alice0, minFee)
    bobAfter_min = chargeFee(bob2, minFee)

    # get the benefit by analyzing the costs in different networks
    aliceBenefit_max = chargeFee(alice1, aliceAfter_max)
    bobBenefit_max = chargeFee(bob0, bobAfter_max)
    aliceBenefit_min = chargeFee(alice1, aliceAfter_min)
    bobBenefit_min = chargeFee(bob0, bobAfter_min)


    titles = ['costs after max fee vs freqeuncy', 
            'costs after min fee vs frequency']
    Zs = [(aliceBenefit_max, bobBenefit_max), (aliceBenefit_min, bobBenefit_min)]

    fig = plt.figure(figsize=plt.figaspect(0.5))

    for i in range(0, 2):
        ax = fig.add_subplot(1, 2, i+1)
        ax.set_xlabel('Frequency (lambda)')
        ax.set_ylabel('Costs/benefit ')
        
    
        ax.plot(fs, bob0, "k")
        ax.plot(fs, alice1, "r")
        ax.plot(fs, Zs[i][0], "b-")
        ax.plot(fs, Zs[i][1], "g-")
        ax.plot(fs, [0 for x in range(len(fs))], "k--")

        fig.text(0, 0, 'Number of trials: %d; Time of each network: %d; Freq mean: %0.2f' % (num_trial, time, freqMean))
        ax.set_title(titles[i])
        fig.legend(["bob", "alice", "alice' ", "bob'", "alice benefit", "bob benefit"])
    fig.savefig('frequencyBenefit.png')


################ Call #####################

runWithPayment(time)
# runWithFreq()




