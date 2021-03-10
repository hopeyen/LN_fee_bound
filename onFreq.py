import simulation_1
from simulation_1 import mainOppo
import random
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms



############## Constants and global variables ################

alice0, bob0, alice1, bob1, alice2, bob2 = [], [], [], [], [], []

num_trial = 20

time = 50

paymentMean = 0.5

paymentSigma = 0.0001

largeFrequency = 1.0

largePayments = 1.0


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
        elif (((list1[i-1] > list2[i-1]) and (list1[i] < list2[i])) 
            or ((list1[i-1] < list2[i-1]) and (list1[i] > list2[i]))):
            points.append((ps[i], (list1[i-1]+list1[i])/2))

    if len(points)>2:
        return [points[0], points[-1]]
    return points

def getIntercepts(list, ps):
    points = []

    for i in range(1, len(list)):
        if list[i] == 0:
            points.append((ps[i], 0))
        elif (((list[i-1] > 0) and (list[i] < 0))
            or (((list[i-1] < 0) and (list[i] > 0)))):
            points.append(((ps[i]+ps[i-1])/2, 0))

    if len(points)>2:
        return [points[0], points[-1]]
    return points

def getMaxFee(a0, a1):
    return a1 - a0

def getMinFee(a0, a2, b0, b2):
    b22 = b2 + a2 - a0
    return b22 - b0

def getFees(ps, f, time):
    # (a0+c0, b0, a1+c1, b1, a2+c2, b2)
    #    a0   b0   a1    b1   a2   b2
    temp = setUp(p=ps, freq=f, timeRun = time)
    return (getMaxFee(temp[0], temp[2]), getMinFee(temp[0], temp[4], temp[1], temp[5]))

def getExpectedTx(list):
    res = [0 for i in range(len(list))]
    for i in range(len(list)):
        res[i] = 1/list[i]

    return res



def networkOG(p, freq, onlineTX = 5.0, onlineTXTime = 1.0, r = 0.01, timeRun = 10.0):
    # a star / fork network
    network = simulation_1.Network(onlineTX, onlineTXTime, r, timeRun)
    
    Alice = simulation_1.Node("Alice", network)
    Bob = simulation_1.Node("Bob", network)
    Charlie = simulation_1.Node("Charlie", network)

    paymentAB = simulation_1.Payment(largeFrequency, largePayments, Alice, Bob)
    paymentBC = simulation_1.Payment(largeFrequency, largePayments, Bob, Charlie)

    channelAB = simulation_1.Channel(Alice, Bob, network)
    channelBC = simulation_1.Channel(Bob, Charlie, network)

    channelAB.addPaymentList([paymentAB])
    channelBC.addPaymentList([paymentBC])

    
    network.addPaymentList([paymentAB, paymentBC])
    # print("network0")
    network.runNetwork()
    # network.printSummary()

    a = Alice.getChCostTotal()
    b = Bob.getChCostTotal()
    c = Charlie.getChCostTotal()

    return (a+c, b)

def networkDirectAC(p, freq, onlineTX = 5.0, onlineTXTime = 1.0, r = 0.01, timeRun = 10.0):
    # set up the network
    network = simulation_1.Network(onlineTX, onlineTXTime, r, timeRun)

    Alice = simulation_1.Node("Alice", network)
    Bob = simulation_1.Node("Bob", network)
    Charlie = simulation_1.Node("Charlie", network)

    paymentAB = simulation_1.Payment(largeFrequency, largePayments, Alice, Bob)
    paymentBC = simulation_1.Payment(largeFrequency, largePayments, Bob, Charlie)
    paymentAC = simulation_1.Payment(freq, p, Alice, Charlie)

    channelAB = simulation_1.Channel(Alice, Bob, network)
    channelAB.addPayment(paymentAB)
    channelBC = simulation_1.Channel(Bob, Charlie, network)
    channelBC.addPayment(paymentBC)

    # Alice creates a direct channel for network 1
    channelAC = simulation_1.Channel(Alice, Charlie, network)
    channelAC.addPayment(paymentAC)
    
    # print("network1")
    network.addPaymentList([paymentAB, paymentBC, paymentAC])

    network.runNetwork()
    # network.printSummary()

    a = Alice.getChCostTotal()
    b = Bob.getChCostTotal()
    c = Charlie.getChCostTotal()

    return (a+c, b)

def networktransferB(p, freq, onlineTX = 5.0, onlineTXTime = 1.0, r = 0.01, timeRun = 10.0):
    # network 2 
    network = simulation_1.Network(onlineTX, onlineTXTime, r, timeRun)
    
    Alice = simulation_1.Node("Alice", network)
    Bob = simulation_1.Node("Bob", network)
    Charlie = simulation_1.Node("Charlie", network)

    paymentAB = simulation_1.Payment(largeFrequency, largePayments, Alice, Bob)
    paymentBC = simulation_1.Payment(largeFrequency, largePayments, Bob, Charlie)
    paymentAC = simulation_1.Payment(freq, p, Alice, Bob)
    paymentAC1 = simulation_1.Payment(freq, p, Bob, Charlie)
    paymentAC.setTransfer(paymentAC1)

    channelAB = simulation_1.Channel(Alice, Bob, network)
    channelAB.addPayment(paymentAB)
    channelBC = simulation_1.Channel(Bob, Charlie, network)
    channelBC.addPayment(paymentBC)

    # payment goes through Channel AB and BC
    channelAB.addPayment(paymentAC)
    channelBC.addPayment(paymentAC1)


    network.addPaymentList([paymentAB, paymentBC, paymentAC])
    network.addTransferredList([paymentAC1])

    # print("network2")
    network.runNetwork()
    # network.printSummary()
    # print([paymentAC2, paymentAC2.numPaid])


    a = Alice.getChCostTotal()
    b = Bob.getChCostTotal()
    c = Charlie.getChCostTotal()

    return (a+c, b)

def setUp(p=0.1, freq=0.5, onlineTX = 5.0, onlineTXTime = 1.0, r = 0.01, timeRun = 10.0):
    simulation_res = []
    simulation_res.append(networkOG(p, freq, timeRun=time))
    simulation_res.append(networkDirectAC(p, freq, timeRun=time))
    simulation_res.append(networktransferB(p, freq, timeRun=time))

    return simulation_res



############# Main functions #####################

def runWithFreq(time, interest, B):
    fs = [x* 2.0 /1 for x in range(1, 260)]
    print("fs %f to %f" %(fs[0], fs[-1]))

    for i in range(len(fs)):
        # trial
        print(str(i))
        res = [0, 0, 0, 0, 0, 0]
        temp = []
        for k in range(num_trial):
            pm = np.random.exponential(paymentMean)
            # print("pm: %f; f: %f" %(pm, fs[i]))
            tmp = setUp(p=pm,  freq=fs[i], timeRun = time, r = interest, onlineTX = B)
            temp = list(np.concatenate(tmp).flat)
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
    maxFee = chargeFee(alice1, alice0)
    
    # aliceAfter_max = payFee(alice0, maxFee)

    # bob2'=bob2+(alice2-alice0)
    # minFee = bob2'-bob0 = bob2+(alice2-alice0)-bob0 
    bob22 = payFee(bob2, chargeFee(alice2, alice0))

    minFee = chargeFee(bob22, bob0)
    print("alice OG")
    print(alice0)
    print("alice1")
    print(alice1)
    print("alice2")
    print(alice2)
    print("bob OG")
    print(bob0)
    print("bob1")
    print(bob1)
    print("bob2")
    print(bob2)
    print("\n")

    print("alice escapes")
    print(chargeFee(alice2, alice0))
    print("Bob22")
    print(bob22)
    print("\n")

    print("max fee")
    print(maxFee)
    print("min fee")
    print(minFee)



    print("max min diff" + str(chargeFee(maxFee, minFee)))
    print("param P " + str(paymentMean))
    

    expectedTx = getExpectedTx(fs)
    diff = chargeFee(maxFee, minFee)

    inter = getIntercepts(diff, expectedTx)
    


    titles = ['Maximum fee and minimum fee vs number of transactions', 'Maximum fee and minimum fee vs frequency']
    # Zs = [(aliceBenefit_max, bobBenefit_max), (aliceBenefit_min, bobBenefit_min)]
    xlabels = [' (expected) number of transactions within 1 time period', 'frequency (lambda, time between two payments) ']
    # maxFee.reverse()
    # minFee.reverse()


    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(xlabels[0])
    ax.set_title(titles[0])
    ax.set_ylabel('fee')

    ax.semilogx(expectedTx, maxFee, "k--")
    ax.semilogx(expectedTx, minFee, "r-.")
    ax.semilogx(expectedTx, diff, "b")
    ax.semilogx(expectedTx, [0 for x in range(len(expectedTx))])
    ax.axvline(interest/B)

    for pt in range(len(inter)):
        label = '({:.3f}, {:.3f})'.format(inter[pt][0], inter[pt][1])
        ax.annotate(label, (inter[pt][0], inter[pt][1]),
            textcoords="offset points",
            xytext = (2,2),
            rotation=45)

    fig.text(0, 0, 'Trials: %d; Time: %d; Payment mean: %0.2f' % (num_trial, time, paymentMean))
    fig.legend(["maximum fee", "minimum fee", "difference between max and min"])
    fig.savefig('3n_f0.png')



    inter = getIntercepts(diff, fs)

    fig2 = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig2.add_subplot(1, 1, 1)
    ax.set_xlabel(xlabels[1])
    ax.set_ylabel('fee')

    ax.plot(fs, maxFee, "k--")
    ax.plot(fs, minFee, "r-.")
    ax.plot(fs, diff, "b")
    ax.plot(fs, [0 for x in range(len(fs))])
    ax.axvline(B/interest)

    for pt in range(len(inter)):
        label = '({:.3f}, {:.3f})'.format(inter[pt][0], inter[pt][1])
        ax.annotate(label, (inter[pt][0], inter[pt][1]),
            textcoords="offset points",
            xytext = (2,2),
            rotation=45)

    fig2.text(0, 0, 'Trials: %d; Time: %d; Payment mean: %0.2f' % (num_trial, time, paymentMean))
    ax.set_title(titles[1])
    fig2.legend(["maximum fee", "minimum fee", "difference between max and min"])
    fig2.savefig('3n_f2.png')

################ Call #####################

runWithFreq(time, 0.01, 5.0)

