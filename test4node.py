import simulation_1
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

num_trial = 1

time = 50

givenP = 0.5

largeFrequency = 1.0

largePayments = 1

littleP = 0.5

littleF = 1.5


############ Helper functions ################

def networkStar(p, freq, onlineTX, onlineTXTime, r, timeRun):
    # a star / fork network
    network = simulation_1.Network(onlineTX, onlineTXTime, r, timeRun)
    
    Alice = simulation_1.Node("Alice", network)
    Bob = simulation_1.Node("Bob", network)
    Charlie = simulation_1.Node("Charlie", network)
    Donna = simulation_1.Node("Donna", network)

    paymentAB = simulation_1.Payment(largePayment, largeFrequency, Alice, Bob)
    paymentBC = simulation_1.Payment(largePayment, largeFrequency, Bob, Charlie)
    paymentAD = simulation_1.Payment(freq, p, Alice, Bob)
    paymentAD1 = simulation_1.Payment(freq, p, Bob, Donna)
    paymentBD = simulation_1.Payment(largePayment, largeFrequency, Bob, Donna)
    paymentCD = simulation_1.Payment(largePayment, largeFrequency, Charlie, Bob)
    paymentCD1 = simulation_1.Payment(largePayment, largeFrequency, Bob, Donna)
    paymentAD.setTransfer(paymentAD1)
    paymentCD.setTransfer(paymentCD1)

    channelAB = simulation_1.Channel(Alice, Bob, network)
    channelBC = simulation_1.Channel(Bob, Charlie, network)
    channelBD = simulation_1.Channel(Bob, Donna, network)

    channelAB.addPaymentList([paymentAB, paymentAD])
    channelBC.addPaymentList([paymentBC, paymentCD])
    channelBD.addPaymentList([paymentBD, paymentAD1, paymentCD1])

    network.addPaymentList([paymentAB, paymentBC, paymentAD, paymentBD, paymentCD])

    network.runNetwork()

    a = Alice.getChCostTotal()
    b = Bob.getChCostTotal()
    c = Charlie.getChCostTotal()
    d = Donna.getChCostTotal()

    return (a, b, c, d)

def networkTri(p, freq, onlineTX, onlineTXTime, r, timeRun):
    # Triangle with c extension
    network = simulation_1.Network(onlineTX, onlineTXTime, r, timeRun)
    
    Alice = simulation_1.Node("Alice", network)
    Bob = simulation_1.Node("Bob", network)
    Charlie = simulation_1.Node("Charlie", network)
    Donna = simulation_1.Node("Donna", network)

    paymentAB = simulation_1.Payment(largePayment, largeFrequency, Alice, Bob)
    paymentBC = simulation_1.Payment(largePayment, largeFrequency, Bob, Charlie)
    paymentAD = simulation_1.Payment(freq, p, Alice, Donna)
    paymentBD = simulation_1.Payment(largePayment, largeFrequency, Bob, Donna)
    paymentCD = simulation_1.Payment(largePayment, largeFrequency, Charlie, Bob)
    paymentCD1 = simulation_1.Payment(largePayment, largeFrequency, Bob, Donna)
    paymentCD.setTransfer(paymentCD1)

    channelAB = simulation_1.Channel(Alice, Bob, network)
    channelBC = simulation_1.Channel(Bob, Charlie, network)
    channelBD = simulation_1.Channel(Bob, Donna, network)
    # added direct channel for payment AD
    channelAD = simulation_1.Channel(Alice, Donna, network)

    channelAB.addPaymentList([paymentAB])
    channelBC.addPaymentList([paymentBC, paymentCD])
    channelAD.addPaymentList([paymentAD])
    channelBD.addPaymentList([paymentBD, paymentCD1])

    network.addNodeList([Alice, Bob, Charlie, Donna])
    network.addChannelList([channelAB, channelBC, channelAD, channelBD])
    network.addPaymentList([paymentAB, paymentBC, paymentAD, paymentBD, paymentCD])

    network.runNetwork()

    a = Alice.getChCostTotal()
    b = Bob.getChCostTotal()
    c = Charlie.getChCostTotal()
    d = Donna.getChCostTotal()

    return (a, b, c, d)


def networkCycle(p, freq, onlineTX, onlineTXTime, r, timeRun):
    # diagonal network    
    network = simulation_1.Network(onlineTX, onlineTXTime, r, timeRun)
    
    Alice = simulation_1.Node("Alice", network)
    Bob = simulation_1.Node("Bob", network)
    Charlie = simulation_1.Node("Charlie", network)
    Donna = simulation_1.Node("Donna", network)

    paymentAB = simulation_1.Payment(largePayment, largeFrequency, Alice, Bob)
    paymentBC = simulation_1.Payment(largePayment, largeFrequency, Bob, Charlie)
    paymentAD = simulation_1.Payment(freq, p, Alice, Donna)
    # paymentBD = simulation_1.Payment(0.5, 1, Bob, Charlie)
    # paymentBD1 = simulation_1.Payment(0.5, 1, Charlie, Donna)
    paymentCD = simulation_1.Payment(largePayment, largeFrequency, Charlie, Donna)
    # paymentBD.setTransfer(paymentBD1)

    channelAB = simulation_1.Channel(Alice, Bob, network)
    channelBC = simulation_1.Channel(Bob, Charlie, network)
    channelCD = simulation_1.Channel(Charlie, Donna, network)
    channelAD = simulation_1.Channel(Charlie, Donna, network)

    channelAB.addPaymentList([paymentAB])
    # channelBC.addPaymentList([paymentBC, paymentBD])
    channelBC.addPaymentList([paymentBC])
    # channelCD.addPaymentList([paymentCD, paymentBD1])
    channelCD.addPaymentList([paymentCD])
    channelAD.addPaymentList([paymentAD])

    network.addNodeList([Alice, Bob, Charlie, Donna])
    network.addChannelList([channelAB, channelBC, channelCD, channelAD])
    # network.addPaymentList([paymentAB, paymentBC, paymentAD, paymentBD, paymentCD])
    network.addPaymentList([paymentAB, paymentBC, paymentAD, paymentCD])

    network.runNetwork()

    a = Alice.getChCostTotal()
    b = Bob.getChCostTotal()
    c = Charlie.getChCostTotal()
    d = Donna.getChCostTotal()

    return (a, b, c, d)


def networkLine(p, freq, onlineTX, onlineTXTime, r, timeRun):
    # the network as a line A-B-C-D
    network = simulation_1.Network(onlineTX, onlineTXTime, r, timeRun)
    
    Alice = simulation_1.Node("Alice", network)
    Bob = simulation_1.Node("Bob", network)
    Charlie = simulation_1.Node("Charlie", network)
    Donna = simulation_1.Node("Donna", network)

    paymentAB = simulation_1.Payment(largePayment, largeFrequency, Alice, Bob)
    paymentBC = simulation_1.Payment(largePayment, largeFrequency, Bob, Charlie)
    paymentAD = simulation_1.Payment(freq, p, Alice, Bob)
    paymentAD1 = simulation_1.Payment(freq, p, Bob, Charlie)
    paymentAD2 = simulation_1.Payment(freq, p, Charlie, Donna)
    # paymentBD = simulation_1.Payment(0.5, 1, Bob, Charlie)
    # paymentBD1 = simulation_1.Payment(0.5, 1, Charlie, Donna)
    paymentCD = simulation_1.Payment(largePayment, largeFrequency, Charlie, Donna)
    paymentAD.setTransfer(paymentAD1)
    paymentAD1.setTransfer(paymentAD2)
    # paymentBD.setTransfer(paymentBD1)

    channelAB = simulation_1.Channel(Alice, Bob, network)
    channelBC = simulation_1.Channel(Bob, Charlie, network)
    channelCD = simulation_1.Channel(Charlie, Donna, network)

    channelAB.addPaymentList([paymentAB, paymentAD])
    # channelBC.addPaymentList([paymentBC, paymentAD1, paymentBD])
    channelBC.addPaymentList([paymentBC, paymentAD1])
    # channelCD.addPaymentList([paymentCD, paymentBD1, paymentAD2])
    channelCD.addPaymentList([paymentCD, paymentAD2])

    network.addNodeList([Alice, Bob, Charlie, Donna])
    network.addChannelList([channelAB, channelBC, channelCD])
    # network.addPaymentList([paymentAB, paymentBC, paymentAD, paymentBD, paymentCD])
    network.addPaymentList([paymentAB, paymentBC, paymentAD, paymentCD])

    network.runNetwork()

    a = Alice.getChCostTotal()
    b = Bob.getChCostTotal()
    c = Charlie.getChCostTotal()
    d = Donna.getChCostTotal()

    return (a, b, c, d)


def networkStar2(p, freq, onlineTX, onlineTXTime, r, timeRun):
    # a star / fork network
    network = simulation_1.Network(onlineTX, onlineTXTime, r, timeRun)
    
    Alice = simulation_1.Node("Alice", network)
    Bob = simulation_1.Node("Bob", network)
    Charlie = simulation_1.Node("Charlie", network)
    Donna = simulation_1.Node("Donna", network)

    paymentAB = simulation_1.Payment(largePayment, largeFrequency, Alice, Bob)
    paymentBC = simulation_1.Payment(largePayment, largeFrequency, Bob, Charlie)
    paymentAD = simulation_1.Payment(freq, p, Alice, Bob)
    paymentAD1 = simulation_1.Payment(freq, p, Bob, Donna)
    paymentBD = simulation_1.Payment(largePayment, largeFrequency, Bob, Donna)
    paymentAD.setTransfer(paymentAD1)

    channelAB = simulation_1.Channel(Alice, Bob, network)
    channelBC = simulation_1.Channel(Bob, Charlie, network)
    channelBD = simulation_1.Channel(Bob, Donna, network)

    channelAB.addPaymentList([paymentAB, paymentAD])
    channelBC.addPaymentList([paymentBC])
    channelBD.addPaymentList([paymentBD, paymentAD1])

    network.addNodeList([Alice, Bob, Charlie, Donna])
    network.addChannelList([channelAB, channelBC, channelBD])
    network.addPaymentList([paymentAB, paymentBC, paymentAD, paymentBD])

    network.runNetwork()

    a = Alice.getChCostTotal()
    b = Bob.getChCostTotal()
    c = Charlie.getChCostTotal()
    d = Donna.getChCostTotal()

    return (a, b, c, d)

def networkTri2(p, freq, onlineTX, onlineTXTime, r, timeRun):
    # Triangle with c extension
    network = simulation_1.Network(onlineTX, onlineTXTime, r, timeRun)
    
    Alice = simulation_1.Node("Alice", network)
    Bob = simulation_1.Node("Bob", network)
    Charlie = simulation_1.Node("Charlie", network)
    Donna = simulation_1.Node("Donna", network)

    paymentAB = simulation_1.Payment(largePayment, largeFrequency, Alice, Bob)
    paymentBC = simulation_1.Payment(largePayment, largeFrequency, Bob, Charlie)
    paymentAD = simulation_1.Payment(freq, p, Alice, Donna)
    paymentBD = simulation_1.Payment(largePayment, largeFrequency, Bob, Donna)

    channelAB = simulation_1.Channel(Alice, Bob, network)
    channelBC = simulation_1.Channel(Bob, Charlie, network)
    channelBD = simulation_1.Channel(Bob, Donna, network)
    # added direct channel for payment AD
    channelAD = simulation_1.Channel(Alice, Donna, network)

    channelAB.addPaymentList([paymentAB])
    channelBC.addPaymentList([paymentBC])
    channelAD.addPaymentList([paymentAD])
    channelBD.addPaymentList([paymentBD])

    network.addNodeList([Alice, Bob, Charlie, Donna])
    network.addChannelList([channelAB, channelBC, channelAD, channelBD])
    network.addPaymentList([paymentAB, paymentBC, paymentAD, paymentBD])

    network.runNetwork()

    a = Alice.getChCostTotal()
    b = Bob.getChCostTotal()
    c = Charlie.getChCostTotal()
    d = Donna.getChCostTotal()

    return (a, b, c, d)

############# single trial #################
def setUp4D(p=0.1, freq=0.5, onlineTX = 5.0, onlineTXTime = 1.0, r = 0.01, timeRun = 10.0):
    simulation_res = []
    simulation_res.append(networkStar(p, freq, onlineTX, onlineTXTime, r, timeRun))
    simulation_res.append(networkCycle(p, freq, onlineTX, onlineTXTime, r, timeRun))
    simulation_res.append(networkLine(p, freq, onlineTX, onlineTXTime, r, timeRun))
    simulation_res.append(networkLineOppoDir(p, freq, onlineTX, onlineTXTime, r, timeRun))

    return simulation_res

def setUpPF_star(p=0.1, freq=0.5, onlineTX = 5.0, onlineTXTime = 1.0, r = 0.01, timeRun = 10.0):
    simulation_res = []
    tempStar = networkStar(p, freq, onlineTX, onlineTXTime, r, timeRun)
    tempTri = networkTri(p, freq, onlineTX, onlineTXTime, r, timeRun)
    # alice and donna, and bob is affected by the direct channel, charlie does not matter
    simulation_res.append((tempStar[0]+tempStar[2], tempStar[1]))
    simulation_res.append((tempTri[0]+tempTri[2], tempTri[1]))

    return simulation_res

def setUpPF_cycle(p=0.1, freq=0.5, onlineTX = 5.0, onlineTXTime = 1.0, r = 0.01, timeRun = 10.0):
    simulation_res = []
    tempCycle = networkCycle(p, freq, onlineTX, onlineTXTime, r, timeRun)
    tempLine = networkLine(p, freq, onlineTX, onlineTXTime, r, timeRun)
    simulation_res.append((tempLine[0]+ tempLine[1], tempLine[2]+tempLine[3]))
    simulation_res.append((tempCycle[0]+tempCycle[1], tempCycle[2]+tempCycle[3]))

    return simulation_res

def setUpPF_star2(p=0.1, freq=0.5, onlineTX = 5.0, onlineTXTime = 1.0, r = 0.01, timeRun = 10.0):
    simulation_res = []
    tempStar = networkStar2(p, freq, onlineTX, onlineTXTime, r, timeRun)
    tempTri = networkTri2(p, freq, onlineTX, onlineTXTime, r, timeRun)
    # alice and donna, and bob is affected by the direct channel, charlie does not matter
    simulation_res.append((tempStar[0]+tempStar[2], tempStar[1]))
    simulation_res.append((tempTri[0]+tempTri[2], tempTri[1]))

    return simulation_res

############# Main functions #####################
def graphPF_Star(p=0.1, freq=0.5, onlineTX = 5.0, onlineTXTime = 1.0, r = 0.01, timeRun = 10.0):
    # 0.2 to 1.2    
    ps = [x* psmultiply /psdivision for x in range(psInit, psInit+psLen)]
    # ps = [x* 1.0 / 1 for x in range(1, 30)]
    # 0.25 to 1.25
    fs = [x* fsmultiply /fsdivision for x in range(fsInit, fsInit+fsLen)]

    costOnPay = [[0 for x in range(len(ps))] for x in range(0, 4)]
    costOnFreq = [[0 for x in range(len(fs))] for x in range(0, 4)]
    
    for i in range(len(fs)):
        print(i)
        for j in range(len(ps)):
            print("-star %d-%d" %(i, j))
            res = [0 for x in range(2)] 
            
            for k in range(num_trial):
                # get the average of the trial given certain freq and p
                tmp = setUpPF_star(p=ps[j], freq=fs[i], timeRun = timeRun)
                temp = [tmp[1][0]-tmp[0][0], tmp[0][1]-tmp[1][1]]

                for h in range(len(temp)):
                    res[h] += temp[h]
                for h in range(len(temp)):
                    res[h]= res[h]/float(num_trial)

                    # get cost for certain p and freq regardless of the other parameter
                    costOnPay[h][j] += res[h]
                    costOnFreq[h][i] += res[h]
        # finished generating all p on certain freq, get cost based on freq
    for i in range(len(costOnFreq)):
        for h in range(len(costOnFreq[i])):
            costOnFreq[i][h] /= len(fs)
    for i in range(len(costOnPay)):
        for h in range(len(costOnPay[i])):
            # finished generting all freq on certain p
            costOnPay[i][h] /= len(ps)
    

    fig = plt.figure(figsize=[6.4, 12.8])

    # based on payment
    ax = fig.add_subplot(2,1, 1)
    ax.set_xlabel('Payment size')
    ax.set_ylabel('benefit')
    ax.set_title('benefit of the transfer in star vs payment size')

    # surf = ax.plot(ps, costOnPay[0][0], "r-.")
    # surf = ax.plot(ps, costOnPay[0][1], "b--")
    # surf = ax.plot(ps, costOnPay[1][0], "g--")
    # surf = ax.plot(ps, costOnPay[1][1], "k-.")
    for i in range(0, 2):
        surf = ax.plot(ps, costOnPay[i])

    ax.legend(["alice", "bob"]) 

    # based on frequency
    ax = fig.add_subplot(2,1, 2)
    ax.set_xlabel('frequency (lambda)')
    ax.set_ylabel('benefit')
    ax.set_title('benefit of the transfer in cycle vs frequency')

    for i in range(0, 2):
        surf = ax.plot(fs, costOnFreq[i])
    
    ax.legend(["alice", "bob"]) 
    fig.text(0, 0, 'trials: %d; Time: %d; psdiv: %d; psmulti: %f; fsdiv: %d; fsmulti: %f' % (num_trial, time, psdivision, psmultiply, fsmultiply, fsdivision))
    fig.savefig('4node_fee_star.png')


def graphPF_Star2(p=0.1, freq=0.5, onlineTX = 5.0, onlineTXTime = 1.0, r = 0.01, timeRun = 10.0):
    # 0.2 to 1.2    
    ps = [x* psmultiply /psdivision for x in range(psInit, psInit+psLen)]
    # ps = [x* 1.0 / 1 for x in range(1, 30)]
    # 0.25 to 1.25
    fs = [x* fsmultiply /fsdivision for x in range(fsInit, fsInit+fsLen)]

    costOnPay = [[0 for x in range(len(ps))] for x in range(0, 4)]
    costOnFreq = [[0 for x in range(len(fs))] for x in range(0, 4)]
    
    for i in range(len(fs)):
        print(i)
        for j in range(len(ps)):
            print("--star %d-%d" %(i, j))
            res = [0 for x in range(2)] 
            
            for k in range(num_trial):
                # get the average of the trial given certain freq and p
                tmp = setUpPF_star2(p=ps[j], freq=fs[i], timeRun = timeRun)
                temp = [tmp[1][0]-tmp[0][0], tmp[0][1]-tmp[1][1]]

                for h in range(len(temp)):
                    res[h] += temp[h]
                for h in range(len(temp)):
                    res[h]= res[h]/float(num_trial)

                    # get cost for certain p and freq regardless of the other parameter
                    costOnPay[h][j] += res[h]
                    costOnFreq[h][i] += res[h]
        # finished generating all p on certain freq, get cost based on freq
    for i in range(len(costOnFreq)):
        for h in range(len(costOnFreq[i])):
            costOnFreq[i][h] /= len(fs)
    for i in range(len(costOnPay)):
        for h in range(len(costOnPay[i])):
            # finished generting all freq on certain p
            costOnPay[i][h] /= len(ps)
    

    fig = plt.figure(figsize=[6.4, 12.8])

    # based on payment
    ax = fig.add_subplot(2,1, 1)
    ax.set_xlabel('Payment size')
    ax.set_ylabel('benefit')
    ax.set_title('benefit of the transfer in star vs payment size')

    for i in range(0, 2):
        surf = ax.plot(ps, costOnPay[i])

    ax.legend(["alice", "bob"]) 

    # based on frequency
    ax = fig.add_subplot(2,1, 2)
    ax.set_xlabel('frequency (lambda)')
    ax.set_ylabel('benefit')
    ax.set_title('benefit of the transfer in cycle vs frequency')

    for i in range(0, 2):
        surf = ax.plot(fs, costOnFreq[i])
    
    ax.legend(["alice", "bob"]) 
    

    fig.text(0, 0, 'trials: %d; Time: %d; pD: %d; pM: %f; fD: %d; fM: %f' % (num_trial, time, psdivision, psmultiply, fsdivision, fsmultiply))
    fig.savefig('4node_fee_star2_0.png')



def graphPF_Cycle(p=0.1, freq=0.5, onlineTX = 5.0, onlineTXTime = 1.0, r = 0.01, timeRun = 10.0):
    # 0.2 to 1.2    
    ps = [x* psmultiply /psdivision for x in range(psInit, psInit+psLen)]
    # ps = [x* 1.0 / 1 for x in range(1, 30)]
    # 0.25 to 1.25
    fs = [x* fsmultiply /fsdivision for x in range(fsInit, fsInit+fsLen)]

    costOnPay = [[0 for x in range(len(ps))] for x in range(0, 2)]
    costOnFreq = [[0 for x in range(len(fs))] for x in range(0, 2)]
    for i in range(len(fs)):
        print(i)
        for j in range(len(ps)):
            print("-cycle %d-%d" %(i, j))
            res = [0 for x in range(0, 2)]

            for k in range(num_trial):
                # get the average of the trial given certain freq and p
                tmp = setUpPF_cycle(p=ps[j], freq=fs[i], timeRun = timeRun)
                temp = [tmp[0][0]-tmp[1][0], tmp[1][1]-tmp[0][1]]

                for h in range(len(temp)):
                    res[h] += temp[h]
                for h in range(len(temp)):
                    res[h]= res[h]/float(num_trial)

                    # # get cost for certain p and freq regardless of the other parameter
                    # print(costOnPay[h][j])
                    # print(res[h])
                    costOnPay[h][j] += res[h]
                    costOnFreq[h][i] += res[h]
                                    # costOnPay[j] += res[h]
                # costOnFreq[i] += res[h]
        
    # get average to avoid counting for payment size
    for i in range(len(costOnFreq)):
        for h in range(len(costOnFreq[i])):
            costOnFreq[i][h] /= len(fs)
    for i in range(len(costOnPay)):
        for h in range(len(costOnPay[i])):
            # finished generting all freq on certain p
            costOnPay[i][h] /= len(ps)
    

    fig = plt.figure(figsize=[6.4, 12.8])

    # based on payment
    ax = fig.add_subplot(2,1, 1)
    ax.set_xlabel('Payment size')
    ax.set_ylabel('benefit')
    ax.set_title('benefit of the transfer in cycle vs payment size')

    for i in range(0, 2):
        surf = ax.plot(ps, costOnPay[i])

    ax.legend(["alice", "bob"]) 

    # based on frequency
    ax = fig.add_subplot(2,1, 2)
    ax.set_xlabel('frequency (lambda)')
    ax.set_ylabel('benefit')
    ax.set_title('benefit of the transfer in cycle vs frequency')

    for i in range(0, 2):
        surf = ax.plot(fs, costOnFreq[i])
    
    ax.legend(["alice", "bob"]) 

    fig.text(0, 0, 'trials: %d; Time: %d; psdiv: %d; psmulti: %f; fsdiv: %d; fsmulti: %f' % (num_trial, time, psdivision, psmultiply, fsmultiply, fsdivision))
    fig.savefig('4node_fee_cycle.png')


############# Main functions #####################

def run(time):
    
        # trial
        
    temp = []
    temp = networkOG(littleP, littleF, onlineTX=5, onlineTXTime=3, r=0.01, timeRun=time)
    alice0.append(temp[0])
    bob0.append(temp[1])
    
    temp = networkDirectAC(littleP, littleF, onlineTX=5, onlineTXTime=3, r=0.01, timeRun=time)
    alice1.append(temp[0])
    bob1.append(temp[1])

    temp = networktransferB(littleP, littleF, onlineTX=5, onlineTXTime=3, r=0.01, timeRun=time)
    alice2.append(temp[0])
    bob2.append(temp[1])

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

    print("max fee")
    print(maxFee)
    print("\n")

    print("alice escapes")
    print(chargeFee(alice2, alice0))
    print("Bob22")
    print(bob22)
    print("min fee")
    print(minFee)

    print("maxfee " + str(maxFee) + "; minfee " + str(minFee))
    print("param pay %f ; freq %f" % (littleP, littleF))




################ Call #####################

run(time)
# runWithFreq()

# getFees(0.3, 0.1, time)

