import simulation_1 
import random
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import copy
import helpersMatrix


############### Set up ###################

def networkStar(p, freq, onlineTX, onlineTXTime, r, timeRun):
    # a star / fork network
    network = simulation_1.Network(onlineTX, onlineTXTime, r, timeRun)
    
    Alice = simulation_1.Node("Alice", network)
    Bob = simulation_1.Node("Bob", network)
    Charlie = simulation_1.Node("Charlie", network)
    Donna = simulation_1.Node("Donna", network)

    paymentAB = simulation_1.Payment(0.5, 1, Alice, Bob)
    paymentBC = simulation_1.Payment(0.5, 1, Bob, Charlie)
    paymentAD = simulation_1.Payment(freq, p, Alice, Bob)
    paymentAD1 = simulation_1.Payment(freq, p, Bob, Donna)
    paymentBD = simulation_1.Payment(0.5, 1, Bob, Donna)
    paymentCD = simulation_1.Payment(0.5, 1, Charlie, Bob)
    paymentCD1 = simulation_1.Payment(0.5, 1, Bob, Donna)
    paymentAD.setTransfer(paymentAD1)
    paymentCD.setTransfer(paymentCD1)

    channelAB = simulation_1.Channel(Alice, Bob, 5, 0, network)
    channelBC = simulation_1.Channel(Bob, Charlie, 20, 20, network)
    channelBD = simulation_1.Channel(Bob, Donna, 20, 20, network)

    channelAB.addPaymentList([paymentAB, paymentAD])
    channelBC.addPaymentList([paymentBC, paymentCD])
    channelBD.addPaymentList([paymentBD, paymentAD1, paymentCD1])

    network.addNodeList([Alice, Bob, Charlie, Donna])
    network.addChannelList([channelAB, channelBC, channelBD])
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

    paymentAB = simulation_1.Payment(0.5, 1, Alice, Bob)
    paymentBC = simulation_1.Payment(0.5, 1, Bob, Charlie)
    paymentAD = simulation_1.Payment(freq, p, Alice, Donna)
    paymentBD = simulation_1.Payment(0.5, 1, Bob, Donna)
    paymentCD = simulation_1.Payment(0.5, 1, Charlie, Bob)
    paymentCD1 = simulation_1.Payment(0.5, 1, Bob, Donna)
    paymentCD.setTransfer(paymentCD1)

    channelAB = simulation_1.Channel(Alice, Bob, 20, 20, network)
    channelBC = simulation_1.Channel(Bob, Charlie, 20, 20, network)
    channelBD = simulation_1.Channel(Bob, Donna, 20, 20, network)
    # added direct channel for payment AD
    channelAD = simulation_1.Channel(Alice, Donna, 5, 0, network)

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

    paymentAB = simulation_1.Payment(0.5, 1, Alice, Bob)
    paymentBC = simulation_1.Payment(0.5, 1, Bob, Charlie)
    paymentAD = simulation_1.Payment(freq, p, Alice, Donna)
    paymentBD = simulation_1.Payment(0.5, 1, Bob, Charlie)
    paymentBD1 = simulation_1.Payment(0.5, 1, Charlie, Donna)
    paymentCD = simulation_1.Payment(0.5, 1, Charlie, Donna)
    paymentBD.setTransfer(paymentBD1)

    channelAB = simulation_1.Channel(Alice, Bob, 5, 0, network)
    channelBC = simulation_1.Channel(Bob, Charlie, 20, 20, network)
    channelCD = simulation_1.Channel(Charlie, Donna, 20, 20, network)
    channelAD = simulation_1.Channel(Charlie, Donna, 20, 20, network)

    channelAB.addPaymentList([paymentAB])
    channelBC.addPaymentList([paymentBC, paymentBD])
    channelCD.addPaymentList([paymentCD, paymentBD1])
    channelAD.addPaymentList([paymentAD])

    network.addNodeList([Alice, Bob, Charlie, Donna])
    network.addChannelList([channelAB, channelBC, channelCD, channelAD])
    network.addPaymentList([paymentAB, paymentBC, paymentAD, paymentBD, paymentCD])

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

    paymentAB = simulation_1.Payment(0.5, 1, Alice, Bob)
    paymentBC = simulation_1.Payment(0.5, 1, Bob, Charlie)
    paymentAD = simulation_1.Payment(freq, p, Alice, Bob)
    paymentAD1 = simulation_1.Payment(freq, p, Bob, Charlie)
    paymentAD2 = simulation_1.Payment(freq, p, Charlie, Donna)
    paymentBD = simulation_1.Payment(0.5, 1, Bob, Charlie)
    paymentBD1 = simulation_1.Payment(0.5, 1, Charlie, Donna)
    paymentCD = simulation_1.Payment(0.5, 1, Charlie, Donna)
    paymentAD.setTransfer(paymentAD1)
    paymentAD1.setTransfer(paymentAD2)
    paymentBD.setTransfer(paymentBD1)

    channelAB = simulation_1.Channel(Alice, Bob, 5, 0, network)
    channelBC = simulation_1.Channel(Bob, Charlie, 20, 20, network)
    channelCD = simulation_1.Channel(Charlie, Donna, 20, 20, network)

    channelAB.addPaymentList([paymentAB, paymentAD])
    channelBC.addPaymentList([paymentBC, paymentAD1, paymentBD])
    channelCD.addPaymentList([paymentCD, paymentBD1, paymentAD2])

    network.addNodeList([Alice, Bob, Charlie, Donna])
    network.addChannelList([channelAB, channelBC, channelCD])
    network.addPaymentList([paymentAB, paymentBC, paymentAD, paymentBD, paymentCD])

    network.runNetwork()

    a = Alice.getChCostTotal()
    b = Bob.getChCostTotal()
    c = Charlie.getChCostTotal()
    d = Donna.getChCostTotal()

    return (a, b, c, d)


############## Constants and global variables ################


num_trial = 100

time = 2

givenP = 0.5

paymentMean = 0

paymentSigma = 0.1


division = 100

psInit = 1

psLen = 200

fsInit = 50

fsLen = 50



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
    simulation_res.append((tempCycle[0]+tempCycle[2]+tempCycle[3], tempCycle[1]))
    simulation_res.append((tempLine[0]+tempLine[2]+tempLine[3], tempLine[1]))

    return simulation_res


############# Main functions #####################
def graphPF_Star(p=0.1, freq=0.5, onlineTX = 5.0, onlineTXTime = 1.0, r = 0.01, timeRun = 10.0):
    # 0.2 to 1.2
    ps = [x* 1.0 /division for x in range(psInit, psInit+psLen)]
    # 0.25 to 1.25
    fs = [x* 1.0 /division for x in range(fsInit, fsInit+fsLen)]

    costOnPay = [[[0 for x in range(len(ps))] for x in range(0, 4)] for x in range(0, 4)]
    costOnFreq = [[[0 for x in range(len(fs))] for x in range(0, 4)] for x in range(0, 4)]
    avg = []
    for i in range(len(fs)):
        print(i)
        for j in range(len(ps)):
            print("star %d-%d" %(i, j))
            res = [[0 for x in range(4)] for x in range(4)]
            avg = []
            temp = 0
            for k in range(num_trial):
                # get the average of the trial given certain freq and p
                temp = setUpPF_star(p=ps[j], freq=fs[i], timeRun = timeRun)

                for h in range(len(temp)):
                    for l in range(len(temp[h])):
                        res[h][l] += temp[h][l]
                for h in range(len(temp)):
                    for l in range(len(temp[h])):
                        res[h][l] = res[h][l]/float(num_trial)

                    # get cost for certain p and freq regardless of the other parameter
                        costOnPay[h][l][j] += res[h][l]
                        costOnFreq[h][l][i] += res[h][l]
        # finished generating all p on certain freq, get cost based on freq
        costOnFreq[h][l][i] /= len(ps)
    for i in range(len(ps)):
        # finished generting all freq on certain p
        costOnPay[h][l][i] /= len(fs)
    

    fig = plt.figure(figsize=[6.4, 12.8])

    people = ['Alice ', 'Bob ']
    titles = ['cost of star', 'line']

    # based on payment
    ax = fig.add_subplot(2,1, 1)
    ax.set_xlabel('Payment size')
    ax.set_ylabel('costs')
    ax.set_title('Cost of channels in star')

    # surf = ax.plot(ps, costOnPay[0][0], "r-.")
    # surf = ax.plot(ps, costOnPay[0][1], "b--")
    # surf = ax.plot(ps, costOnPay[1][0], "g--")
    # surf = ax.plot(ps, costOnPay[1][1], "k-.")
    for i in range(0, 2):
        for j in range(0, 2):
            surf = ax.plot(ps, costOnPay[i][j])

    ax.legend(["alice star", "bob star", "alice direct", "bob direct"]) 

    # based on frequency
    ax = fig.add_subplot(2,1, 2)
    ax.set_xlabel('frequency (lambda)')
    ax.set_ylabel('costs')
    ax.set_title('Cost of channels in cycle')

    for i in range(0, 2):
        for j in range(0, 2):
            surf = ax.plot(fs, costOnFreq[i][j])
    
    ax.legend(["alice star", "bob star", "alice direct", "bob direct"]) 
    fig.savefig('4node_star_payment.png')



def graphPF_Cycle(p=0.1, freq=0.5, onlineTX = 5.0, onlineTXTime = 1.0, r = 0.01, timeRun = 10.0):
    # 0.2 to 1.2
    ps = [x* 1.0 /division for x in range(psInit, psInit+psLen)]
    # 0.25 to 1.25
    fs = [x* 1.0 /division for x in range(fsInit, fsInit+fsLen)]

    costOnPay = [[[0 for x in range(len(ps))] for x in range(0, 4)] for x in range(0, 4)]
    costOnFreq = [[[0 for x in range(len(fs))] for x in range(0, 4)] for x in range(0, 4)]
    avg = []
    for i in range(len(fs)):
        print(i)
        for j in range(len(ps)):
            print("cycle %d-%d" %(i, j))
            res = [[0 for x in range(4)] for x in range(4)]
            avg = []
            temp = 0
            for k in range(num_trial):
                # get the average of the trial given certain freq and p
                temp = setUpPF_cycle(p=ps[j], freq=fs[i], timeRun = timeRun)
                for h in range(len(temp)):
                    for l in range(len(temp[h])):
                        res[h][l] += temp[h][l]
                for h in range(len(temp)):
                    for l in range(len(temp[h])):
                        res[h][l] = res[h][l]/float(num_trial)

                    # get cost for certain p and freq regardless of the other parameter
                        costOnPay[h][l][j] += res[h][l]
                        costOnFreq[h][l][i] += res[h][l]
        # finished generating all p on certain freq, get cost based on freq
        costOnFreq[h][l][i] /= len(ps)
    for i in range(len(ps)):
        # finished generting all freq on certain p
        costOnPay[h][l][i] /= len(fs)
    

    fig = plt.figure(figsize=[6.4, 12.8])

    people = ['Alice ', 'Bob ']
    titles = ['cost of star', 'line']

    # based on payment
    ax = fig.add_subplot(2,1, 1)
    ax.set_xlabel('Payment size')
    ax.set_ylabel('costs')
    ax.set_title('Cost of channels vs payment')

    surf = ax.plot(ps, costOnPay[0][0], "r-.")
    surf = ax.plot(ps, costOnPay[0][1], "b--")
    surf = ax.plot(ps, costOnPay[1][0], "g--")
    surf = ax.plot(ps, costOnPay[1][1], "k-.")
    # for i in range(0, 2):
    #     for j in range(0, 2):
    #         surf = ax.plot(ps, costOnPay[i][j])

    ax.legend(["alice direct", "bob direct", "alice cycle", "bob cycle"]) 

    # based on frequency
    ax = fig.add_subplot(2,1, 2)
    ax.set_xlabel('frequency (lambda)')
    ax.set_ylabel('costs')
    ax.set_title('Cost of channels vs frequency')

    for i in range(0, 2):
        for j in range(0, 2):
            surf = ax.plot(fs, costOnFreq[i][j])
    
    ax.legend(["alice direct", "bob direct", "alice cycle", "bob cycle"]) 
    fig.savefig('4node_cycle_payment.png')

################ Call #####################



graphPF_Cycle(20)
graphPF_Star(20)



