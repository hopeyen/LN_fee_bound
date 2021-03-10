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


############## Constants and global variables ################

num_trial = 10

time = 2

givenP = 0.5

paymentMean = 0.1

paymentSigma = 0.1

psInit = 20

fsInit = 25

psLen = 100

fsLen = 100

psIncre = 10

fsIncre = 10

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

    channelAB = simulation_1.Channel(Alice, Bob, 5, 0, network)
    channelBC = simulation_1.Channel(Bob, Charlie, 20, 20, network)
    channelBD = simulation_1.Channel(Bob, Donna, 20, 20, network)

    channelAB.addPaymentList([paymentAB, paymentAD])
    channelBC.addPaymentList([paymentBC, paymentBD, paymentCD])
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


def networkOG(p, freq, onlineTX, onlineTXTime, r, timeRun):
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

def networkLineOppoDir(p, freq, onlineTX, onlineTXTime, r, timeRun):
    # network as a line D-A-B-C
    network = simulation_1.Network(onlineTX, onlineTXTime, r, timeRun)
    
    Alice = simulation_1.Node("Alice", network)
    Bob = simulation_1.Node("Bob", network)
    Charlie = simulation_1.Node("Charlie", network)
    Donna = simulation_1.Node("Donna", network)

    paymentAB = simulation_1.Payment(0.5, 1, Alice, Bob)
    paymentBC = simulation_1.Payment(0.5, 1, Bob, Charlie)
    paymentAD = simulation_1.Payment(freq, p, Alice, Donna)
    paymentBD = simulation_1.Payment(0.5, 1, Bob, Alice)
    paymentBD1 = simulation_1.Payment(0.5, 1, Alice, Donna)
    paymentCD = simulation_1.Payment(0.5, 1, Charlie, Bob)
    paymentCD1 = simulation_1.Payment(0.5, 1, Bob, Alice)
    paymentCD2 = simulation_1.Payment(0.5, 1, Alice, Donna)

    channelAB = simulation_1.Channel(Alice, Bob, 5, 0, network)
    channelBC = simulation_1.Channel(Bob, Charlie, 20, 20, network)
    channelAD = simulation_1.Channel(Alice, Donna, 20, 20, network)

    channelAB.addPaymentList([paymentAB, paymentBD, paymentCD1])
    channelBC.addPaymentList([paymentBC, paymentCD])
    channelAD.addPaymentList([paymentAD, paymentBD1, paymentCD2])

    network.addNodeList([Alice, Bob, Charlie, Donna])
    network.addChannelList([channelAB, channelBC, channelAD])
    network.addPaymentList([paymentAB, paymentBC, paymentAD, paymentBD, paymentCD])

    network.runNetwork()

    a = Alice.getChCostTotal()
    b = Bob.getChCostTotal()
    c = Charlie.getChCostTotal()
    d = Donna.getChCostTotal()

    return (a, b, c, d)


############## Constants and global variables ################


num_trial = 10

time = 2

givenP = 0.5

paymentMean = 0

paymentSigma = 0.1

psInit = 1

fsInit = 1

psLen = 200

fsLen = 200


############# single trial #################
def setUp(p=0.1, freq=0.5, onlineTX = 5.0, onlineTXTime = 1.0, r = 0.01, timeRun = 10.0):
    simulation_res = []
    simulation_res.append(networkStar(p, freq, onlineTX, onlineTXTime, r, timeRun))
    simulation_res.append(networkOG(p, freq, onlineTX, onlineTXTime, r, timeRun))
    simulation_res.append(networkLine(p, freq, onlineTX, onlineTXTime, r, timeRun))
    simulation_res.append(networkLineOppoDir(p, freq, onlineTX, onlineTXTime, r, timeRun))

    return simulation_res


############# Main functions #####################
def graph4D(p=0.1, freq=0.5, onlineTX = 5.0, onlineTXTime = 1.0, r = 0.01, timeRun = 10.0):
    # 0.2 to 1.2
    ps = [x* 1.0 /10 for x in range(psInit, psInit+psLen)]
    # 0.25 to 1.25
    fs = [x* 1.0 /10 for x in range(fsInit, fsInit+fsLen)]

    X = np.array(ps)
    Y = np.array(fs)
    X, Y = np.meshgrid(X, Y)

    costs = [[[[0 for x in range(psInit, psInit+psLen)] for x in range(fsInit, fsInit+fsLen)] for x in range(0, 4)] for x in range(0, 4)]

    for i in range(len(fs)):

        for j in range(len(ps)):
            print("%d--%d" %(i, j))
            
            for k in range(num_trial):

                res = setUp(p=ps[j], freq=fs[i], timeRun = timeRun)
                for h in range(len(res)):
                    for l in range(len(res[h])):
                        costs[h][l][i][j] += res[l][h]

            for h in range(len(costs)):
                for l in range(len(costs[h])):
                    costs[h][l][i][j] /= float(num_trial)
    

    fig = plt.figure(figsize=plt.figaspect(0.5))

    people = ['Alice ', 'Bob ', 'Charlie ', 'Donna ']
    titles = ['cost of network star', 'cost of network cycle', 'cost of network line', 'cost of network line oppo']

    index = 0
    Z = np.array(costs)
    for i in range(0, 4):
        for j in range(0, 4):
            index += 1

            ax = fig.add_subplot(4, 4, index, projection='3d')
            ax.set_xlabel('Payment size')
            ax.set_ylabel('Frequency')
            ax.set_zlabel("Channel costs")
            ax.set_title(people[i] + titles[j])
            surf = ax.plot_surface(X, Y, Z[i][j], rstride=1, cstride=1, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
            fig.colorbar(surf, shrink=0.5, aspect=10)

            fig.savefig('4D.png')

    # fig.savefig('3D-3.png')
def graph3D(p=0.1, freq=0.5, onlineTX = 5.0, onlineTXTime = 1.0, r = 0.01, timeRun = 10.0):
    # 0.2 to 1.2
    ps = [x* 1.0 /100 for x in range(psInit, psInit+psLen)]
    # 0.25 to 1.25
    fs = [x* 1.0 /100 for x in range(fsInit, fsInit+fsLen)]

    costOnPay = [[[0 for x in range(len(ps))] for x in range(0, 4)] for x in range(0, 4)]
    costOnFreq = [[[0 for x in range(len(fs))] for x in range(0, 4)] for x in range(0, 4)]
    avg = []
    for i in range(len(fs)):
        print(i)
        for j in range(len(ps)):
            # print("%d--%d" %(i, j))
            res = [[0 for x in range(4)] for x in range(4)]
            avg = []
            temp = 0
            for k in range(num_trial):
                # get the average of the trial given certain freq and p
                temp = setUp(p=ps[j], freq=fs[i], timeRun = timeRun)
                for h in range(len(temp)):
                    for l in range(len(temp[h])):
                        res[h][l] += temp[h][l]
                for h in range(len(res)):
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
    

    fig = plt.figure(figsize=[12.8, 12.8])

    people = ['Alice ', 'Bob ', 'Charlie ', 'Donna ']
    titles = ['cost of star', 'cycle', 'line', 'oppo-line']

    # based on payment
    index = 0
    for i in range(0, 4):
        for j in range(0, 4):
            index += 1
            ax = fig.add_subplot(4, 4, index)
            ax.set_xlabel('Payment size')
            ax.set_ylabel('Channel costs')
            ax.set_title(people[i] + titles[j])
            surf = ax.plot(ps, costOnPay[i][j])
            fig.savefig('4node_payment.png')

    # based on frequency
    index = 0
    for i in range(0, 4):
        for j in range(0, 4):
            index += 1

            ax = fig.add_subplot(4, 4, index)
            ax.set_xlabel('Frequency (lambda)')
            ax.set_ylabel('Channel costs')
            ax.set_title(people[i] + titles[j])
            surf = ax.plot(ps, costOnFreq[i][j])
            fig.savefig('4node_frequency.png')

################ Call #####################





graph3D(2)
# runWithFreq()




