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

# network

num_trial = 10

time = 10

# payments

largePayment = 1

psInit = 1

psLen = 50

psdivision = 10

psmultiply = 1.0

# frequency 

largeFrequency = 1

fsInit = 1

fsLen = 50

fsdivision = 100

fsmultiply = 1.0


############### Set up ###################
def networkStarOG(p, freq, onlineTX, onlineTXTime, r, timeRun):
    # a star / fork network
    network = simulation_1.Network(onlineTX, onlineTXTime, r, timeRun)
    
    Alice = simulation_1.Node("Alice", network)
    Bob = simulation_1.Node("Bob", network)
    Charlie = simulation_1.Node("Charlie", network)
    Donna = simulation_1.Node("Donna", network)

    paymentAB = simulation_1.Payment(largeFrequency, largePayment, Alice, Bob)
    paymentBC = simulation_1.Payment(largeFrequency, largePayment, Charlie, Bob)
    paymentBD = simulation_1.Payment(largeFrequency, largePayment, Donna, Bob)
    paymentCD = simulation_1.Payment(largeFrequency, largePayment, Charlie, Bob)
    paymentCD1 = simulation_1.Payment(largeFrequency, largePayment, Bob, Donna)
    paymentCD.setTransfer(paymentCD1)

    channelAB = simulation_1.Channel(Alice, Bob, network)
    channelBC = simulation_1.Channel(Bob, Charlie, network)
    channelBD = simulation_1.Channel(Bob, Donna, network)

    channelAB.addPaymentList([paymentAB])
    channelBC.addPaymentList([paymentBC, paymentCD])
    channelBD.addPaymentList([paymentBD, paymentCD1])

    network.addPaymentList([paymentAB, paymentBC, paymentBD, paymentCD])

    network.runNetwork()

    a = Alice.getChCostTotal()
    b = Bob.getChCostTotal()
    c = Charlie.getChCostTotal()
    d = Donna.getChCostTotal()

    return (a, b, c, d)

def networkStar(p, freq, onlineTX, onlineTXTime, r, timeRun):
    # a star / fork network
    network = simulation_1.Network(onlineTX, onlineTXTime, r, timeRun)
    
    Alice = simulation_1.Node("Alice", network)
    Bob = simulation_1.Node("Bob", network)
    Charlie = simulation_1.Node("Charlie", network)
    Donna = simulation_1.Node("Donna", network)

    paymentAB = simulation_1.Payment(largeFrequency, largePayment, Alice, Bob)
    paymentBC = simulation_1.Payment(largeFrequency, largePayment, Charlie, Bob)
    paymentAD = simulation_1.Payment(freq, p, Alice, Bob)
    paymentAD1 = simulation_1.Payment(freq, p, Bob, Donna)
    paymentBD = simulation_1.Payment(largeFrequency, largePayment, Donna, Bob)
    paymentCD = simulation_1.Payment(largeFrequency, largePayment, Charlie, Bob)
    paymentCD1 = simulation_1.Payment(largeFrequency, largePayment, Bob, Donna)
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

    paymentAB = simulation_1.Payment(largeFrequency, largePayment, Alice, Bob)
    paymentBC = simulation_1.Payment(largeFrequency, largePayment, Charlie, Bob)
    paymentAD = simulation_1.Payment(freq, p, Alice, Donna)
    paymentBD = simulation_1.Payment(largeFrequency, largePayment, Donna, Bob)
    paymentCD = simulation_1.Payment(largeFrequency, largePayment, Charlie, Bob)
    paymentCD1 = simulation_1.Payment(largeFrequency, largePayment, Bob, Donna)
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

    network.addPaymentList([paymentAB, paymentBC, paymentAD, paymentBD, paymentCD])

    network.runNetwork()

    a = Alice.getChCostTotal()
    b = Bob.getChCostTotal()
    c = Charlie.getChCostTotal()
    d = Donna.getChCostTotal()

    return (a, b, c, d)


def networkLineOG(p, freq, onlineTX, onlineTXTime, r, timeRun):
    # the network as a line A-B-C-D
    network = simulation_1.Network(onlineTX, onlineTXTime, r, timeRun)
    
    Alice = simulation_1.Node("Alice", network)
    Bob = simulation_1.Node("Bob", network)
    Charlie = simulation_1.Node("Charlie", network)
    Donna = simulation_1.Node("Donna", network)

    paymentAB = simulation_1.Payment(largeFrequency, largePayment, Alice, Bob)
    paymentBC = simulation_1.Payment(largeFrequency, largePayment, Charlie, Bob)
    # paymentBD = simulation_1.Payment(0.5, 1, Bob, Charlie)
    # paymentBD1 = simulation_1.Payment(0.5, 1, Charlie, Donna)
    paymentCD = simulation_1.Payment(largeFrequency, largePayment, Charlie, Donna)
    # paymentBD.setTransfer(paymentBD1)

    channelAB = simulation_1.Channel(Alice, Bob, network)
    channelBC = simulation_1.Channel(Bob, Charlie, network)
    channelCD = simulation_1.Channel(Charlie, Donna, network)

    channelAB.addPaymentList([paymentAB])
    # channelBC.addPaymentList([paymentBC, paymentAD1, paymentBD])
    channelBC.addPaymentList([paymentBC])
    # channelCD.addPaymentList([paymentCD, paymentBD1, paymentAD2])
    channelCD.addPaymentList([paymentCD])

    # network.addPaymentList([paymentAB, paymentBC, paymentAD, paymentBD, paymentCD])
    network.addPaymentList([paymentAB, paymentBC, paymentCD])

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

    paymentAB = simulation_1.Payment(largeFrequency, largePayment, Alice, Bob)
    paymentBC = simulation_1.Payment(largeFrequency, largePayment, Charlie, Bob)
    paymentAD = simulation_1.Payment(freq, p, Alice, Donna)
    # paymentBD = simulation_1.Payment(0.5, 1, Bob, Charlie)
    # paymentBD1 = simulation_1.Payment(0.5, 1, Charlie, Donna)
    paymentCD = simulation_1.Payment(largeFrequency, largePayment, Charlie, Donna)
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

    paymentAB = simulation_1.Payment(largeFrequency, largePayment, Alice, Bob)
    paymentBC = simulation_1.Payment(largeFrequency, largePayment, Charlie, Bob)
    paymentAD = simulation_1.Payment(freq, p, Alice, Bob)
    paymentAD1 = simulation_1.Payment(freq, p, Bob, Charlie)
    paymentAD2 = simulation_1.Payment(freq, p, Charlie, Donna)
    # paymentBD = simulation_1.Payment(0.5, 1, Bob, Charlie)
    # paymentBD1 = simulation_1.Payment(0.5, 1, Charlie, Donna)
    paymentCD = simulation_1.Payment(largeFrequency, largePayment, Charlie, Donna)
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

    paymentAB = simulation_1.Payment(largeFrequency, largePayment, Alice, Bob)
    paymentBC = simulation_1.Payment(largeFrequency, largePayment, Charlie, Bob)
    paymentAD = simulation_1.Payment(freq, p, Alice, Bob)
    paymentAD1 = simulation_1.Payment(freq, p, Bob, Donna)
    paymentAC = simulation_1.Payment(freq, p, Alice, Bob)
    paymentAC1 = simulation_1.Payment(freq, p, Bob, Charlie)
    paymentBD = simulation_1.Payment(largeFrequency, largePayment, Bob, Donna)
    paymentAD.setTransfer(paymentAD1)
    paymentAC.setTransfer(paymentAC1)

    channelAB = simulation_1.Channel(Alice, Bob, network)
    channelBC = simulation_1.Channel(Bob, Charlie, network)
    channelBD = simulation_1.Channel(Bob, Donna, network)

    channelAB.addPaymentList([paymentAB, paymentAD, paymentAC])
    channelBC.addPaymentList([paymentBC, paymentAC1])
    channelBD.addPaymentList([paymentBD, paymentAD1])

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

    paymentAB = simulation_1.Payment(largeFrequency, largePayment, Alice, Bob)
    paymentBC = simulation_1.Payment(largeFrequency, largePayment, Charlie, Bob)
    paymentAD = simulation_1.Payment(freq, p, Alice, Donna)
    paymentAC = simulation_1.Payment(freq, p, Alice, Charlie)
    paymentBD = simulation_1.Payment(largeFrequency, largePayment, Donna, Bob)

    channelAB = simulation_1.Channel(Alice, Bob, network)
    channelBC = simulation_1.Channel(Bob, Charlie, network)
    channelBD = simulation_1.Channel(Bob, Donna, network)
    # added direct channel for payment AD
    channelAD = simulation_1.Channel(Alice, Donna, network)
    channelAC = simulation_1.Channel(Alice, Charlie, network)

    channelAB.addPaymentList([paymentAB])
    channelBC.addPaymentList([paymentBC])
    channelAD.addPaymentList([paymentAD])
    channelBD.addPaymentList([paymentBD])
    channelAC.addPaymentList([paymentAC])

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
    tempStarOG = networkStarOG(p, freq, onlineTX, onlineTXTime, r, timeRun)
    tempStar = networkStar(p, freq, onlineTX, onlineTXTime, r, timeRun)
    tempTri = networkTri(p, freq, onlineTX, onlineTXTime, r, timeRun)
    simulation_res.append((tempStarOG[0]+tempStarOG[3], tempStarOG[1]+tempStarOG[2]))
    simulation_res.append((tempStar[0]+tempStar[3], tempStar[1]+tempStar[2]))
    simulation_res.append((tempTri[0]+tempTri[3], tempTri[1]+tempStar[2]))

    return simulation_res

def setUpPF_cycle(p=0.1, freq=0.5, onlineTX = 5.0, onlineTXTime = 1.0, r = 0.01, timeRun = 10.0):
    simulation_res = []
    tempCycle = networkCycle(p, freq, onlineTX, onlineTXTime, r, timeRun)
    tempLine = networkLine(p, freq, onlineTX, onlineTXTime, r, timeRun)
    tempLineOG = networkLineOG(p, freq, onlineTX, onlineTXTime, r, timeRun)
    simulation_res.append((tempLineOG[0]+ tempLineOG[3], tempLineOG[2]+tempLineOG[1]))
    simulation_res.append((tempLine[0]+ tempLine[3], tempLine[2]+tempLine[1]))
    simulation_res.append((tempCycle[0]+tempCycle[3], tempCycle[2]+tempCycle[1]))

    return simulation_res

def setUpPF_star2(p=0.1, freq=0.5, onlineTX = 5.0, onlineTXTime = 1.0, r = 0.01, timeRun = 10.0):
    simulation_res = []
    tempStarOG = networkStarOG(p, freq, onlineTX, onlineTXTime, r, timeRun)
    tempStar = networkStar2(p, freq, onlineTX, onlineTXTime, r, timeRun)
    tempTri = networkTri2(p, freq, onlineTX, onlineTXTime, r, timeRun)
    simulation_res.append((tempStarOG[0]+tempStarOG[3], tempStarOG[1]+tempStarOG[2]))
    simulation_res.append((tempStar[0]+tempStar[2]+tempStar[3], tempStar[1]))
    simulation_res.append((tempTri[0]+tempTri[2]+tempTri[3], tempTri[1]))

    return simulation_res

############# Main functions #####################

def graphPF(type, p=0.1, freq=0.5, onlineTX = 5.0, onlineTXTime = 1.0, r = 0.01, timeRun = 10.0):
    # 0.2 to 1.2    
    ps = [x* psmultiply /psdivision for x in range(psInit, psInit+psLen)]
    # ps = [x* 1.0 / 1 for x in range(1, 30)]
    # 0.25 to 1.25
    fs = [x* fsmultiply /fsdivision for x in range(fsInit, fsInit+fsLen)]

    maxFee = [[0 for x in range(len(ps))] for x in range(len(fs))]
    minFee = [[0 for x in range(len(ps))] for x in range(len(fs))]
    diff = [[0 for x in range(len(ps))] for x in range(len(fs))]

    
    for i in range(len(fs)):
        # print(i)
        for j in range(len(ps)):
            print("%d-%d" %(i, j))
            # res = [0 for x in range(2)] 
            maxF, minF, diffTemp = 0, 0, 0

            for k in range(num_trial):
                print("%d-%d-%d" %(i, j, k))
                # setup returns ((a0,b0), (aTransfer, bTransfer), (aDirect, bDirect))
                if type == 0:
                    tmp = setUpPF_star(p=ps[j], freq=fs[i], timeRun = timeRun)
                elif type == 1:
                    tmp = setUpPF_star2(p=ps[j], freq=fs[i], timeRun = timeRun)    
                else: 
                    tmp = setUpPF_cycle(p=ps[j], freq=fs[i], timeRun = timeRun)               
                # temp = [tmp[1][0]-tmp[0][0], tmp[0][1]-tmp[1][1]]

                # for h in range(len(temp)):
                #     res[h] += temp[h]

                maxF += tmp[2][0] - tmp[0][0]
                bob22 = tmp[1][1] + (tmp[1][0] - tmp[0][0])
                minF += bob22 - tmp[0][1]
                

            # print((maxFee[i][j] - minFee[i][j]))
            # for h in range(len(temp)):
            #     res[h]= res[h]/float(num_trial)
            maxFee[i][j] = (maxF/num_trial)
            minFee[i][j] = (minF/num_trial)
            diff[i][j] = (maxFee[i][j] - minFee[i][j])

    X = np.array(ps)
    Y = np.array(fs)
    X, Y = np.meshgrid(X, Y)
    Z = np.array(maxFee)
    U = np.array(minFee)
    W = np.array(diff)


    fig2 = plt.figure(figsize=plt.figaspect(0.5))
   
    ax = fig2.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel('Payment size')
    ax.set_ylabel('Frequency')
    ax.set_zlabel("Fee Bound Difference")
    ax.set_title("4 node fee bound differences vs payment size vs frequency")
    ax.view_init(azim=0, elev=90)        
    fig2.text(0, 0, 'trials: %d; Time: %d' % (num_trial, time))
    surf = ax.plot_surface(X, Y, W, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    fig2.colorbar(surf, shrink=0.5, aspect=10)


    fig2.savefig('4node_pf%d.png'%(type))

    intercepts = helpersMatrix.getIntercepts(diff, ps, fs)
    intercepts = np.transpose(np.array(intercepts))

    fitm, fitb = np.polyfit(intercepts[0], intercepts[1], 1)
                        
    if (len(intercepts) != 0):
        print("intercepts")
        print(intercepts)

        fit1 = np.poly1d(np.polyfit(intercepts[0], intercepts[1], 1))
        fit2 = np.poly1d(np.polyfit(intercepts[0], intercepts[1], 2))
        fit3 = np.poly1d(np.polyfit(intercepts[0], intercepts[1], 3))



        fig3 = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig3.add_subplot(1, 1, 1)
        
        ax.set_xlabel('Payment size')
        ax.set_ylabel('Frequency')
        ax.set_title("Intersection points")

        ax.scatter(intercepts[0], intercepts[1], marker='o')
        ax.plot(intercepts[0], intercepts[1], '.', intercepts[0], fit1(intercepts[0]), '-', 
            intercepts[0], fit2(intercepts[0]), '--', intercepts[0], fit3(intercepts[0]), '-.')



        fig3.text(0, 0, 'trials: %d; Time: %d' % (num_trial, time))
        fig3.savefig('4node_pf_inter%d.png'%(type))


    

################ Call #####################


# type  0 - single payment star
#       1 - double payment star
#       2 - chain/line

graphPF(1, timeRun = time)

# network flow
# signle payment star: A->B, B->C, A->D, A->C
# double payment star: A->B, B->C, C->D, B->D, A->D
# chain: A->B, B->C, C->D, A->D