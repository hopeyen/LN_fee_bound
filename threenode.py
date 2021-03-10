from simulation_1 import main
from simulation_1 import mainOppo
from simulation_1 import mainOppo2
import random
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

############ Helper functions ################

def getHigherMean(list1, list2):
    avg1 = float(sum(map(sum,list1)))/(len(list1)*len(list1[0]))
    avg2 = float(sum(map(sum,list2)))/(len(list2)*len(list2[0]))

    if avg1 > avg2:
        return list1
    else:
        return list2

def getLowerMean(list1, list2):
    avg1 = float(sum(map(sum,list1)))/(len(list1)*len(list1[0]))
    avg2 = float(sum(map(sum,list2)))/(len(list2)*len(list2[0]))

    if avg1 <= avg2:
        return list1
    else:
        return list2

def calculateFee(list1, list2):
    lower = getLowerMean(list1, list2)
    higher = getHigherMean(list1, list2)

    ans = []
    for i in range(len(lower)):
        temp = []
        for j in range(len(lower[i])):
            temp.append(higher[i][j] -lower[i][j])
        ans.append(temp)
    return ans


def chargeFee(list, fee):
    ans = []
    for i in range(len(fee)):
        temp = []
        for j in range(len(fee[i])):
            temp.append(list[i][j] - fee[i][j])
        ans.append(temp)
    return ans

def payFee(list, fee):
    ans = []
    for i in range(len(fee)):
        temp = []
        for j in range(len(fee[i])):
            temp.append(list[i][j] + fee[i][j])
        ans.append(temp)
    return ans

def getIntersections(list1, list2, ps, fs):
    points = []

    for i in range(1, len(list1)):
        for j in range(1, len(list1[i])):
            if list1[i][j] == list2[i][j]:
                points.append([ps[i], fs[j], list2[i][j]])
            elif (((list1[i][j-1] > list2[i][j-1]) and (list1[i][j] < list2[i][j]))
                or ((list1[i][j-1] < list2[i][j-1]) and (list1[i][j] > list2[i][j]))):
                points.append([ps[i], fs[j], (list1[i][j-1]+list1[i][j])/2])
            elif (((list1[i-1][j] > list2[i-1][j]) and (list1[i][j] < list2[i][j]))
                or ((list1[i-1][j] < list2[i-1][j]) and (list1[i][j] > list2[i][j]))):
                points.append([ps[i], fs[j], (list1[i][j-1]+list1[i][j])/2])


    return points

def getIntercepts(list, ps, fs):
    points = []

    for i in range(1, len(ps)):
        for j in range(1, len(fs)):
            if list[i][j] == 0:
                points.append((ps[i], fs[j], 0))
            elif (((list[i-1][j] < 0) or (list[i][j-1] < 0)) and (list[i][j] > 0)):
                points.append((ps[i], fs[j], 0))
            elif (((list[i-1][j] > 0) or (list[i][j-1] > 0)) and (list[i][j] < 0)):
                points.append((ps[i], fs[j], 0))


    return points

def transform(points):
    xs = []
    ys = []

    for i in points:
        xs.append(i[0])
        ys.append(i[1])

    return (xs, ys)


############## Constants and global variables ################

num_trial = 3

time = 50

givenP = 0.5

paymentMean = 0.5

paymentSigma = 0.1

psInit = 1

fsInit = 1

psLen = 100

fsLen = 100

psIncre = 10

fsIncre = 10
############# Main functions #####################

def runWithPayment(time, r, B, type):
    ps = [x * 2.0 /10 for x in range(psInit, psInit+psLen)]
    # # 0.25 to 1.25
    fs = [x * 2.0 /10 for x in range(fsInit, fsInit+fsLen)]

    X = np.array(ps)
    Y = np.array(fs)
    X, Y = np.meshgrid(X, Y)

    costs = [[[0 for x in range(psLen)] for x in range(fsLen)] for x in range(0, 6)]

    for i in range(len(fs)):

        for j in range(len(ps)):
            print("%d-%d" %(i, j))

            for k in range(num_trial):
                if type == 0:
                    res = main(p=ps[j], freq=fs[i], timeRun = time, r = r, onlineTX = B)
                elif type == 1:
                    res = mainOppo(p=ps[j], freq=fs[i], timeRun = time, r = r, onlineTX = B)
                else:
                    res = mainOppo2(p=ps[j], freq=fs[i], timeRun = time, r = r, onlineTX = B)
                for h in range(len(res)):
                    costs[h][i][j] += res[h]

            for h in range(len(costs)):
                costs[h][i][j] /= float(num_trial)

    #         # alice0         costs[0] 
    #         # bob0           costs[1] 
    #         # alice1         costs[2] 
    #         # bob1           costs[3] 
    #         # alice2         costs[4] 
    #         # bob2           costs[5] 

    maxFee = chargeFee(costs[2], costs[0])
    
    # aliceAfter_max = payFee(alice0, maxFee)

    # bob2'=bob2+(alice2-alice0)
    # minFee = bob2'-bob0 = bob2+(alice2-alice0)-bob0 
    bob22 = payFee(costs[5], chargeFee(costs[4], costs[0]))
    minFee = chargeFee(bob22, costs[1])

    diff = chargeFee(maxFee, minFee)


    Z = np.array(maxFee)
    U = np.array(minFee)
    W = np.array(diff)


    # returns a list of (ps, fs, pt)
    inter = getIntersections(maxFee, minFee, ps, fs)
    inter = np.transpose(np.array(inter))



    plotsize = 1
    if len(inter) != 0:
        plotsize = 2
    # set up the plot
    

    fig = plt.figure(figsize=[12.8, 14.4])

    titles = ['Payment vs Frequency\n vs Maximum fee bound', 
            'Payment vs Frequency\n vs Minimum fee bound',
            'Payment vs Frequency\n vs fee bound difference']
    Zs = [Z, U, W]

    titles_users = ['alice0', 'bob0', 'alice1', 'bob1', 'alice2', 'bob2']
    index = 1
    for i in range(6):
        ax = fig.add_subplot(3, 2, i+1, projection='3d')
        ax.set_xlabel('Payment size')
        ax.set_ylabel('Frequency')
        ax.set_zlabel("Channel Costs")
        ax.set_title(titles_users[i])
        ax.view_init(azim=0, elev=90)

        Z = np.array(costs[i])
        
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=10)
        index += 1
    fig.savefig('3node_pf_usersOppo%d.png'%type)


    fig2 = plt.figure(figsize=plt.figaspect(0.5))
   
    ax = fig2.add_subplot(1, 1, 1, projection='3d')

    zerosPlane = [[0 for i in range(psLen)] for i in range(fsLen)]
    Z = np.array(zerosPlane)


    ax.set_xlabel('Payment size')
    ax.set_ylabel('Frequency')
    ax.set_zlabel("Fee Bound Difference")
    ax.set_title(titles[2])

    surf = ax.plot_surface(X, Y, Zs[2], zorder=1, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    fig2.colorbar(surf, shrink=0.5, aspect=10)

    


    fig2.savefig('3node_pfOppo%d.png'%type)

    intercepts = getIntercepts(diff, ps, fs)
    intercepts = np.transpose(np.array(intercepts))

    fit1 = np.poly1d(np.polyfit(intercepts[0], intercepts[1], 5))
    fit2 = np.poly1d(np.polyfit(intercepts[0], intercepts[1], 2))
    fit3 = np.poly1d(np.polyfit(intercepts[0], intercepts[1], 3))

    
                        
    if (len(intercepts) != 0):
        print("intercepts")
        print(intercepts)


        fig3 = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig3.add_subplot(1, 1, 1)
        
        ax.set_xlabel('Payment size')
        ax.set_ylabel('Frequency')
        ax.set_title("Intersection points")

        ax.scatter(intercepts[0], intercepts[1], marker='o')
        ax.plot(intercepts[0], intercepts[1], '.', intercepts[0], fit1(intercepts[0]), '-', intercepts[0], fit2(intercepts[0]), '--', intercepts[0], fit3(intercepts[0]), '-.')


        fig3.savefig('3node_pf_interOppo%d.png'%type)


    return intercepts



################ Call #####################

# runWithPayment(time, 0.01, 5.0, 0)
# runWithFreq()




