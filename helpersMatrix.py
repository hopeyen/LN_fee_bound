from simulation_1 import main
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


def chargeFee(bob, fee):
    ans = []
    for i in range(len(fee)):
        temp = []
        for j in range(len(fee[i])):
            temp.append(bob[i][j] - fee[i][j])
        ans.append(temp)
    return ans

def payFee(alice, fee):
    ans = []
    for i in range(len(fee)):
        temp = []
        for j in range(len(fee[i])):
            temp.append(alice[i][j] + fee[i][j])
        ans.append(temp)
    return ans

# def getIntersections(list1, list2, ps):
#     points = []

#     for i in range(1, len(list1)):
#         if list1[i] == list2[i]:
#             points.append((i, list2[i]))
#         elif ((list1[i-1] > list2[i-1]) and (list1[i] < list2[i])): 
#             # or ((list1[i-1] < list2[i-1]) and (list1[i] > list2[i]))):
#             points.append((ps[i], (list1[i-1]+list1[i])/2))

#     return points

def transform(points):
    xs = []
    ys = []

    for i in points:
        xs.append(i[0])
        ys.append(i[1])

    return (xs, ys)


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
