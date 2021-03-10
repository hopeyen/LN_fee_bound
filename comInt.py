from threenode import runWithPayment
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


time = 50
def runCompare():
	inter0 = runWithPayment(time, 0.01, 5.0, 0)
	inter1 = runWithPayment(time, 0.01, 5.0, 1)
	inter2 = runWithPayment(time, 0.01, 5.0, 2)

	fig = plt.figure(figsize=plt.figaspect(0.5))

	ax = fig.add_subplot(1, 1, 1)
    
	ax.set_xlabel('Payment size')
	ax.set_ylabel('Frequency')
	ax.set_title("Intersection points")

                        
	if (len(inter0) != 0):
		fit = np.poly1d(np.polyfit(inter0[0], inter0[1], 5))
		# ax.plot(inter0[0], inter0[1], '.', inter0[0], fit(inter0[0]), '-.')
		ax.plot(inter0[0], fit(inter0[0]), '-.')

		# ax.scatter(inter0[0], inter0[1], marker='o')
	    
	if (len(inter1) != 0):
		fit = np.poly1d(np.polyfit(inter1[0], inter1[1], 5))
		# ax.plot(inter1[0], inter1[1], '.', inter1[0], fit(inter1[0]), '-.')
		ax.plot(inter1[0], fit(inter1[0]), '-.')
		# ax.scatter(inter1[0], inter1[1], marker='v')
	    
	if (len(inter2) != 0):
		fit = np.poly1d(np.polyfit(inter2[0], inter2[1], 5))
		# ax.plot(inter2[0], inter2[1], '.', inter2[0], fit(inter2[0]), '-.')
		ax.plot(inter2[0], fit(inter2[0]), '-.')

		# ax.scatter(inter2[0], inter2[1], marker='.')

	# fig.legend(["setup 0", "setup 0 best fit line", "setup 1", "setup 1 best fit line", "setup 2", "setup 2 best fit line"])

	fig.legend(["setup 0 best fit line", "setup 1 best fit line", "setup 2 best fit line"])

	    

	fig.savefig('compareInt_large.png')

runCompare()
