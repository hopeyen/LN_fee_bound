import numpy as np 
import pandas as pd 
import random
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

class Node(object):
	self.nodes = []
	def __init__(self, name, network):
		self.name = name
		self.channels = []
		self.payments = []
		self.transferred = []
		self.revenue = 0
		self.channelCost = 0

		self.network = network
		self.nodes.append(self)


	def __eq__(self, other):
	    return (isinstance(other, Node) and (self.name == other.name))

	def __repr__(self):
	    return "%s" % (self.name)


	def addChannel(self, channel):
		self.channels.append(channel)

	def addPayment(self, payment):
		self.payments.append(payment)

		for c in self.channels:
			# has direct
			if c.B == payment.reciever:
				return



		

	# def updateChannelCost(self):
	# 	ans = 0
	# 	for c in self.channels:
	# 		ans += c.getChannelCost()/2
	# 	self.channelCost = ans

	# def costDirectChannel(self, payment):
	# 	# if the node creates an unidir channel for the payment
	# 	if payment != None:
	# 		return math.sqrt(2 * self.network.onlineTX * payment.freq * payment.amt / self.network.r)
	# 	return 0

	def getChCostTotal(self):
		self.channelCost = 0
		for c in self.channels:
			self.channelCost += c.cost *1.0 /2
		return self.channelCost

	def chooseTransfer(self):
		minFeeChannel = self.channels[0]
		for c in self.channels:
			if (self == c.A) and (c.feeA < minFeeChannel.feeA):
				minFeeChannel = c
			elif (self == c.B) and (c.feeB < minFeeChannel.feeB):
				minFeeChannel = c
		return minFeeChannel



class Payment(object):
	"""docstring for Payment"""
	payments = []
	def __init__(self, freq, amt, sender, reciever):
		self.freq = freq
		self.amt = amt
		self.sender = sender
		self.reciever = reciever
		self.numPaid = 0
		self.txTimes = []
		self.nextTime = self.nextPaymentTime()
		self.channel = None
		self.transferTo = None


		Payment.payments.append(self)

	def __eq__(self, other):
	    return (isinstance(other, Payment) and (self.freq == other.freq)
	    	and (self.amt == other.amt))

	def __repr__(self):
	    return ("%s sends %f to %s" 
	    	    	% (self.sender, self.amt, self.reciever))

	def nextPaymentTime(self):
		self.nextTime = random.expovariate(self.freq)
		self.txTimes.append(self.nextTime)
		self.numPaid += 1
		return self.nextTime

	def setChannel(self, channel):
		self.channel = channel

	def setTransfer(self, payment):
		self.transferTo = payment

	def estimateUbd(self):
	    network = self.sender.network
	    return math.sqrt(2 * network.onlineTX * self.freq * self.amt / network.r)

	def estimtedLbd(self, paymentAB):
	    n = self.sender.network
	    # interests = (payment.amt * network.r)/ (lifetime + network.r)
	    cnb1 = math.sqrt(2 * n.onlineTX * (self.freq * self.amt + paymentAB.freq * paymentAB.amt) / n.r)
	    cnb2 = 3 * ((2 * n.onlineTX * (self.freq + paymentAB.freq) / n.r)**(1.0/3))
	    cob1 = math.sqrt(2 * n.onlineTX * paymentAB.freq * paymentAB.amt / n.r)
	    cob2 = 3 * ((2 * n.onlineTX * paymentAB.freq / n.r)**(1.0/3))
	    return cnb1 + cnb2 - cob1 - cob2



class Channel(object):
	channels = []
	def __init__(self, A, B, mA, mB, network):
		# super().__init__(A, B, network)
		self.A = A
		self.B = B
		self.mA = mA
		self.mB = mB
		self.balanceA = 0
		self.balanceB = 0
		self.paymentsA = []
		self.paymentsB = []
		self.txTimesA = []
		self.txTimesB = []
		self.feeA = 0
		self.feeB = 0

		self.network = network

		self.cost = 0
		# add for A and B
		self.transferPayment = None

		A.addChannel(self)
		B.addChannel(self)		

		Channel.channels.append(self)

	def __eq__(self, other):
	    return (isinstance(other, Channel) and (self.A == other.A)
	    	and (self.B == other.B) and (self.network == other.network))

	def __repr__(self):
	    return ("%s has balance %f, %s has balance %f" 
	    	    	% (self.A, self.balanceA, self.B, self.balanceB))

	def getChannelCost(self):
		# # simplied version of calculating the cost
		# freq = 0
		# for p in self.paymentsA:
		# 	freq += p.freq
		# for p in self.paymentsB:
		# 	freq -= p.freq
		# freq = abs(freq)
		# self.cost = 3* ((2* network.onlineTX * freq / network.r)**(1.0/3))
		return self.cost

	def updateCost(self):
		# add discounted onlineTx cost to the total cost
		self.cost += self.network.onlineTX * math.exp(-1 * self.network.r * (self.network.totalTime - self.network.timeLeft))

	def addPayment(self, payment):
		# print("add payment in  " + payment.sender.name + " to " + payment.reciever.name)
		if payment.sender == self.A:
			self.paymentsA.append(payment)

		elif payment.sender == self.B:
			self.paymentsB.append(payment)
		

		self.A.addPayment(payment)
		self.B.addPayment(payment)

		payment.setChannel(self)
		
	def setChannelSize(self, mA, mB):
		self.mA = mA
		self.mB = mB

	def avergeFreq(self, payments):
		pavg = 0
		psum = 0
		for p in payments:
			pavg += (p.amt * p.freq)
			psum += p.amt
		if psum == 0: return 0
		return (pavg/psum)

	def optimizeSize(self):
		fA = self.avergeFreq(self.paymentsA)
		fB = self.avergeFreq(self.paymentsB)
		oneWay = 0
		bidir = 0

		alpha = min(fA, fB)
		beta = max(fA, fB)

		oneWay = math.sqrt(self.network.onlineTX * (beta - alpha) / self.network.r)
		bidir = (2 * self.network.onlineTX * alpha / self.network.r)**(1.0/3)

		if alpha == fA:
			self.setChannelSize(bidir, bidir+oneWay)
		else:
			self.setChannelSize(bidir+oneWay, bidir)


	def reopen(self, side):
		self.updateCost()

		payments = []
		if self.A == side:
			self.balanceA = self.mA
			payments = self.paymentsA
		elif self.B == side:
			self.balanceB = self.mB
			payments = self.paymentsB

		# channel suspended for network online transaction time
		for p in payments:
			p.nextTime += self.network.onlineTXTime



	def processPayment(self, payment):
		time = payment.nextTime
		# print([self.A, self.B, payment.sender])

		if self.A == payment.sender:

			if self.balanceA < payment.amt:
				# A has to reopen the channel
				self.reopen(self.A)
				
			else:
				# able to make the payment, generate the next payment
				self.balanceA -= payment.amt
				self.balanceB += payment.amt
				payment.nextPaymentTime()
				return True
	
		elif self.B == payment.sender:

			if self.balanceB < payment.amt:
				# B has to reopen the channel
				self.reopen(self.B)

			else:
				# able to make the payment, generate the next payment
				self.balanceB -= payment.amt
				self.balanceA += payment.amt
				payment.nextPaymentTime()
				return True

		# payment is not processed because of reopening 
		return False		

	def processTransfer(self, payment):
		if self.A == payment.sender:
			if self.balanceA < payment.amt:
				# A has to reopen the channel
				self.reopen(self.A)
				
			else:
				self.balanceA -= payment.amt
				self.balanceB += payment.amt
				payment.numPaid += 1
				return True
	
		elif self.B == payment.sender:
			if self.balanceB < payment.amt:
				self.reopen(self.B)

			else:
				self.balanceB -= payment.amt
				self.balanceA += payment.amt
				payment.numPaid += 1
				return True
		return False	

class Network(object):
	# keep track of the state of the network, include structure and flow
	def __init__(self, onlineTX, onlineTXTime, r, timeLeft):
		self.onlineTX = onlineTX
		self.onlineTXTime = onlineTXTime
		self.r = r
		self.nodes = []
		self.channels = []
		self.totalTime = timeLeft

		self.timeLeft = timeLeft
		self.payments = []
		self.history = []

	def addNode(self, node):
		self.nodes.append(node)

	def addNodeList(self, ns):
		self.nodes.extend(ns)

	def addChannel(self, channel):
		self.channels.append(channel)

	def addChannelList(self, chs):
		self.channels.extend(chs)

	def addPayment(self, payment):
		self.payments.append(payment)

	def addPaymentList(self, ps):
		self.payments.extend(ps)

	def runNetwork(self):
		# payments can be concurrent on different channels
		# the payment that takes the smallest time should be processed first
		# and when it has been processed, all other payments' interval decrement by the interval of the processed payment
		# and the processed payment has a new interval that gets put into the timeline

		for c in self.channels:
			c.optimizeSize()

		while self.timeLeft >= 0:

			nextPayment = self.payments[0]
			nPTime = self.payments[0].nextTime
			for p in self.payments:
				if p.nextTime < nPTime:
					nextPayment = p
					nPTime = p.nextTime
			if nPTime > self.timeLeft:
				break

			# process the next payment in the channel
			# the channel checks for the channel balance, reopen if balance not enough
			if (nextPayment.channel.processPayment(nextPayment)):
				
				self.timeLeft -= nPTime
				for p in self.payments:
					if p != nextPayment:
						p.nextTime -= nPTime
				if nextPayment.transferTo != None:
					nextPayment.transferTo.channel.processTransfer(nextPayment.transferTo)

				self.history.append((self.timeLeft, nextPayment, nextPayment.channel.balanceA, nextPayment.channel.balanceB))

		# self.printSummary()
		# print(self.history)
		# for p in self.payments:
		# 	print([p, p.numPaid])
			
	def printSummary(self):
		for c in self.channels:
			print([c.A, c.B, c.mA, c.mB])

	def getTotalCost(self):
		s = 0
		for n in self.nodes:
			s += n.getChCostTotal()
		return s

def networkDirectAC(p, freq, onlineTX, onlineTXTime, r, timeRun):
	# set up the network
	network = Network(onlineTX, onlineTXTime, r, timeRun)

	Alice = Node("Alice", network)
	Bob = Node("Bob", network)
	Charlie = Node("Charlie", network)

	paymentAB = Payment(2, 1, Alice, Bob)
	paymentBC = Payment(2, 1, Bob, Charlie)
	paymentAC = Payment(freq, p, Alice, Charlie)

	channelAB = Channel(Alice, Bob, 5, 0, network)
	channelAB.addPayment(paymentAB)
	channelBC = Channel(Bob, Charlie, 20, 20, network)
	channelBC.addPayment(paymentBC)

	# Alice creates a direct channel for network 1
	channelAC = Channel(Alice, Charlie, 5, 0, network)
	channelAC.addPayment(paymentAC)
	
	# print("network")
	network.addNodeList([Alice, Bob, Charlie])
	network.addChannelList([channelAB, channelBC, channelAC])
	network.addPaymentList([paymentAB, paymentBC, paymentAC])
	network.runNetwork()

	a = Alice.getChCostTotal()
	b = Bob.getChCostTotal()
	c = Charlie.getChCostTotal()

	return (a, b, c)
	
def networkOG(p, freq, onlineTX, onlineTXTime, r, timeRun):
	# network 2 
	network = Network(onlineTX, onlineTXTime, r, timeRun)
	
	Alice = Node("Alice", network)
	Bob = Node("Bob", network)
	Charlie = Node("Charlie", network)

	paymentAB = Payment(2, 1, Alice, Bob)
	paymentBC = Payment(2, 1, Bob, Charlie)

	channelAB = Channel(Alice, Bob, 5, 0, network)
	channelAB.addPayment(paymentAB)
	channelBC = Channel(Bob, Charlie, 20, 20, network)
	channelBC.addPayment(paymentBC)

	network.addNodeList([Alice, Bob, Charlie])
	network.addChannelList([channelAB, channelBC])
	network.addPaymentList([paymentAB, paymentBC])
	# print("network2")
	network.runNetwork()
	# print([paymentAC2, paymentAC2.numPaid])


	a = Alice.getChCostTotal()
	b = Bob.getChCostTotal()
	c = Charlie.getChCostTotal()

	return (a, b, c)


def networktransferB(p, freq, onlineTX, onlineTXTime, r, timeRun):
	# network 2 
	network = Network(onlineTX, onlineTXTime, r, timeRun)
	
	Alice = Node("Alice", network)
	Bob = Node("Bob", network)
	Charlie = Node("Charlie", network)

	paymentAB = Payment(2, 1, Alice, Bob)
	paymentBC = Payment(2, 1, Bob, Charlie)
	paymentAC = Payment(0.5, p, Alice, Bob)
	paymentAC1 = Payment(0.5, p, Bob, Charlie)
	paymentAC.setTransfer(paymentAC1)

	channelAB = Channel(Alice, Bob, 5, 0, network)
	channelAB.addPayment(paymentAB)
	channelBC = Channel(Bob, Charlie, 20, 20, network)
	channelBC.addPayment(paymentBC)

	# payment goes through Channel AB and BC
	channelAB.addPayment(paymentAC)
	channelBC.addPayment(paymentAC1)


	network.addNodeList([Alice, Bob, Charlie])
	network.addChannelList([channelAB, channelBC])
	network.addPaymentList([paymentAB, paymentBC, paymentAC])
	# print("network2")
	network.runNetwork()
	# print([paymentAC2, paymentAC2.numPaid])


	a = Alice.getChCostTotal()
	b = Bob.getChCostTotal()
	c = Charlie.getChCostTotal()

	return (a, b, c)

def networktransferC(p, freq, onlineTX, onlineTXTime, r, timeRun):
	# network 2 
	network = Network(onlineTX, onlineTXTime, r, timeRun)
	
	Alice = Node("Alice", network)
	Bob = Node("Bob", network)
	Charlie = Node("Charlie", network)

	# AB becomes the transferred payment that go through C
	paymentBC = Payment(2, 1, Bob, Charlie)
	paymentAC = Payment(0.5, p, Alice, Charlie)
	paymentAB = Payment(2, 1, Alice, Charlie)
	paymentAB1 = Payment(2, 1, Charlie, Bob)
	paymentAB.setTransfer(paymentAB1)

	# existing channels become AC and BC
	channelBC = Channel(Bob, Charlie, 20, 20, network)
	channelBC.addPayment(paymentBC)
	channelAC = Channel(Alice, Charlie, 5, 0, network)
	channelAC.addPayment(paymentAC)

	# payment goes through Channel AC and BC
	channelAC.addPayment(paymentAB)
	channelBC.addPayment(paymentAB1)


	network.addNodeList([Alice, Bob, Charlie])
	network.addChannelList([channelBC, channelAC])
	network.addPaymentList([paymentAB, paymentBC, paymentAC])
	# print("network2")
	network.runNetwork()
	# print([paymentAC2, paymentAC2.numPaid])


	a = Alice.getChCostTotal()
	b = Bob.getChCostTotal()
	c = Charlie.getChCostTotal()

	return (a, b, c)


def networktransferA(p, freq, onlineTX, onlineTXTime, r, timeRun):
	# network 2 
	network = Network(onlineTX, onlineTXTime, r, timeRun)
	
	Alice = Node("Alice", network)
	Bob = Node("Bob", network)
	Charlie = Node("Charlie", network)

	# BC becomes the transferred payment that go through A
	paymentAB = Payment(2, 1, Alice, Bob)
	paymentAC = Payment(0.5, p, Alice, Charlie)
	paymentBC = Payment(2, 1, Bob, Alice)	
	paymentBC1 = Payment(2, 1, Alice, Charlie)
	paymentBC.setTransfer(paymentBC1)

	# existing channels become AC and AB
	channelAB = Channel(Alice, Bob, 5, 0, network)
	channelAB.addPayment(paymentAB)
	channelAC = Channel(Alice, Charlie, 5, 0, network)
	channelAC.addPayment(paymentAC)

	# payment goes through Channel AC and AC
	channelAB.addPayment(paymentBC)
	channelAC.addPayment(paymentBC1)


	network.addNodeList([Alice, Bob, Charlie])
	network.addChannelList([channelAB, channelAC])
	network.addPaymentList([paymentAB, paymentBC, paymentAC])
	# print("network2")
	network.runNetwork()
	# print([paymentAC2, paymentAC2.numPaid])


	a = Alice.getChCostTotal()
	b = Bob.getChCostTotal()
	c = Charlie.getChCostTotal()

	return (a, b, c)

	

def main(p=0.1, freq=0.5, onlineTX = 5.0, onlineTXTime = 1.0, r = 0.01, timeRun = 10.0):
	(a0, b0, c0) = networkOG(p, freq, onlineTX, onlineTXTime, r, timeRun)
	(a1, b1, c1) = networkDirectAC(p, freq, onlineTX, onlineTXTime, r, timeRun)
	(a2, b2, c2) = networktransferB(p, freq, onlineTX, onlineTXTime, r, timeRun)
	(a3, b3, c3) = networktransferC(p, freq, onlineTX, onlineTXTime, r, timeRun)
	(a4, b4, c4) = networktransferA(p, freq, onlineTX, onlineTXTime, r, timeRun)
	
	# lbd = paymentAC1.estimtedLbd(paymentAB1)
	# ubd = paymentAC1.estimateUbd()

	# graphing 
	# return (a1-a2+c1-c2, b2-b1)

	# channels
	# return (a1+c1, b1, a2+c2, b2, a3+c3, b3, a4+ c4, b4)

	# max_fee based on network 1 and 2
	# return (a0+c0, b0, a1+c1, b1, a2+c2, b2)

	# just bob
	return (a0+c0, b0, a1+c1, b1, a2+c2, b2)





if __name__ == '__main__':
    main()