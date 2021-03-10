import numpy as np 
import pandas as pd 
import random
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

class Node(object):
	def __init__(self, name, network):
		self.name = name
		self.channels = []
		self.payments = []
		self.revenue = 0
		self.channelCost = 0

		self.network = network
		network.nodes.append(self)


	def __eq__(self, other):
	    return (isinstance(other, Node) and (self.name == other.name))

	def __repr__(self):
	    return "%s" % (self.name)

	def addChannel(self, channel):
		self.channels.append(channel)

	def addPayment(self, payment):
		self.payments.append(payment)

	def getChCostTotal(self):
		self.channelCost = 0
		for c in self.channels:
			self.channelCost += c.cost *1.0 /2
		return self.channelCost


class Payment(object):
	"""docstring for Payment"""
	payments = []
	def __init__(self, freq, amt, sender, reciever):
		self.freq = freq
		self.amt = amt
		self.sender = sender
		self.reciever = reciever
		self.numPaid = 0
		self.nextTime = self.nextPaymentTime()
		self.channel = None
		self.transferTo = None
		self.fee = 0
		self.numProcessed = 0
		
		Payment.payments.append(self)

	def __eq__(self, other):
	    return (isinstance(other, Payment) and (self.freq == other.freq)
	    	and (self.amt == other.amt))

	def __repr__(self):
	    return ("%s sends %f to %s at rate %f, num processed: %d" 
	    	    	% (self.sender, self.amt, self.reciever, self.freq, self.numProcessed))

	def nextPaymentTime(self):
		self.nextTime = random.expovariate(1/self.freq)
		# self.txTimes.append(self.nextTime)
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

	def setFee(self, fee):
		self.fee = fee

class Channel(object):
	channels = []
	def __init__(self, A, B, network):
		# super().__init__(A, B, network)
		self.A = A
		self.B = B
		self.network = network

		self.mA = 0
		self.mB = 0
		self.balanceA = 0
		self.balanceB = 0
		self.paymentsA = []
		self.paymentsB = []
		self.numReopen = 0
		
		self.cost = 0
		self.transferPayment = []

		A.addChannel(self)
		B.addChannel(self)

		Channel.channels.append(self)
		network.channels.append(self)

	def __eq__(self, other):
	    return (isinstance(other, Channel) and (self.A == other.A)
	    	and (self.B == other.B) and (self.network == other.network))

	def __repr__(self):
	    # return ("%s has balance %f, %s has balance %f" 
	    # 	    	% (self.A, self.balanceA, self.B, self.balanceB))
	    return ("%s and %s with cost %f, reopens %d times\n -- average frequencies (%f, %f) \n" 
	    	% (self.A, self.B, self.cost, self.numReopen, self.avergeFreq(self.paymentsA), self.avergeFreq(self.paymentsB))
	    	+ str(self.paymentsA))

	def getChannelCost(self):
		return self.cost

	def updateCost(self):
		# add discounted onlineTx cost to the total cost
		# Be^(-rt) with continuous compounding
		self.cost += self.network.onlineTX * math.exp(-1 * self.network.r * (self.network.totalTime - self.network.timeLeft))

	def addPayment(self, payment):
		if payment.sender == self.A:
			self.paymentsA.append(payment)

		elif payment.sender == self.B:
			self.paymentsB.append(payment)

		self.A.addPayment(payment)
		self.B.addPayment(payment)

		payment.setChannel(self)

	def addPaymentList(self, payments):
		for p in payments:
			self.addPayment(p)

	def addTransferPayment(self, payment):
		self.addPayment(payment)
		self.transferPayment.append(payment)

	def setChannelSize(self, mA, mB):
		self.mA = mA
		self.mB = mB

	def getSlowestFreq(self, payments):
		# at least one payment
		slowest = payments[0]

		for p in payments:
			if p.freq > slowest.freq:
				slowest = p

		return slowest

	def getSlowestFreq(self, payments):
		# at least one payment
		slowest = payments[0]
		sumFreq = 0

		for p in payments:
			if p.freq > slowest.freq:
				slowest = p
				sumFreq += 1/p.freq

		return (slowest, sumFreq)

	def getPortionFreq(self, payments):
		(slowest, sumFreq) = self.getSlowestFreq(payments)
		portion = (1/ slowest.freq) / sumFreq



	def avergeFreq(self, payments):
		sumFreq = 0

		for p in payments:
			sumFreq += p.amt / p.freq
			# print("paymnt f: %f; p: %f" %(p.freq, p.amt))

		if len(payments) == 0: return 0
		return sumFreq
		

	def optimizeSize(self):
		fA = self.avergeFreq(self.paymentsA)
		fB = self.avergeFreq(self.paymentsB)
		oneWay = 0
		bidir = 0

		oneWay = (self.network.onlineTX * abs(fA - fB) / self.network.r) **(1.0/2)
		bidir = (2 * self.network.onlineTX * min(fA, fB) / self.network.r)**(1.0/3)

		if min(fA, fB) == fA:
			self.setChannelSize(bidir, bidir+oneWay)
		else:
			self.setChannelSize(bidir+oneWay, bidir)


		self.balanceA = self.mA
		self.balanceB = self.mB
		self.updateCost()


	def reopen(self, side):
		self.updateCost()
		self.numReopen += 1

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

		if self.A == payment.sender:

			if self.balanceA < payment.amt:
				# A has to reopen the channel
				self.reopen(self.A)
				
			else:
				# able to make the payment, generate the next payment
				self.balanceA -= payment.amt
				self.balanceB += payment.amt
				payment.nextPaymentTime()
				payment.numProcessed += 1
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
				payment.numProcessed += 1
				return True

		# payment is not processed because of reopening 
		return False		

	def processTransfer(self, payment):
		# instant transfer
		if self.A == payment.sender:
			if self.balanceA < payment.amt:
				# A has to reopen the channel
				self.reopen(self.A)
				
			else:
				self.balanceA -= payment.amt
				self.balanceB += payment.amt
				payment.numProcessed += 1
				return True
	
		elif self.B == payment.sender:
			if self.balanceB < payment.amt:
				self.reopen(self.B)

			else:
				self.balanceB -= payment.amt
				self.balanceA += payment.amt
				payment.numProcessed += 1
				return True
		return False	

	def expectedTX(self):
		expectedA, expectedB = [], []
		totalTk = 0

		for p in self.paymentsA:
			expectedA.append(self.network.totalTime / p.freq)
			totalTk += (self.network.totalTime / p.freq) * p.amt

		for p in self.paymentsB:
			expectedB.append(self.network.totalTime / p.freq)
			totalTk += (self.network.totalTime / p.freq) * p.amt
		print("expected txs: A:" + str(expectedA) + "; B: " + str(expectedB) + 
			"; total number of transactions: " + str((sum(expectedA)+sum(expectedB))) + 
			"; total token: " + str(totalTk) + "; expected reopens: " + str(totalTk/self.mA))

		if (len(expectedA)> 1):
			print("A with multiple payments: " + str(self.network.totalTime * self.avergeFreq(self.paymentsA)))

		print("while channel size is (%f,%f)" %(self.mA, self.mB))


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
		self.transferredPayments = []
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

	def addTransferred(self, payment):
		self.transferredPayments.append(payment)

	def addTransferredList(self, ps):
		self.transferredPayments.extend(ps)

	def getNextPayment(self):
		nextPayment = self.payments[0]
		nPTime = self.payments[0].nextTime

		for p in self.payments:
			if p.nextTime < nPTime:
				nextPayment = p
				nPTime = p.nextTime

		return nextPayment

	def timeDecrease(self, pm, time):
		self.timeLeft -= time
		for p in self.payments:
			if p != pm:
				p.nextTime -= time


	def runNetwork(self):
		# payments can be concurrent on different channels
		# the payment that takes the smallest time should be processed first
		# and when it has been processed, all other payments' interval decrement by the interval of the processed payment
		# and the processed payment has a new interval that gets put into the timeline
		
		# initialize the channels
		for c in self.channels:
			c.optimizeSize()
			
			self.history.append(( 
				"Time %03f, initialize channel %s to %s, %f" 
				%(self.timeLeft, c.A.name, c.B.name, c.cost)))


		while self.timeLeft >= 0:
			nextPayment = self.getNextPayment()
			nPTime = nextPayment.nextTime
				
			# even the soonest payment is out of time
			if nPTime > self.timeLeft:
				break

			# process the next payment in the channel
			# true if processed, false if reopen due to balance 
			if (nextPayment.channel.processPayment(nextPayment)):
				# decrease the time of all other payments 
				self.timeDecrease(nextPayment, nPTime)

				self.history.append(( 
					"Time %03f, processed %dth %s to %s (f: %03f, p: %03f)" 
					%(self.timeLeft, nextPayment.numProcessed, nextPayment.sender.name, 
						nextPayment.reciever.name, nextPayment.freq, nextPayment.amt)))
				# without waiting, do the transfer
				if nextPayment.transferTo != None:
					nextPayment.transferTo.channel.processTransfer(nextPayment.transferTo)
					self.history.append(( 
						"Time %03f, transfer %dth %s to %s (f: %03f, p: %03f)" 
						%(self.timeLeft, nextPayment.transferTo.numProcessed, nextPayment.transferTo.sender.name, 
							nextPayment.transferTo.reciever.name, nextPayment.transferTo.freq, nextPayment.transferTo.amt)))
			else:
				# attempt to send failed, payment not processed
				errorTime = 0.001
				self.timeDecrease(nextPayment, errorTime)
				self.history.append(("Time %03f, reopen %s with %s, size (%03f, %03f)" 
					%(self.timeLeft, nextPayment.sender.name, nextPayment.reciever.name, nextPayment.channel.mA, nextPayment.channel.mB)))


		# print the history of the network
		# self.printSummary()
		

			
	def printSummary(self):
		print("history")
		print(np.array(self.history))
		print("Summary; Timeleft: %f" %self.timeLeft)
		print(" - Nodes: ")
		for n in self.nodes:
			print(n)
		print(" - Payments: ")
		for p in self.payments:
			print(p)
		print(" - Transferred Payments: ")
		for p in self.transferredPayments:
			print(p)
		print(" - Channels: ")
		for c in self.channels:
			print(c)
		print(" - testing")
		self.testing()
		print("\n")


	def getTotalCost(self):
		s = 0
		for n in self.nodes:
			s += n.getChCostTotal()
		return s
	
	def testing(self):
		print("--network expected num tx")
		for c in self.channels:
			c.expectedTX()


############################ example calls ###################

##### constants ######
largePayments = 1
largeFrequency = 1

##### Setup #####
def networkOG(p, freq, onlineTX, onlineTXTime, r, timeRun):
	# network 2 
	network = Network(onlineTX, onlineTXTime, r, timeRun)
	
	Alice = Node("Alice", network)
	Bob = Node("Bob", network)
	Charlie = Node("Charlie", network)

	paymentAB = Payment(largeFrequency, largePayments, Alice, Bob)
	paymentBC = Payment(largeFrequency, largePayments, Bob, Charlie)

	channelAB = Channel(Alice, Bob, network)
	channelAB.addPayment(paymentAB)
	channelBC = Channel(Bob, Charlie, network)
	channelBC.addPayment(paymentBC)

	network.addNodeList([Alice, Bob, Charlie])
	network.addChannelList([channelAB, channelBC])
	network.addPaymentList([paymentAB, paymentBC])
	network.runNetwork()

	a = Alice.getChCostTotal()
	b = Bob.getChCostTotal()
	c = Charlie.getChCostTotal()

	return (a+c, b)


def networkDirectAC(p, freq, onlineTX, onlineTXTime, r, timeRun):
	# set up the network
	network = Network(onlineTX, onlineTXTime, r, timeRun)

	Alice = Node("Alice", network)
	Bob = Node("Bob", network)
	Charlie = Node("Charlie", network)

	paymentAB = Payment(largeFrequency, largePayments, Alice, Bob)
	paymentBC = Payment(largeFrequency, largePayments, Bob, Charlie)
	paymentAC = Payment(freq, p, Alice, Charlie)

	channelAB = Channel(Alice, Bob, network)
	channelAB.addPayment(paymentAB)
	channelBC = Channel(Bob, Charlie, network)
	channelBC.addPayment(paymentBC)

	# Alice creates a direct channel for network 1
	channelAC = Channel(Alice, Charlie, network)
	channelAC.addPayment(paymentAC)
	
	network.addNodeList([Alice, Bob, Charlie])
	network.addChannelList([channelAB, channelBC, channelAC])
	network.addPaymentList([paymentAB, paymentBC, paymentAC])
	network.runNetwork()

	a = Alice.getChCostTotal()
	b = Bob.getChCostTotal()
	c = Charlie.getChCostTotal()

	return (a+c, b)
	


def networktransferB(p, freq, onlineTX, onlineTXTime, r, timeRun):
	# network 2 
	network = Network(onlineTX, onlineTXTime, r, timeRun)
	
	Alice = Node("Alice", network)
	Bob = Node("Bob", network)
	Charlie = Node("Charlie", network)

	paymentAB = Payment(largeFrequency, largePayments, Alice, Bob)
	paymentBC = Payment(largeFrequency, largePayments, Bob, Charlie)
	paymentAC = Payment(freq, p, Alice, Bob)
	paymentAC1 = Payment(freq, p, Bob, Charlie)
	paymentAC.setTransfer(paymentAC1)

	channelAB = Channel(Alice, Bob, network)
	channelAB.addPayment(paymentAB)
	channelBC = Channel(Bob, Charlie, network)
	channelBC.addPayment(paymentBC)

	# payment goes through Channel AB and BC
	channelAB.addPayment(paymentAC)
	channelBC.addPayment(paymentAC1)

	network.addNodeList([Alice, Bob, Charlie])
	network.addChannelList([channelAB, channelBC])
	network.addPaymentList([paymentAB, paymentBC, paymentAC])
	network.runNetwork()

	a = Alice.getChCostTotal()
	b = Bob.getChCostTotal()
	c = Charlie.getChCostTotal()

	return (a+c, b)

def networkOGOppoDir(p, freq, onlineTX, onlineTXTime, r, timeRun):
	# network 2 
	network = Network(onlineTX, onlineTXTime, r, timeRun)
	
	Alice = Node("Alice", network)
	Bob = Node("Bob", network)
	Charlie = Node("Charlie", network)

	paymentAB = Payment(largeFrequency, largePayments, Alice, Bob)
	paymentBC = Payment(largeFrequency, largePayments, Charlie, Bob)

	channelAB = Channel(Alice, Bob, network)
	channelAB.addPayment(paymentAB)
	channelBC = Channel(Bob, Charlie, network)
	channelBC.addPayment(paymentBC)

	network.addNodeList([Alice, Bob, Charlie])
	network.addChannelList([channelAB, channelBC])
	network.addPaymentList([paymentAB, paymentBC])
	network.runNetwork()

	a = Alice.getChCostTotal()
	b = Bob.getChCostTotal()
	c = Charlie.getChCostTotal()

	return (a+c, b)


def networkDirectACOppoDir(p, freq, onlineTX, onlineTXTime, r, timeRun):
	# set up the network
	network = Network(onlineTX, onlineTXTime, r, timeRun)

	Alice = Node("Alice", network)
	Bob = Node("Bob", network)
	Charlie = Node("Charlie", network)

	paymentAB = Payment(largeFrequency, largePayments, Alice, Bob)
	paymentBC = Payment(largeFrequency, largePayments, Charlie, Bob)
	paymentAC = Payment(freq, p, Alice, Charlie)

	channelAB = Channel(Alice, Bob, network)
	channelAB.addPayment(paymentAB)
	channelBC = Channel(Bob, Charlie, network)
	channelBC.addPayment(paymentBC)

	# Alice creates a direct channel for network 1
	channelAC = Channel(Alice, Charlie, network)
	channelAC.addPayment(paymentAC)
	
	network.addNodeList([Alice, Bob, Charlie])
	network.addChannelList([channelAB, channelBC, channelAC])
	network.addPaymentList([paymentAB, paymentBC, paymentAC])
	network.runNetwork()
	


	a = Alice.getChCostTotal()
	b = Bob.getChCostTotal()
	c = Charlie.getChCostTotal()

	return (a+c, b)
	


def networktransferBOppoDir(p, freq, onlineTX, onlineTXTime, r, timeRun):
	network = Network(onlineTX, onlineTXTime, r, timeRun)
	
	Alice = Node("Alice", network)
	Bob = Node("Bob", network)
	Charlie = Node("Charlie", network)

	paymentAB = Payment(largeFrequency, largePayments, Alice, Bob)
	paymentBC = Payment(largeFrequency, largePayments, Charlie, Bob)
	paymentAC = Payment(freq, p, Alice, Bob)
	paymentAC1 = Payment(freq, p, Bob, Charlie)
	paymentAC.setTransfer(paymentAC1)

	channelAB = Channel(Alice, Bob, network)
	channelAB.addPayment(paymentAB)
	channelBC = Channel(Bob, Charlie, network)
	channelBC.addPayment(paymentBC)

	# payment goes through Channel AB and BC
	channelAB.addPayment(paymentAC)
	channelBC.addPayment(paymentAC1)


	network.addNodeList([Alice, Bob, Charlie])
	network.addChannelList([channelAB, channelBC])
	network.addPaymentList([paymentAB, paymentBC, paymentAC])
	network.runNetwork()

	a = Alice.getChCostTotal()
	b = Bob.getChCostTotal()
	c = Charlie.getChCostTotal()

	return (a+c, b)

def networkOGOppoDir2(p, freq, onlineTX, onlineTXTime, r, timeRun):
	network = Network(onlineTX, onlineTXTime, r, timeRun)
	
	Alice = Node("Alice", network)
	Bob = Node("Bob", network)
	Charlie = Node("Charlie", network)

	paymentAB = Payment(largeFrequency, largePayments, Bob, Alice)
	paymentBC = Payment(largeFrequency, largePayments, Charlie, Bob)

	channelAB = Channel(Alice, Bob, network)
	channelAB.addPayment(paymentAB)
	channelBC = Channel(Bob, Charlie, network)
	channelBC.addPayment(paymentBC)

	network.addNodeList([Alice, Bob, Charlie])
	network.addChannelList([channelAB, channelBC])
	network.addPaymentList([paymentAB, paymentBC])
	network.runNetwork()

	a = Alice.getChCostTotal()
	b = Bob.getChCostTotal()
	c = Charlie.getChCostTotal()

	return (a+c, b)


def networkDirectACOppoDir2(p, freq, onlineTX, onlineTXTime, r, timeRun):
	# set up the network
	network = Network(onlineTX, onlineTXTime, r, timeRun)

	Alice = Node("Alice", network)
	Bob = Node("Bob", network)
	Charlie = Node("Charlie", network)

	paymentAB = Payment(largeFrequency, largePayments, Bob, Alice)
	paymentBC = Payment(largeFrequency, largePayments, Charlie, Bob)
	paymentAC = Payment(freq, p, Alice, Charlie)

	channelAB = Channel(Alice, Bob, network)
	channelAB.addPayment(paymentAB)
	channelBC = Channel(Bob, Charlie, network)
	channelBC.addPayment(paymentBC)

	# Alice creates a direct channel for network 1
	channelAC = Channel(Alice, Charlie, network)
	channelAC.addPayment(paymentAC)
	
	network.addNodeList([Alice, Bob, Charlie])
	network.addChannelList([channelAB, channelBC, channelAC])
	network.addPaymentList([paymentAB, paymentBC, paymentAC])
	network.runNetwork()

	a = Alice.getChCostTotal()
	b = Bob.getChCostTotal()
	c = Charlie.getChCostTotal()

	return (a+c, b)
	

def networktransferBOppoDir2(p, freq, onlineTX, onlineTXTime, r, timeRun):
	# network 2 
	network = Network(onlineTX, onlineTXTime, r, timeRun)
	
	Alice = Node("Alice", network)
	Bob = Node("Bob", network)
	Charlie = Node("Charlie", network)

	paymentAB = Payment(largeFrequency, largePayments, Bob, Alice)
	paymentBC = Payment(largeFrequency, largePayments, Charlie, Bob)
	paymentAC = Payment(freq, p, Alice, Bob)
	paymentAC1 = Payment(freq, p, Bob, Charlie)
	paymentAC.setTransfer(paymentAC1)

	channelAB = Channel(Alice, Bob, network)
	channelAB.addPayment(paymentAB)
	channelBC = Channel(Bob, Charlie, network)
	channelBC.addPayment(paymentBC)

	# payment goes through Channel AB and BC
	channelAB.addPayment(paymentAC)
	channelBC.addPayment(paymentAC1)

	network.addNodeList([Alice, Bob, Charlie])
	network.addChannelList([channelAB, channelBC])
	network.addPaymentList([paymentAB, paymentBC, paymentAC])
	network.runNetwork()

	a = Alice.getChCostTotal()
	b = Bob.getChCostTotal()
	c = Charlie.getChCostTotal()

	return (a+c, b)


def main(p=0.1, freq=0.5, onlineTX = 5.0, onlineTXTime = 1.0, r = 0.01, timeRun = 10.0):
	(a0, b0) = networkOG(p, freq, onlineTX, onlineTXTime, r, timeRun)
	(a1, b1) = networkDirectAC(p, freq, onlineTX, onlineTXTime, r, timeRun)
	(a2, b2) = networktransferB(p, freq, onlineTX, onlineTXTime, r, timeRun)

	return (a0, b0, a1, b1, a2, b2)




def mainOppo(p=0.1, freq=0.5, onlineTX = 5.0, onlineTXTime = 1.0, r = 0.01, timeRun = 10.0):
	(a0, b0) = networkOGOppoDir(p, freq, onlineTX, onlineTXTime, r, timeRun)
	(a1, b1) = networkDirectACOppoDir(p, freq, onlineTX, onlineTXTime, r, timeRun)
	(a2, b2) = networktransferBOppoDir(p, freq, onlineTX, onlineTXTime, r, timeRun)

	return (a0, b0, a1, b1, a2, b2)





def mainOppo2(p=0.1, freq=0.5, onlineTX = 5.0, onlineTXTime = 1.0, r = 0.01, timeRun = 10.0):
	(a0, b0) = networkOGOppoDir2(p, freq, onlineTX, onlineTXTime, r, timeRun)
	(a1, b1) = networkDirectACOppoDir2(p, freq, onlineTX, onlineTXTime, r, timeRun)
	(a2, b2) = networktransferBOppoDir2(p, freq, onlineTX, onlineTXTime, r, timeRun)

	return (a0, b0, a1, b1, a2, b2)




if __name__ == '__main__':
    main()