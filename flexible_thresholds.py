import networkx as nx
import numpy as np
import random
import pickle
import sys
from numba import jit
import os
from scipy.spatial import distance
import scipy.stats
from matplotlib import pyplot as plt
import pandas as pd


def mean_confidence_interval(data, confidence=0.95):
		a = 1.0 * np.array(data)
		n = len(a)
		m, se = np.mean(a), scipy.stats.sem(a)
		h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
		return m, m-h, m+h

def testThresh(agents, mn, mx):
		if np.mean(agents) > mx:
				return False
		if np.mean(agents) < mn:
				return False
		if np.mean(agents) <= 0:
				return False
		if np.mean(agents) >= 1:
				return False

		return True


def genNet(n, k=4, pRewire=.1, type='grid'):
	# create net
	if type == 'grid': #wrap the grid
		net = nx.grid_2d_graph(int(n**.5), int(n**.5), periodic=True)
		#net = nx.grid_2d_graph(3, 3, periodic=True)
		#net = nx.grid_2d_graph(2, 2, periodic=False)
		#plotNet(net)
		#print(len(net.edges()))
		#assert(0)
		# rewire
		numRewired = 0
		#while numRewired < (pRewire * nx.number_of_nodes(net)):
		while numRewired < 1:
			tries = 0
			while tries < 100:
				tries = tries + 1
				#print([numRewired, pRewire * nx.number_of_nodes(net), tries])
				v1 = random.choice(net.nodes())
				v2 = random.choice(net.nodes())
				if not( net.has_edge(v1,v2) or v1==v2 or len(net.neighbors(v1)) <= 1): #net.neighbors is sometimes (often?) a blank set, changed so v1 needs 2 nb
					#print net.neighbors(v1)
					break
			v1Neighbors = net.neighbors(v1)
			#print v1Neighbors
			#print v1
			#print v2
			#print(len(net.edges()))
			tobeDeleted = random.choice(v1Neighbors)
			net.remove_edge(v1, tobeDeleted)
			#print(len(net.edges()))
			#print([v1, tobeDeleted, v2])
			net.add_edge(v1, v2)
			numRewired = numRewired + 1
		#plotNet(net)
		#assert(0)
		return net, nx.to_numpy_matrix(net, dtype=np.float)

	elif type == 'smallworld':
		#net = nx.connected_watts_strogatz_graph(n, k, .15)
		net = nx.connected_watts_strogatz_graph(n, k, pRewire)
		return net, nx.to_numpy_matrix(net, dtype=np.float)    
	elif type == 'pref':
		net = nx.barabasi_albert_graph(n, 2)
		return net, nx.to_numpy_matrix(net, dtype=np.float)
	elif type == 'ER':
		#net = nx.erdos_renyi_graph(n, .006)
		targetDegree = 4.
		nEdgesPossible = ((n*n)-n)/2.
		pEdge = (n * targetDegree) / (2. * nEdgesPossible)
		assert(pEdge <= 1)

		# spare networks will likely be disconnected, so try a bunch
		tries = 100
		while tries > 0:
			net = nx.erdos_renyi_graph(n, pEdge)
			if nx.number_connected_components(net) > 1:
				tries = tries - 1
			else:
				break
		return net, nx.to_numpy_matrix(net, dtype=np.float)


def influence(agents, adjMatrix, avgDegree=0, type='', haltMin=.49, haltMax=.51):

	# init some stuff for numba
	thresholds = np.zeros_like(nAgents)
	inp = np.zeros_like(nAgents)
	step = 0
	numNeighbors = np.zeros_like(nAgents)
	prevMean = 0.
	liveEdges = np.zeros_like(adjMatrix)
	pInfect = np.zeros_like(adjMatrix)
	flips = np.zeros_like(adjMatrix)

	# this controls the thresholds/pInfects for all contagion types
	globalThreshold = .5

	gstep = []

	if type == '':
		# Weighted average (more weight on self)
		w = .05
		for i in range(15):
			inp = np.dot(agents, adjMatrix)
			agents = ((1-w) * agents) + (w * inp)

	elif type == 'LT-absolute' or type == 'LT-proportional':
		# Linear Threshold Model
		# determine each agent's threshold
		if type == 'LT-proportional':
			thresholds = globalThreshold * np.random.random((1, nAgents))
		elif type == 'LT-absolute':
			thresholds = np.random.randint(low=1, high=round(avgDegree*globalThreshold)+1, size=(1, nAgents))
		#thresholds = .3 * np.ones((1, nAgents))
		#print(thresholds)
		numNeighbors = np.sum(adjMatrix, axis=0)
		prevMean = -1
		step = 1
		while not testThresh(agents, haltMin, haltMax) and (np.mean(agents) > prevMean):
		#while np.mean(agents) > prevMean:
			prevMean = np.mean(agents)
			if type == 'LT-proportional':
				# proportion of neighbors that are active
				inp = np.true_divide(np.dot(agents, adjMatrix), numNeighbors)
			elif type == 'LT-absolute':
				# absolute number of neighbors that are active
				inp = np.dot(agents, adjMatrix)
			agents = np.logical_or(agents, (inp >= thresholds)).astype(int)
			gstep.append(agents)
			step = step + 1
			#print('ltabs - step:'+str(step)+' pAct:'+str(np.mean(agents)) + ' testThresh:' + str(testThresh(agents, haltMin, haltMax)))

	elif type == 'IC':
		# Independent Cascade
		# calculate each edges probability of allowing infection
		pInfect = np.multiply(adjMatrix, np.random.random(adjMatrix.shape))
		# determine 'live' and 'blocked' edges
		flips = np.random.random(pInfect.shape)
		liveEdges = flips < pInfect
		prevMean = -1
		step = 1
		while not testThresh(agents, haltMin, haltMax) and (np.mean(agents) > prevMean):
			prevMean = np.mean(agents)
			inp = np.dot(agents, liveEdges)
			agents = np.logical_or(agents, inp).astype(int)
			gstep.append(agents)
			step = step + 1
	# print(type+' stopped at step ' + str(step) + ' '+ str(np.mean(agents)))

	return agents,gstep


####################################
#parameters of the script

networktype = 'pref' #pref, smallworld, grid, ER, korea1, korea2, ckm
hmin = .6 # minimum % of nodes active
hmax = .7 # maximum % of nodes active

dataSamples = 1000
numDatasets = 100 # number of data sets to generate
datThresh = 0

saveData = True #save the output to file?
thresh = 0.05

sizeVec = [100, 225, 400, 625, 900, 1225, 1600, 2025, 2500]
tsVec = [10, 11, 12, 13, 14, 15, 16, 17, 18]
diff_proc = "prop"

#######################################


###################################################################


for nAgents in sizeVec:
	for tstep in tsVec:

		s1 = ''.join([diff_proc, "1"])
		s2 = ''.join([diff_proc, "2"])
		activation_list = []
		for s in range(dataSamples):
			dirn = "../results/graphs_subintervals/"+"g_"+str(s+1)
			f1 = os.path.join(dirn, s1, s1+"-"+str(tstep)+"-"+networktype+'-'+str(nAgents)+'-'+str(hmin)+'-'+str(hmax)+'.pkl')
			f2 = os.path.join(dirn, s2, s2+"-"+str(tstep)+"-"+networktype+'-'+str(nAgents)+'-'+str(hmin)+'-'+str(hmax)+'.pkl')
			if os.path.exists(f1):
				with open(f1, 'rb') as f:
					d1 = pickle.load(f)
				activation_list.append(np.mean(d1))

			if os.path.exists(f2):
				with open(f2, 'rb') as f:
					d2 = pickle.load(f)
				activation_list.append(np.mean(d2))

		if len(activation_list) > 0:

			activation_mean = np.mean(activation_list)

			haltMin = round(activation_mean - 0.05, 2)
			haltMax = round(activation_mean + 0.05, 2)

			print(haltMin, haltMax)
			####################################
			dataSets = []
			dist_ic = [[] for i in range(50)]
			dist_prop = [[] for i in range(50)]
			dist_abs = [[] for i in range(50)]



			while True:
				# create agents
				agents = np.zeros((1, nAgents))
				net, adjMatrix = genNet(nAgents, type='pref')

				#         network
				for i in range(1): #make seeding more random, exp with neighbors
					#agents[0][1] = 1
					agents[0][random.randint(0, nAgents-1)] = 1.
				#print(agents)
				# seed neighbors of seeds
				for i in range(1):
					agents = np.logical_or(agents, np.dot(agents, adjMatrix)).astype(float) #all neighbors
				#print(agents)
				graphs_store_path = "../results/subroutines/" + str(nAgents) + "/" + str(tstep) + "/graphs_subintervals/g_" + str(len(dataSets) + 1)
				
				ic1_store_path = os.path.join(graphs_store_path, "ic1")
				ic2_store_path = os.path.join(graphs_store_path, "ic2")
				prop1_store_path = os.path.join(graphs_store_path, "prop1")
				prop2_store_path = os.path.join(graphs_store_path, "prop2")
				abs1_store_path = os.path.join(graphs_store_path, "abs1")
				abs2_store_path = os.path.join(graphs_store_path, "abs2")


				if not os.path.exists(graphs_store_path):
					os.makedirs(graphs_store_path)

				if not os.path.exists(ic1_store_path):
					os.makedirs(ic1_store_path)

				if not os.path.exists(ic2_store_path):
					os.makedirs(ic2_store_path)

				if not os.path.exists(prop1_store_path):
					os.makedirs(prop1_store_path)

				if not os.path.exists(prop2_store_path):
					os.makedirs(prop2_store_path)

				if not os.path.exists(abs1_store_path):
					os.makedirs(abs1_store_path)

				if not os.path.exists(abs2_store_path):
					os.makedirs(abs2_store_path)

				agents_LT_abs1 = np.copy(agents)
				agents_LT_abs2 = np.copy(agents)
				agents_LT_prop1 = np.copy(agents)
				agents_LT_prop2 = np.copy(agents)
				agents_IC1 = np.copy(agents)
				agents_IC2 = np.copy(agents)

				# let influence cascade
				avgDegree = 2*net.number_of_edges() / float(net.number_of_nodes())

				agents_LT_abs1,gs_abs1 = influence(agents_LT_abs1, adjMatrix, avgDegree=avgDegree, type='LT-absolute', haltMin=haltMin, haltMax=haltMax)
				if not testThresh(agents_LT_abs1, haltMin, haltMax):
					# print('bad data, LTabs1:\t'+str(np.mean(agents_LT_abs1)))
					continue
					
				agents_LT_abs2,gs_abs2 = influence(agents_LT_abs2, adjMatrix, avgDegree=avgDegree, type='LT-absolute', haltMin=haltMin, haltMax=haltMax)
				if not testThresh(agents_LT_abs2, haltMin, haltMax):
					# print('bad data, LTabs2:\t'+str(np.mean(agents_LT_abs2)))
					continue
					
				agents_LT_prop1,gs_prop1 = influence(agents_LT_prop1, adjMatrix, type='LT-proportional', haltMin=haltMin, haltMax=haltMax)
				if not testThresh(agents_LT_prop1, haltMin, haltMax):
					# print('bad data, LTprop1:\t'+str(np.mean(agents_LT_prop1)))
					continue
					
				agents_LT_prop2,gs_prop2 = influence(agents_LT_prop2, adjMatrix, type='LT-proportional', haltMin=haltMin, haltMax=haltMax)
				if not testThresh(agents_LT_prop2, haltMin, haltMax):
					# print('bad data, LTprop2:\t'+str(np.mean(agents_LT_prop2)))
					continue
					
				agents_IC1,gs_ic1 = influence(agents_IC2, adjMatrix, type='IC', haltMin=haltMin, haltMax=haltMax)
				if not testThresh(agents_IC1, haltMin, haltMax):
					# print('bad data, IC1:\t'+str(np.mean(agents_IC1)))
					continue

				agents_IC2,gs_ic2 = influence(agents_IC2, adjMatrix, type='IC', haltMin=haltMin, haltMax=haltMax)
				if not testThresh(agents_IC2, haltMin, haltMax):
					# print('bad data, IC2:\t'+str(np.mean(agents_IC2)))
					continue


				pac_ic1 = []
				pac_abs1 = []
				pac_prop1 = []

				pac_ic2 = []
				pac_abs2 = []
				pac_prop2 = []

				ts = 0
				for i in gs_ic1:
					pac = sum(i.tolist()[0])/len(i.tolist()[0])
					pac_ic1.append(pac)
					ts = ts + 1
					fn_sub = 'ic1-'+str(ts)+'-'+networktype+'-'+str(nAgents)+'-'+str(haltMin)+'-'+str(haltMax)
					pth = os.path.join(ic1_store_path, fn_sub)
					with open(pth+'.pkl', 'wb') as p:
						pickle.dump(i, p)

				ts = 0
				for i in gs_prop1:
					pac = sum(i.tolist()[0])/len(i.tolist()[0])
					pac_prop1.append(pac)
					ts = ts + 1
					fn_sub = 'prop1-'+str(ts)+'-'+networktype+'-'+str(nAgents)+'-'+str(haltMin)+'-'+str(haltMax)
					pth = os.path.join(prop1_store_path, fn_sub)
					with open(pth+'.pkl', 'wb') as p:
						pickle.dump(i, p)


				ts = 0
				for i in gs_abs1:
					pac = sum(i.tolist()[0])/len(i.tolist()[0])
					pac_abs1.append(pac)
					ts = ts + 1
					fn_sub = 'abs1-'+str(ts)+'-'+networktype+'-'+str(nAgents)+'-'+str(haltMin)+'-'+str(haltMax)
					pth = os.path.join(abs1_store_path, fn_sub)
					with open(pth+'.pkl', 'wb') as p:
						pickle.dump(i, p)

				ts = 0
				for i in gs_ic2:
					pac = sum(i.tolist()[0])/len(i.tolist()[0])
					pac_ic2.append(pac)
					ts = ts + 1
					fn_sub = 'ic2-'+str(ts)+'-'+networktype+'-'+str(nAgents)+'-'+str(haltMin)+'-'+str(haltMax)
					pth = os.path.join(ic2_store_path, fn_sub)
					with open(pth+'.pkl', 'wb') as p:
						pickle.dump(i, p)
				

				ts = 0
				for i in gs_prop2:
					pac = sum(i.tolist()[0])/len(i.tolist()[0])
					pac_prop2.append(pac)
					ts = ts + 1
					fn_sub = 'prop2-'+str(ts)+'-'+networktype+'-'+str(nAgents)+'-'+str(haltMin)+'-'+str(haltMax)
					pth = os.path.join(prop2_store_path, fn_sub)
					with open(pth+'.pkl', 'wb') as p:
						pickle.dump(i, p)


				ts = 0
				for i in gs_abs2:
					pac = sum(i.tolist()[0])/len(i.tolist()[0])
					pac_abs2.append(pac)
					ts = ts + 1
					fn_sub = 'abs2-'+str(ts)+'-'+networktype+'-'+str(nAgents)+'-'+str(haltMin)+'-'+str(haltMax)
					pth = os.path.join(abs2_store_path, fn_sub)
					with open(pth+'.pkl', 'wb') as p:
						pickle.dump(i, p)




				## IC -IC comparison

				# data was good, log it
				data = {}
				data['net'] = net
				data['agents_IC1'] = agents_IC1
				data['agents_IC2'] = agents_IC2
				data['agents_LT_abs1'] = agents_LT_abs1
				data['agents_LT_abs2'] = agents_LT_abs2
				data['agents_LT_prop1'] = agents_LT_prop1
				data['agents_LT_prop2'] = agents_LT_prop2

				dataSets += [data]

				if len(dataSets) == numDatasets:
					break



			if saveData == True:
				data_path = "../results/subroutines/" + str(nAgents) + "/" + str(tstep) + '/end/'
				if not os.path.exists(data_path):
					os.makedirs(data_path)

				fn = os.path.join(data_path, networktype+'-'+str(nAgents)+'-'+str(haltMin)+'-'+str(haltMax)+'.pkl')

				with open(fn, 'wb') as f:
					pickle.dump(dataSets, f)

