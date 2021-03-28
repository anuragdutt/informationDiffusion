import os
os.environ["PATH"] += os.pathsep + 'C:/Users/anura/anaconda3/Library/bin/graphviz'
import sys
import networkx as nx
import numpy as np
import random
import pickle
import sys
from numba import jit
import elfi
import sklearn as sk
import scipy
from scipy.spatial import distance
import pandas as pd
import sys
import pickle
import statistics
import importlib
print(sys.version)


global final_activation
global generated_graph
global all_i

global ic_count
global ltp_count
global lta_count

global ic_ltp
global ic_lta
global ltp_lta

global count_samples

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


def testThresh(agents, agents_ini, mn, mx):
	
	if np.mean(agents) == 0:
		print("stopping because no nodes are activated in the initial state")
		return agents, False
	
	if np.mean(agents) >= mx:
		print("Bad data discarding the dataset because activation reached ", np.mean(agents))
		agents = agents_ini
		return agents, False
	
	if np.mean(agents) <= mn:
		return agents, False
	
	
	if np.mean(agents) > mn and np.mean(agents) < mx:
		return agents, True
		

	return False


def ltProp(agents, adjMatrix, nAgents, avgDegree=0, haltMin=.49, haltMax=.51, rs = None):
	# init some stuff for numba
	thresholds = np.zeros_like(nAgents)
	inp = np.zeros_like(nAgents)
	step = 0
	numNeighbors = np.zeros_like(nAgents)
	prevMean = 0.
	liveEdges = np.zeros_like(adjMatrix)
	pInfect = np.zeros_like(adjMatrix)
	flips = np.zeros_like(adjMatrix)
	

	if rs is None:
		thresholds = globalThreshold * np.random.random((1, nAgents))
	else:
		thresholds = globalThreshold * rs.random((1, nAgents))
	
	numNeighbors = np.sum(adjMatrix, axis=0)
	prevMean = -1
	step = 1
#     print("stopping conditions: ", haltMin, haltMax)
	
	agents_ini = agents
	
	loop_manager = False
	
	while not loop_manager  and (np.mean(agents) > prevMean):
		
		agents, loop_manager = testThresh(agents, agents_ini, haltMin, haltMax)
#         print("state activation: ", prevMean)

		#while np.mean(agents) > prevMean:
		prevMean = np.mean(agents)
		inp = np.true_divide(np.dot(agents, adjMatrix), numNeighbors)
		agents = np.logical_or(agents, (inp >= thresholds)).astype(int)

		step = step + 1

#     print("state activation: ", prevMean)
	generated_graph.append(agents)
	final_activation.append(prevMean)
	#print('LT-proportional stopped at step ' + str(step) + ' '+ str(np.mean(agents)))
	return agents


def ltAbs(agents, adjMatrix, nAgents, avgDegree=0, haltMin=.49, haltMax=.51, rs = None):
	# init some stuff for numba
	thresholds = np.zeros_like(nAgents)
	inp = np.zeros_like(nAgents)
	step = 0
	numNeighbors = np.zeros_like(nAgents)
	prevMean = 0
	liveEdges = np.zeros_like(adjMatrix)
	pInfect = np.zeros_like(adjMatrix)
	flips = np.zeros_like(adjMatrix)

	# this controls the thresholds/pInfects for all contagion types
	
	if rs is None:
		thresholds = np.random.randint(low=1, high=round(avgDegree*globalThreshold), size=(1, nAgents))
	else:
		thresholds = rs.randint(low=1, high=round(avgDegree*globalThreshold), size=(1, nAgents))
		
	numNeighbors = np.sum(adjMatrix, axis=0)
	prevMean = -1
	step = 1
	
	agents_ini = agents
	loop_manager = False
	while not loop_manager and (np.mean(agents) > prevMean):
		
		agents, loop_manager = testThresh(agents, agents_ini, haltMin, haltMax)
#         print("state activation: ", prevMean)

		prevMean = np.mean(agents)
		inp = np.dot(agents, adjMatrix)
		agents = np.logical_or(agents, (inp >= thresholds)).astype(int)
		step = step + 1

#     print("state activation: ", prevMean)

	# print('LT-absolute stopped at step ' + str(step) + ' '+ str(np.mean(agents)))
	final_activation.append(prevMean)
	generated_graph.append(agents)
	return agents

def generateGraph(net, adjMatrix, pp, batch_size = 1, random_state=None):
	gg = []

	while True:
		nAgents = nx.number_of_nodes(net)
		agents = np.zeros((1, nAgents))
		nodes = nx.nodes(net)
		seed_nodes = []
		inx = []
		count = 0
		for n in nodes:
			if n in seeds['ID'].tolist():
				seed_nodes.append(n)
				inx.append(count)
			count += 1
		
		agents[:,inx] = 1

		es_nodes = []
		es_inx = []
		count_es = 0
		num_es = 0
		for n in nodes:
			if n in es['ID'].tolist():
				es_nodes.append(n)
				es_inx.append(count_es)
				num_es += 1
			count_es += 1
	
		p_activation = num_es/count_es
#         print("final_activation: ", p_activation)
#         print("Adjaecency Matrix: ", adjMatrix.shape)
		# seed neighbors of seeds
#         print("zero initialized agents: ", agents[agents == 1].shape)
#         for i in range(1):
#             agents = np.logical_or(agents, np.dot(agents, adjMatrix)).astype(float) #all neighbors

		avgDegree = 2*net.number_of_edges() / float(net.number_of_nodes())
#         print("initially activated agents: ", agents[agents == 1].shape)
		t = 1/2
	
#         if pp <= t:
#             agents_type = IC(agents, adjMatrix, nAgents, avgDegree=avgDegree, 
#                                  haltMin = max(0,p_activation-activation_ci), 
#                                  haltMax = min(p_activation+activation_ci,1), 
#                                  rs = random_state)

		if pp <= t:
			agents_type = ltProp(agents, adjMatrix, nAgents, avgDegree=avgDegree, 
								 haltMin = max(0,p_activation-activation_ci), 
								 haltMax = min(p_activation+activation_ci,1), 
								 rs = random_state)


		else:
			agents_type = ltAbs(agents, adjMatrix, nAgents, avgDegree=avgDegree, 
								haltMin = max(0,p_activation-activation_ci), 
								haltMax = min(p_activation+activation_ci,1), 
								rs = random_state)


		#if not testThresh(agents_type, haltMin, haltMax):
		 #   print('bad data, LTabs1:\t'+str(np.mean(agents_type)))
	   #     continue
		gparam = {}
		gparam['init'] = net
		gparam['graph'] = agents_type
		
		gg.append(gparam['graph'])
		if len(gg) >= batch_size:
			break
	
#     print("final graph: ", gg[0])
	return gg[0]


def compileG(ll, batch_size = 1, random_state=None):
	return generateGraph(net_s, adjMat_s, ll[0], batch_size, random_state)


def yMetric(x, i = 0):
	try:
		ttmp = x[0,i]
		all_i.append(i)
	except:
		print(i)
		print(all_i)
		sys.exit()
	return x[0,i]


def eucMultiArgs(X, Y):
	dist = np.linalg.norm(X - Y) 
	return np.array([dist])




if __name__ == "__main__":

	#for i in range(len(sglist)):
	#    print('# of nodes in {i}th component is  - ', str(np.mean(gmat[i])))
	seeds = pd.read_csv("../data/icpsr/DS0001/paluck-seed.csv")
	seeds['ID'] = ((seeds['SCHIDW2'] * 1000) + pd.to_numeric(seeds['ID'], errors='coerce'))
	es = pd.read_csv("../data/icpsr/DS0001/paluck-endstate.csv")
	es['ID'] = ((es['SCHIDW2'] * 1000) + pd.to_numeric(es['ID'], errors='coerce'))


		#for i in range(len(sglist)):
#    print('# of nodes in {i}th component is  - ', str(np.mean(gmat[i])))
	seeds = pd.read_csv("../data/icpsr/DS0001/paluck-seed.csv")
	seeds['ID'] = ((seeds['SCHIDW2'] * 1000) + pd.to_numeric(seeds['ID'], errors='coerce'))
	es = pd.read_csv("../data/icpsr/DS0001/paluck-endstate.csv")
	es['ID'] = ((es['SCHIDW2'] * 1000) + pd.to_numeric(es['ID'], errors='coerce'))

	ng = int(sys.argv[1])
	N = int(sys.argv[2]) # samples for rejection sampling or sequential monte carlo

	# ng = 4
	# seed = 20170530  # this will be separately given by ELFI
	# np.random.seed(seed)
	# schedule = [70, 50]
	schedule_per = [90, 60, 30, 10]
	# schedule_per = [70, 50] # percentile of rejection sampling to be used as schedule thresholds
	
	networktype = 'pref' #pref, smallworld, grid, ER, korea1, korea2, ckm

	ic_count = 0
	ltp_count = 0
	lta_count = 0

	ic_ltp = 0
	ic_lta = 0
	ltp_lta = 0

	count_samples = 0

	print(nx.__version__)
	#parameters of the script
	path = '../data/icpsr/DS0001/paluck-edgelist.csv'
	globalThreshold = 1.5
	activation_ci = 0.075
	cascadeParameter = 0.5

	edgelist = pd.read_csv(path)
	G = nx.from_pandas_edgelist(edgelist, source='ID', target='PEERID')
	print(nx.info(G))
	print(nx.number_of_nodes(G))
	print(f'connected?\t{nx.is_connected(G)}')
	print(f'# of connected components:\t{nx.number_connected_components(G)}')

	components = nx.connected_components(G)
	sglist = [G.subgraph(c) for c in nx.connected_components(G)]

	gmat = []
	for g in sglist:
		gmat.append(nx.to_numpy_matrix(g, dtype=np.float))


	graph_size_list = [nx.number_of_nodes(g) for g in sglist]
	graph_size_series = pd.Series(graph_size_list)
	ordered_graph_series = graph_size_series.sort_values()    
	ordered_graph_list = ordered_graph_series.index.tolist()


	reslist = []

	# for ng in range(nx.number_connected_components(G)):

	# for ng in ordered_graph_list:
		#n,a = getGraphFromEdgelist(path)

	print("Importance Sampling for graph: ", ng+1)
	net_s = sglist[ng]
	adjMat_s = gmat[ng]
	nAgobs = nx.number_of_nodes(net_s)
	agobs = np.zeros((1, nAgobs))
	nodeobs = nx.nodes(net_s)
	es_nodeobs = []
	es_inxobs = []
	count_esobs = 0
	num_esobs = 0

	seed_node_count = 0

	final_activation = []
	generated_graph = []
	all_i = []

	for s in nodeobs:
		if s in seeds['ID'].tolist():
			seed_node_count += 1

	print("number of seed nodes: ", seed_node_count)

	if seed_node_count > 0:
		for n in nodeobs:
			if n in es['ID'].tolist():
				es_nodeobs.append(n)
				es_inxobs.append(count_esobs)
				num_esobs += 1
			count_esobs += 1

		print("number of total nodes: ", count_esobs)

		agobs[:,es_inxobs] = 1

		gr_obs = np.matrix(agobs).astype(int)
		dcom = "d = elfi.Distance('euclidean'"


		prop_prob = elfi.Prior('uniform', 0, 1)
		Y = elfi.Simulator(compileG, prop_prob, observed = gr_obs)
		ret = []

		for i in range(gr_obs.shape[1]):
			st = str(i)
			var_s = ''.join(['s',st])
			ret.append(var_s)
			com = var_s + ' = ' + 'elfi.Summary(yMetric, Y, ' + st + ')'
			exec(com)
			if i == 0:
				dcom = dcom + ',s' + str(i)
			else:
				dcom = dcom + ',s' + str(i)
		dcom = dcom + ')'

		exec(dcom)


		# d = elfi.Distance('euclidean',s7)
		# d = elfi.Distance('euclidean',s0, s1, s2, s3, s4, s5, s6)

		# running rejection sampling
		rej = elfi.Rejection(d, batch_size=1)
		rej_result = rej.sample(N, quantile=0.1)

		distl = []
		for ggr in generated_graph:
			distl.append(eucMultiArgs(ggr, gr_obs)[0])

		print("Schedule thresholds: ", np.percentile(np.array(distl), schedule_per))
		print("***********************************")
		print("***********************************")

		final_activation = []
		generated_graph = []
		schedule = list(np.percentile(np.array(distl), schedule_per))

	#         print("observation matrix shape after transformation: ", gr_obs.shape[1])

		smc = elfi.SMC(d, batch_size=1)

		result = smc.sample(N, schedule)
		print(result.summary(all = True))

		print(count_esobs, min(all_i), max(all_i))
		print("******************************************************************")

			
		count_samp = pd.to_numeric(pd.Series(list(result.samples['prop_prob'])))
		count_samples = len(count_samp)
		ltp_count = len(count_samp[count_samp >= 0.5])
		lta_count = len(count_samp[count_samp < 0.5])
		ltp_lta = len(count_samp[(count_samp < 0.5 ) | (count_samp >= 0.5)])


		# print("final activations check: ", final_activation)
		reslist.append([ng+1, 
						result.samples['prop_prob'].mean(), 
						np.median(result.samples['prop_prob']),
						statistics.stdev(result.samples['prop_prob']),
						ltp_count/count_samples,
						lta_count/count_samples,
						ltp_lta/count_samples,
						seed_node_count/count_esobs, 
						num_esobs/count_esobs, np.max(final_activation), 
						schedule,
						result.samples['prop_prob']])
		resdf = pd.DataFrame(reslist, 
							 columns = ["graph_index", 
										"probability_parameter_mean", 
										"probability_parameter_median",
										"probability_parameter_stdev",
										"ltp_probability_inference",
										"lta_probability_inference",
										"ltp_or_lta_probability_inference",
										"seed_activation",
										"actual_end_activation", 
										"observed_max_end_activation_10_samples",
										"distribution_thresholds",
										"probability_parameter_samples"])
		fname = "../results/sequential_monte_carlo/LT/N_" + str(N) + ".csv" 
		if ng == 0:
			resdf.to_csv(fname, index = False)
		else:
			prevdf = pd.read_csv(fname)
			df_all = pd.concat([prevdf, resdf], ignore_index=True)
			df_all.to_csv(fname, index = False)

		dirname = "N_" + str(N)
		dirpath = "../results/sequential_monte_carlo/LT/result_objects/" + dirname
		if not os.path.exists(dirpath):
			os.makedirs(dirpath)
		filename = os.path.join(dirpath, "graph_" + str(ng+1) +".pkl")
		filehandler = open(filename, 'wb')
		pickle.dump(result, filehandler)
		filehandler.close()

	else:
		print("Number of seed nodes is zero. Hence not running")
