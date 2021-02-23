import networkx as nx
import numpy as np
import random
import pickle
import sys
from numba import jit
import os
from scipy.spatial import distance,cKDTree
import scipy.stats
from matplotlib import pyplot as plt
import pandas as pd
from os import listdir
from os.path import isfile, join

import pandas as pd
####################################
#parameters of the script

networktype = 'pref' #pref, smallworld, grid, ER, korea1, korea2, ckm
# nAgents = 4**2
# nAgents = 15**2
# sidelength = int(nAgents**.5) #10^2 = 100, 32^2 = 1024
haltMin = .6 # minimum % of nodes active
haltMax = .7 # maximum % of nodes active

numDatasets = 1000 # number of data sets to generate
datThresh = 10

saveData = True #save the output to file?
thresh = 0.05

sizeVec = [100, 225, 400, 625, 900, 1225, 1600, 2025, 2500]
## sizeVec = [100]

###################################


def nearest_neighbors_kd_tree(x, y, k = 1) :
	x, y = map(np.asarray, (x, y))
	tree = cKDTree(y[:, None])    
	ordered_neighbors = tree.query(x[:, None], k)[1]
	return ordered_neighbors.tolist()


def get_vec(m_c, c1, c2, c3):
	
	mc = idf[idf[m_c] != -1].loc[:,m_c]
	mc_lab = pd.Series([m_c for i in range(len(mc))])
	mc_i = 'index_'+m_c
	mc_ts = idf[idf[m_c] != -1].loc[:,mc_i]

	comp1 = idf[idf[c1] != -1].loc[:,c1]
	comp1_lab = pd.Series([c1 for i in range(len(comp1))])
	ci_1 = 'index_'+c1
	c1_ts = idf[idf[c1] != -1].loc[:,ci_1]
	
	comp2 = idf[idf[c2] != -1].loc[:,c2]
	comp2_lab = pd.Series([c2 for i in range(len(comp2))])
	ci_2 = 'index_'+c2
	c2_ts = idf[idf[c2] != -1].loc[:,ci_2]
	
	comp3 = idf[idf[c3] != -1].loc[:,c3]
	comp3_lab = pd.Series([c3 for i in range(len(comp3))])
	ci_3 = 'index_'+c3
	c3_ts = idf[idf[c3] != -1].loc[:,ci_3]

	vec_u = pd.Series()
	vec = pd.Series()
	lab = pd.Series()
	vi = pd.Series()
	
	vec = vec.append(mc)
	lab = lab.append(mc_lab)
	vi = vi.append(mc_ts)

	vec = vec.append(comp1)
	lab = lab.append(comp1_lab)
	vi = vi.append(c1_ts)

	vec = vec.append(comp2)
	lab = lab.append(comp2_lab)
	vi = vi.append(c2_ts)

	vec = vec.append(comp3)
	lab = lab.append(comp3_lab)
	vi = vi.append(c3_ts)


	vec.index = lab.values
	vi.index = lab.values
	vsort = vec.sort_values(ascending = True)
	visort = vi.sort_values(ascending = True)

	return vsort, mc, mc_lab, comp1, comp1_lab, comp2, comp2_lab, comp3, comp3_lab, vi


def plot_prelim(m, m_lab, c1, c1_lab, c2, c2_lab, c3, c3_lab, fn, ds):
	plt.scatter(m, m_lab, color = 'red')
	plt.scatter(c1, c1_lab, color = 'blue')
	plt.scatter(c2, c2_lab, color = 'green')
	plt.scatter(c3, c3_lab, color = 'orange')

	gff = networktype+'-'+str(nAgents)+'-'+str(haltMin)+'-'+str(haltMax)+'-'+fn+'-'+str(ds+1)+'.png'
	gf = os.path.join("../results/graph_prelims", str(nAgents), fn)
	if not os.path.exists(gf):
		os.makedirs(gf)
	plt.savefig(os.path.join(gf, gff))
	plt.close()


def get_dist(m_vec,m_vi,colm, colc, pm, pc, show = 0):
	f_m = colm.replace('_', '')
	f_c = colc.replace('_', '')

	mvec = m_vec[m_vec.index == colm]
	mvi = m_vi[m_vi.index == colm]

	c1vec = m_vec[m_vec.index == colc]
	c1vi = m_vi[m_vi.index == colc]

	nn = nearest_neighbors_kd_tree(mvec, c1vec)
	c1_cv = c1vec[nn]
	c1_ci = c1vi[nn]
	diff = abs(c1_cv.values - mvec.values)

	if show == 1:
		print(c1_cv.values)
		print(mvec.values)


	dist = []
	for i in range(len(diff)):
		if diff[i] <= thresh:
			fnm = f_m+'-'+str(mvi[i])+'-'+networktype+'-'+str(nAgents)+'-'+str(haltMin)+'-'+str(haltMax)
			pthm = os.path.join(pm, fnm)
			with open(pthm+'.pkl', 'rb') as p:
				gm = pickle.load(p)

			fnm = f_c+'-'+str(c1_ci[i])+'-'+networktype+'-'+str(nAgents)+'-'+str(haltMin)+'-'+str(haltMax)
			pthm = os.path.join(pc, fnm)
			with open(pthm+'.pkl', 'rb') as p:
				gc1 = pickle.load(p)

			dist.append(distance.euclidean(gm, gc1))
		else:
			dist.append(-1)

	return dist

def remove_nans(a):
    acc = np.maximum.accumulate
    m = ~np.isnan(a)
    n = a.ndim

    if n==1:
        return a[acc(m) & acc(m[::-1])[::-1]]    
    else:
        r = np.tile(np.arange(n),n)
        per_axis_combs = np.delete(r,range(0,len(r),n+1)).reshape(-1,n-1)
        per_axis_combs_tuple = map(tuple,per_axis_combs)

        mask = []
        for i in per_axis_combs_tuple:            
            m0 = m.any(i)            
            mask.append(acc(m0) & acc(m0[::-1])[::-1])
        return a[np.ix_(*mask)]


def nan_helper(y):

    return np.isnan(y), lambda z: z.nonzero()[0]

def get_plotval(p):

	p1 = remove_nans(np.array(p))
	y = np.array(p1)
	nans, x = nan_helper(p1)
	y[nans]= np.interp(x(nans), x(~nans), y[~nans])
	return y

def comparison_plots(val1, val2, val3, l1, l2, l3, fn):
	p1 = []
	for i in val1:
		if len(i) > datThresh:
			p1.append(sum(i)/len(i))
		else:
			p1.append(np.nan)

	p1 = get_plotval(p1)

	p2 = []
	for i in val2:
		if len(i) > datThresh:
			p2.append(sum(i)/len(i))
		else:
			p2.append(np.nan)

	p2 = get_plotval(p2)

	p3 = []
	for i in val3:
		if len(i) > datThresh:
			p3.append(sum(i)/len(i))
		else:
			p3.append(np.nan)


	plt.plot(range(1,len(p1)+1), p1, color = 'red', label = l1)
	plt.scatter(range(1,len(p1)+1), p1, color = 'red')

	plt.plot(range(1,len(p2)+1), p2, color = 'blue', label = l2)
	plt.scatter(range(1,len(p2)+1), p2, color = 'blue')

	plt.plot(range(1,len(p3)+1), p3, color = 'green', label = l3)
	plt.scatter(range(1,len(p3)+1), p3, color = 'green')

	plt.xlabel("time-step")
	plt.ylabel("Euclidean distance")
	plt.legend()
	fname = os.path.join("../results/comparison_plots",fn+'.png')
	plt.savefig(fname)
	plt.close()



t_ic1_ic2 = np.full([50, len(sizeVec), numDatasets], np.nan)
t_ic1_prop2 = np.full([50, len(sizeVec), numDatasets], np.nan)
t_ic1_abs2 = np.full([50, len(sizeVec), numDatasets], np.nan)

t_prop1_ic2 = np.full([50, len(sizeVec), numDatasets], np.nan)
t_prop1_prop2 = np.full([50, len(sizeVec), numDatasets], np.nan)
t_prop1_abs2 = np.full([50, len(sizeVec), numDatasets], np.nan)

t_abs1_ic2 = np.full([50, len(sizeVec), numDatasets], np.nan)
t_abs1_prop2 = np.full([50, len(sizeVec), numDatasets], np.nan)
t_abs1_abs2 = np.full([50, len(sizeVec), numDatasets], np.nan)

t_ic2_ic1 = np.full([50, len(sizeVec), numDatasets], np.nan)
t_ic2_prop1 = np.full([50, len(sizeVec), numDatasets], np.nan)
t_ic2_abs1 = np.full([50, len(sizeVec), numDatasets], np.nan)

t_prop2_ic1 = np.full([50, len(sizeVec), numDatasets], np.nan)
t_prop2_prop1 = np.full([50, len(sizeVec), numDatasets], np.nan)
t_prop2_abs1 = np.full([50, len(sizeVec), numDatasets], np.nan)

t_abs2_ic1 = np.full([50, len(sizeVec), numDatasets], np.nan)
t_abs2_prop1 = np.full([50, len(sizeVec), numDatasets], np.nan)
t_abs2_abs1 = np.full([50, len(sizeVec), numDatasets], np.nan)



for count in range(len(sizeVec)):
	nAgents = sizeVec[count]
	print(nAgents)

	ic1_ic2 = [[] for i in range(50)]
	ic1_prop2 = [[] for i in range(50)]
	ic1_abs2 = [[] for i in range(50)]

	prop1_ic2 = [[] for i in range(50)]
	prop1_prop2 = [[] for i in range(50)]
	prop1_abs2 = [[] for i in range(50)]

	abs1_ic2 = [[] for i in range(50)]
	abs1_prop2 = [[] for i in range(50)]
	abs1_abs2 = [[] for i in range(50)]

	ic2_ic1 = [[] for i in range(50)]
	ic2_prop1 = [[] for i in range(50)]
	ic2_abs1 = [[] for i in range(50)]

	prop2_ic1 = [[] for i in range(50)]
	prop2_prop1 = [[] for i in range(50)]
	prop2_abs1 = [[] for i in range(50)]

	abs2_ic1 = [[] for i in range(50)]
	abs2_prop1 = [[] for i in range(50)]
	abs2_abs1 = [[] for i in range(50)]

	for dataSets in range(numDatasets):
		if (dataSets+1)%50 == 0:
			print(dataSets)
		graphs_store_path = "../results/graphs_subintervals/g_" + str(dataSets + 1)

		ic1_store_path = os.path.join(graphs_store_path, "ic1")
		ic2_store_path = os.path.join(graphs_store_path, "ic2")
		prop1_store_path = os.path.join(graphs_store_path, "prop1")
		prop2_store_path = os.path.join(graphs_store_path, "prop2")
		abs1_store_path = os.path.join(graphs_store_path, "abs1")
		abs2_store_path = os.path.join(graphs_store_path, "abs2")

		ts_sub = 'ts-'+networktype+'-'+str(nAgents)+'-'+str(haltMin)+'-'+str(haltMax)+'-'+str(dataSets+1)+'.csv'
		path_read = os.path.join("../results/time_activation", ts_sub)
		idf = pd.read_csv(path_read)

		## IC1 comparison
		ic1_vec, ic1, ic1_lab, ic2, ic2_lab, prop2, prop2_lab, abs2, abs2_lab, ic1_vi = get_vec("ic_1", "ic_2", "prop_2", "abs_2")
		plot_prelim(ic1, ic1_lab, ic2, ic2_lab, prop2, prop2_lab, abs2, abs2_lab, "ic1", dataSets)

		d = get_dist(ic1_vec, ic1_vi, "ic_1", "ic_2", ic1_store_path, ic2_store_path)
		for i in range(len(d)):
			if d[i] != -1:
				ic1_ic2[i].append(d[i])
				t_ic1_ic2[i][count][dataSets] = d[i]

		# if dataSets == 1 and nAgents == 100:
		# 	d = get_dist(ic1_vec, ic1_vi, "ic_1", "ic_2", ic1_store_path, ic2_store_path,1)
		# 	print(d)


		d = get_dist(ic1_vec, ic1_vi, "ic_1", "prop_2", ic1_store_path, prop2_store_path)
		for i in range(len(d)):
			if d[i] != -1:
				ic1_prop2[i].append(d[i])
				t_ic1_prop2[i][count][dataSets] = d[i]


		d = get_dist(ic1_vec, ic1_vi, "ic_1", "abs_2", ic1_store_path, abs2_store_path)
		for i in range(len(d)):
			if d[i] != -1:
				ic1_abs2[i].append(d[i])
				t_ic1_abs2[i][count][dataSets] = d[i]


		## prop1 comparison
		prop1_vec, prop1, prop1_lab, prop2, prop2_lab, ic2, ic2_lab, abs2, abs2_lab, prop1_vi = get_vec("prop_1", "prop_2", "ic_2", "abs_2")
		plot_prelim(prop1, prop1_lab, prop2, prop2_lab, ic2, ic2_lab, abs2, abs2_lab, "prop1", dataSets)
		d = get_dist(prop1_vec, prop1_vi, "prop_1", "prop_2", prop1_store_path, prop2_store_path)
		for i in range(len(d)):
			if d[i] != -1:
				prop1_prop2[i].append(d[i])
				t_prop1_prop2[i][count][dataSets] = d[i]


		d = get_dist(prop1_vec, prop1_vi, "prop_1", "ic_2", prop1_store_path, ic2_store_path)
		for i in range(len(d)):
			if d[i] != -1:
				prop1_ic2[i].append(d[i])
				t_prop1_ic2[i][count][dataSets] = d[i]

		d = get_dist(prop1_vec, prop1_vi, "prop_1", "abs_2", prop1_store_path, abs2_store_path)
		for i in range(len(d)):
			if d[i] != -1:
				prop1_abs2[i].append(d[i])
				t_prop1_abs2[i][count][dataSets] = d[i]


		## abs1 comparison
		abs1_vec, abs1, abs1_lab, abs2, abs2_lab, ic2, ic2_lab, prop2, prop2_lab, abs1_vi = get_vec("abs_1", "abs_2", "ic_2", "prop_2")
		plot_prelim(abs1, abs1_lab, abs2, abs2_lab, ic2, ic2_lab, prop2, prop2_lab, "abs1", dataSets)
		d = get_dist(abs1_vec, abs1_vi, "abs_1", "abs_2", abs1_store_path, abs2_store_path)
		for i in range(len(d)):
			if d[i] != -1:
				abs1_abs2[i].append(d[i])
				t_abs1_abs2[i][count][dataSets] = d[i]

		d = get_dist(abs1_vec, abs1_vi, "abs_1", "ic_2", abs1_store_path, ic2_store_path)
		for i in range(len(d)):
			if d[i] != -1:
				abs1_ic2[i].append(d[i])
				t_abs1_ic2[i][count][dataSets] = d[i]

		d = get_dist(abs1_vec, abs1_vi, "abs_1", "prop_2", abs1_store_path, prop2_store_path)
		for i in range(len(d)):
			if d[i] != -1:
				abs1_prop2[i].append(d[i])
				t_abs1_prop2[i][count][dataSets] = d[i]


		## IC2 comparison
		ic2_vec, ic2, ic2_lab, ic1, ic1_lab, prop1, prop1_lab, abs1, abs1_lab, ic2_vi = get_vec("ic_2", "ic_1", "prop_1", "abs_1")
		plot_prelim(ic2, ic2_lab, ic1, ic1_lab, prop1, prop1_lab, abs1, abs1_lab, "ic2", dataSets)

		d = get_dist(ic2_vec, ic2_vi, "ic_2", "ic_1", ic2_store_path, ic1_store_path)
		for i in range(len(d)):
			if d[i] != -1:
				ic2_ic1[i].append(d[i])
				t_ic2_ic1[i][count][dataSets] = d[i]

		d = get_dist(ic2_vec, ic2_vi, "ic_2", "prop_1", ic2_store_path, prop1_store_path)
		for i in range(len(d)):
			if d[i] != -1:
				ic2_prop1[i].append(d[i])
				t_ic2_prop1[i][count][dataSets] = d[i]

		d = get_dist(ic2_vec, ic2_vi, "ic_2", "abs_1", ic2_store_path, abs1_store_path)
		for i in range(len(d)):
			if d[i] != -1:
				ic2_abs1[i].append(d[i])
				t_ic2_abs1[i][count][dataSets] = d[i]


		## prop2 comparison
		prop2_vec, prop2, prop2_lab, prop1, prop1_lab, ic1, ic1_lab, abs1, abs1_lab, prop2_vi = get_vec("prop_2", "prop_1", "ic_1", "abs_1")
		plot_prelim(prop2, prop2_lab, prop1, prop1_lab, ic1, ic1_lab, abs1, abs1_lab, "prop2", dataSets)
		d = get_dist(prop2_vec, prop2_vi, "prop_2", "prop_1", prop2_store_path, prop1_store_path)
		for i in range(len(d)):
			if d[i] != -1:
				prop2_prop1[i].append(d[i])
				t_prop2_prop1[i][count][dataSets] = d[i]

		d = get_dist(prop2_vec, prop2_vi, "prop_2", "ic_1", prop2_store_path, ic1_store_path)
		for i in range(len(d)):
			if d[i] != -1:
				prop2_ic1[i].append(d[i])
				t_prop2_ic1[i][count][dataSets] = d[i]

		d = get_dist(prop2_vec, prop2_vi, "prop_2", "abs_1", prop2_store_path, abs1_store_path)
		for i in range(len(d)):
			if d[i] != -1:
				prop2_abs1[i].append(d[i])
				t_prop2_abs1[i][count][dataSets] = d[i]


		## abs2 comparison
		abs2_vec, abs2, abs2_lab, abs1, abs1_lab, ic1, ic1_lab, prop1, prop1_lab, abs2_vi = get_vec("abs_2", "abs_1", "ic_1", "prop_1")
		plot_prelim(abs2, abs2_lab, abs1, abs1_lab, ic1, ic1_lab, prop1, prop1_lab, "abs2", dataSets)
		d = get_dist(abs2_vec, abs2_vi, "abs_2", "abs_1", abs2_store_path, abs1_store_path)
		for i in range(len(d)):
			if d[i] != -1:
				abs2_abs1[i].append(d[i])
				t_abs2_abs1[i][count][dataSets] = d[i]

		d = get_dist(abs2_vec, abs2_vi, "abs_2", "ic_1", abs2_store_path, ic1_store_path)
		for i in range(len(d)):
			if d[i] != -1:
				abs2_ic1[i].append(d[i])
				t_abs2_ic1[i][count][dataSets] = d[i]

		d = get_dist(abs2_vec, abs2_vi, "abs_2", "prop_1", abs2_store_path, prop1_store_path)
		for i in range(len(d)):
			if d[i] != -1:
				abs2_prop1[i].append(d[i])
				t_abs2_prop1[i][count][dataSets] = d[i]



	fn = "ic1/ic1_comparison-"+networktype+'-'+str(nAgents)+'-'+str(haltMin)+'-'+str(haltMax)
	comparison_plots(ic1_ic2,ic1_prop2,ic1_abs2,"ic1-ic2","ic1-prop2","ic1-abs2",fn)

	fn = "prop1/prop1_comparison-"+networktype+'-'+str(nAgents)+'-'+str(haltMin)+'-'+str(haltMax)
	comparison_plots(prop1_ic2,prop1_prop2,prop1_abs2,"prop1-ic2","prop1-prop2","prop1-abs2",fn)

	fn = "abs1/abs1_comparison-"+networktype+'-'+str(nAgents)+'-'+str(haltMin)+'-'+str(haltMax)
	comparison_plots(abs1_ic2,abs1_prop2,abs1_abs2,"abs1-ic2","abs1-prop2","abs1-abs2",fn)


	fn = "ic2/ic2_comparison-"+networktype+'-'+str(nAgents)+'-'+str(haltMin)+'-'+str(haltMax)
	comparison_plots(ic2_ic1,ic2_prop1,ic2_abs1,"ic2-ic1","ic2-prop1","ic2-abs1",fn)

	fn = "prop2/prop2_comparison-"+networktype+'-'+str(nAgents)+'-'+str(haltMin)+'-'+str(haltMax)
	comparison_plots(prop2_ic1,prop2_prop1,prop2_abs1,"prop2-ic1","prop2-prop1","prop2-abs1",fn)

	fn = "abs2/abs2_comparison-"+networktype+'-'+str(nAgents)+'-'+str(haltMin)+'-'+str(haltMax)
	comparison_plots(abs2_ic1,abs2_prop1,abs2_abs1,"abs2-ic1","abs2-prop1","abs2-abs1",fn)



fn = os.path.join('../results/pairwise_intermediate_datasets', 'ic1_ic2-'+networktype+'-'+str(haltMin)+'-'+str(haltMax)+'.pkl')
with open(fn, 'wb') as f:
	pickle.dump(t_ic1_ic2, f)

fn = os.path.join('../results/pairwise_intermediate_datasets', 'ic1_prop2-'+networktype+'-'+str(haltMin)+'-'+str(haltMax)+'.pkl')
with open(fn, 'wb') as f:
	pickle.dump(t_ic1_prop2, f)

fn = os.path.join('../results/pairwise_intermediate_datasets', 'ic1_abs2-'+networktype+'-'+str(haltMin)+'-'+str(haltMax)+'.pkl')
with open(fn, 'wb') as f:
	pickle.dump(t_ic1_abs2, f)


fn = os.path.join('../results/pairwise_intermediate_datasets', 'prop1_prop2-'+networktype+'-'+str(haltMin)+'-'+str(haltMax)+'.pkl')
with open(fn, 'wb') as f:
	pickle.dump(t_prop1_prop2, f)

fn = os.path.join('../results/pairwise_intermediate_datasets', 'prop1_ic2-'+networktype+'-'+str(haltMin)+'-'+str(haltMax)+'.pkl')
with open(fn, 'wb') as f:
	pickle.dump(t_prop1_ic2, f)

fn = os.path.join('../results/pairwise_intermediate_datasets', 'prop1_abs2-'+networktype+'-'+str(haltMin)+'-'+str(haltMax)+'.pkl')
with open(fn, 'wb') as f:
	pickle.dump(t_prop1_abs2, f)


fn = os.path.join('../results/pairwise_intermediate_datasets', 'abs1_abs2-'+networktype+'-'+str(haltMin)+'-'+str(haltMax)+'.pkl')
with open(fn, 'wb') as f:
	pickle.dump(t_abs1_abs2, f)

fn = os.path.join('../results/pairwise_intermediate_datasets', 'abs1_prop2-'+networktype+'-'+str(haltMin)+'-'+str(haltMax)+'.pkl')
with open(fn, 'wb') as f:
	pickle.dump(t_abs1_prop2, f)

fn = os.path.join('../results/pairwise_intermediate_datasets', 'abs1_ic2-'+networktype+'-'+str(haltMin)+'-'+str(haltMax)+'.pkl')
with open(fn, 'wb') as f:
	pickle.dump(t_abs1_ic2, f)


################################################################################


fn = os.path.join('../results/pairwise_intermediate_datasets', 'ic2_ic1-'+networktype+'-'+str(haltMin)+'-'+str(haltMax)+'.pkl')
with open(fn, 'wb') as f:
	pickle.dump(t_ic2_ic1, f)

fn = os.path.join('../results/pairwise_intermediate_datasets', 'ic2_prop1-'+networktype+'-'+str(haltMin)+'-'+str(haltMax)+'.pkl')
with open(fn, 'wb') as f:
	pickle.dump(t_ic2_prop1, f)
	
fn = os.path.join('../results/pairwise_intermediate_datasets', 'ic2_abs1-'+networktype+'-'+str(haltMin)+'-'+str(haltMax)+'.pkl')
with open(fn, 'wb') as f:
	pickle.dump(t_ic2_abs1, f)


fn = os.path.join('../results/pairwise_intermediate_datasets', 'prop2_prop1-'+networktype+'-'+str(haltMin)+'-'+str(haltMax)+'.pkl')
with open(fn, 'wb') as f:
	pickle.dump(t_prop2_prop1, f)

fn = os.path.join('../results/pairwise_intermediate_datasets', 'prop2_ic1-'+networktype+'-'+str(haltMin)+'-'+str(haltMax)+'.pkl')
with open(fn, 'wb') as f:
	pickle.dump(t_prop2_ic1, f)

fn = os.path.join('../results/pairwise_intermediate_datasets', 'prop2_abs1-'+networktype+'-'+str(haltMin)+'-'+str(haltMax)+'.pkl')
with open(fn, 'wb') as f:
	pickle.dump(t_prop2_abs1, f)


fn = os.path.join('../results/pairwise_intermediate_datasets', 'abs2_abs1-'+networktype+'-'+str(haltMin)+'-'+str(haltMax)+'.pkl')
with open(fn, 'wb') as f:
	pickle.dump(t_abs2_abs1, f)

fn = os.path.join('../results/pairwise_intermediate_datasets', 'abs2_prop1-'+networktype+'-'+str(haltMin)+'-'+str(haltMax)+'.pkl')
with open(fn, 'wb') as f:
	pickle.dump(t_abs2_prop1, f)

fn = os.path.join('../results/pairwise_intermediate_datasets', 'abs2_ic1-'+networktype+'-'+str(haltMin)+'-'+str(haltMax)+'.pkl')
with open(fn, 'wb') as f:
	pickle.dump(t_abs2_ic1, f)







# d1 = t_ic1_ic2
# d2 = t_ic1_prop2
# d3 = t_ic1_abs2


# for ts in range(d1.shape[0]):
# 	for nAgents in range(d2.shape[0]):
# 		for ds in range(d3.shape[0]):


# numTimeSteps = max()