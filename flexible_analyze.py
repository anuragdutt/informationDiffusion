from scipy.spatial import distance
import pickle
import pandas as pd
import os

####################################
#parameters of the script
networktype = "pref"
saveData = True #save the output to file?

sizeVec = [100, 225, 400, 625, 900, 1225, 1600, 2025, 2500]
tsVec = [10, 11, 12, 13, 14, 15, 16, 17, 18]
diff_proc = "prop"
#######################################


for nAgents in sizeVec:
	for ts in tsVec:
		sizep = "../results/subroutines/"+str(nAgents)
		if os.path.exists(sizep):
			tsp = sizep+"/"+str(ts)
						
			if os.path.exists(tsp):
				dirn = os.path.join(tsp, 'end')
				onlyfiles = [os.path.join(dirn, f) for f in os.listdir(dirn) if os.path.isfile(os.path.join(dirn, f))]
				########################## may need revision with multiple inputs
				fp = onlyfiles[0]
				haltMin = fp.split('-')[2]
				haltMax = fp.split('-')[3].split('.pkl')[0]
				#############################

				with open(fp, 'rb') as f:
					dataSets = pickle.load(f)



				count = 0
				rows_list = []
				for data in dataSets:
					count += 1
					row = {}
					for i, (key1, value1) in enumerate(data.items()):
						for j, (key2, value2) in enumerate(data.items()):
							if 'net' not in [key1, key2]:
								
								sim = distance.euclidean(value1, value2)
				#                rows_list.append({'d1':key1, 'd2':key2, 'sim': sim})
								row[key1[7:]+'-'+key2[7:]] = sim
						
					rows_list.append(row)

				df = pd.DataFrame(rows_list)
				df.loc[len(df)] = df.mean(axis = 0)
				df['sample'] = range(1,df.shape[0]+1)
				df.loc[len(df)-1,'sample'] = "mean"

				dirs = os.path.join(tsp, "termination_means")
				if not os.path.exists(dirs):
					os.makedirs(dirs)

				df1 = df.loc[:,['sample', 'IC1-IC2', 'IC1-LT_prop2', 'IC1-LT_abs2', 
						'LT_prop1-LT_prop2','LT_prop1-IC2','LT_prop1-LT_abs2',
						'LT_abs1-LT_abs2', 'LT_abs1-LT_prop2', 'LT_abs1-IC2'
						]]

				sf1 = os.path.join(dirs, networktype+'-'+str(nAgents)+'-'+str(haltMin)+'-'+str(haltMax)+'-comparison-1.csv')
				df1.to_csv(sf1)

				df2 = df[['sample', 'IC2-IC1', 'IC2-LT_prop1', 'IC2-LT_abs1', 
						'LT_prop2-LT_prop1','LT_prop2-IC1','LT_prop2-LT_abs1',
						'LT_abs2-LT_abs1', 'LT_abs2-LT_prop1', 'LT_abs2-IC1'
						]]

				sf2 = os.path.join(dirs, networktype+'-'+str(nAgents)+'-'+str(haltMin)+'-'+str(haltMax)+'-comparison-2.csv')
				df2.to_csv(sf2)

				print(nAgents, ts)



