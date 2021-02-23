import networkx as nx
import pandas as pd

edgelist = pd.read_csv('paluck-edgelist.csv')
G = nx.from_pandas_edgelist(edgelist, source='ID', target='PEERID')

print(nx.info(G))
print(f'connected?\t{nx.is_connected(G)}')
print(f'# of connected components:\t{nx.number_connected_components(G)}')


components = nx.connected_components(G)

for i,c in enumerate(components):	
    print(f'# of nodes in {i}th component is {len(c)}')

