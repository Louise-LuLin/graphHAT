import numpy as np
import pickle as pkl
import scipy.sparse as sp
import networkx as nx
import sys
import ogb
from ogb.nodeproppred import NodePropPredDataset
print ('python version: ', sys.version_info)


dataset = 'youtube'
names = ['feature', 'label', 'graph', 'idx_train', 'idx_eval', 'idx_test']
objects = []
for i in range(len(names)):
    f = open("./data/{}/{}.bin".format(dataset, names[i]), 'rb')
    if sys.version_info > (3, 0): # if python==3.x
        objects.append(pkl.load(f, encoding='latin1'))
    else: # if python==2.x
        objects.append(pkl.load(f))
feature, label, graph, idx_train, idx_eval, idx_test = objects

print ("Below shows the type of the stored objects:")
print ("-- feature: type={}, shape={}".format(type(feature), feature.shape))
print ("-- label: type={}, shape={}, entry_type={}".format(type(label), label.shape, type(label[0][0])))
print ("-- graph: type={}, node num={}".format(type(graph), len(graph)))
print ("-- idx_train: type={}, size={}".format(type(idx_train), len(idx_train)))
print ("-- idx_eval: type={}, size={}".format(type(idx_eval), len(idx_eval)))
print ("-- idx_test: type={}, size={}".format(type(idx_test), len(idx_test)))


G = nx.from_dict_of_lists(graph)
density = nx.density(G)
print ('density: ', density)
coeffiences = nx.average_clustering(G)
print ('clustering coefficience: ', coeffiences)
triadic_closure = nx.transitivity(G)
print("Triadic closure:", triadic_closure)
betweenness_dict = nx.eigenvector_centrality(G)
avg_betweenness = np.mean(np.array([v for k, v in betweenness_dict.items()]))
print('avg betweenness: ', avg_betweenness)