import networkx as nx 
import numpy as np 
import random
from sklearn.metrics import roc_auc_score

from deepwalk import DeepWalk

random.seed(616)
edges = np.load('../tencent/train_edges.npy')
G = nx.Graph()
for i in range(169209):
    G.add_node(i)
G.add_edges_from(edges)
        
deepwalk = DeepWalk(G, emb_size=128, length_walk=50, num_walks=10, window_size=10, num_iters=2)
w2v = deepwalk.train(workers=4, is_loadmodel=False, is_loaddata=False)

pos_test = np.load('../tencent/test_edges.npy')
neg_test = np.load('../tencent/test_edges_false.npy')

y_true = [True]*pos_test.shape[0] + [False]*neg_test.shape[0]
X = np.vstack([pos_test, neg_test])

print('Testing...')
y_score = []
for u, v in X:
    y_score.append(w2v.wv.similarity(str(u), str(v)))

auc_test = roc_auc_score(y_true, y_score)
print('Tencent, test AUC:', auc_test)



