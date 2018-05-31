import networkx as nx 
import numpy as np 
import random
from gensim.models import Word2Vec

class DeepWalk:
	def __init__(self, G, emb_size=128, length_walk=40, num_walks=10, window_size=10, num_iters=1):
		self.G = G
        self.emb_size = emb_size
		self.length_walk = length_walk
		self.num_walks = num_walks
		self.window_size = window_size
		self.num_iters = num_iters


	def random_walk(self):
		# random walk with every node as start point once
		walks = []
		for node in self.G.nodes():
			walk = [str(node)]
			v = node
			for _ in range(self.length_walk):
				nbs = list(self.G.neighbors(v))
				if len(nbs) == 0:
					break
				v = random.choice(nbs)
				walk.append(str(v))
			walks.append(walk)

		return walks


	def sentenses(self):
		sts = []
		for _ in range(self.num_walks):
			sts.extend(self.random_walk())

		return sts


	def train(self, workers=4, is_loadmodel=False, is_loaddata=False):
		if is_loadmodel:
			print('Load model from file')
			w2v = Word2Vec.load('../models/DeepWalk.model')
			return w2v

		if is_loaddata:
			print('Load data from file')
			with open('../data/tencent_random_walk.txt', 'r') as f:
				sts = f.read()
				sentenses = eval(sts)
		else:
			print('Random walk to get training data...')
			sentenses = self.sentenses()
			print('Number of sentenses to train: ', len(sentenses))
			with open('../data/tencent_random_walk.txt', 'w') as f:
				f.write(str(sentenses))

		print('Start training...')
		random.seed(616)
		w2v = Word2Vec(sentences=sentenses, size=self.emb_size, window=self.window_size, iter=self.num_iters, sg=1, hs=1, min_count=0, workers=workers)
		w2v.save('../models/DeepWalk.model')
		print('Training Done.')
		
		return w2v


