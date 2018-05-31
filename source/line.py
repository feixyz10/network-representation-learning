import networkx as nx 
import numpy as np
import random
import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from numpy.linalg import norm


class Line_1st(nn.Module):
	def __init__(self, num_nodes, emb_size=64):
		super(Line_1st, self).__init__()
		self.order = 1
		self.emb_size = emb_size
		self.num_nodes = num_nodes
		self.emb = nn.Embedding(num_nodes, emb_size)

	def forward(self, x1, x2, w):
		x1 = self.emb(x1)
		x2 = self.emb(x2)
		x = w * t.sum(x1*x2, dim=1)
		return -F.logsigmoid(x).mean()

	def similarity(self, u, v):
		v1 = self.emb.weight[u].detach().cpu().numpy()
		v2 = self.emb.weight[v].detach().cpu().numpy()
		
		return v1.dot(v2)/(norm(v1)*norm(v2))


class Line_2nd(nn.Module):
	def __init__(self, num_nodes, emb_size=64):
		super(Line_2nd, self).__init__()
		self.order = 2
		self.emb_size = emb_size
		self.num_nodes = num_nodes
		self.emb = nn.Embedding(num_nodes, emb_size)
		self.ctx = nn.Embedding(num_nodes, emb_size) # context vector

	def forward(self, x1, x2, w):
		x1 = self.emb(x1)
		x2 = self.ctx(x2)
		x = w * t.sum(x1*x2, dim=1)
		return -F.logsigmoid(x).mean()

	def similarity(self, u, v):
		v1 = self.emb.weight[u].detach().cpu().numpy()
		v2 = self.emb.weight[v].detach().cpu().numpy()
		
		return v1.dot(v2)/(norm(v1)*norm(v2))


class Line:
	def __init__(self, line_1st, line_2nd, alpha=2):
		self.alpha = alpha
		emb1 = line_1st.to(t.device('cpu')).emb.weight.detach().numpy()
		emb2 = line_2nd.to(t.device('cpu')).emb.weight.detach().numpy() * self.alpha
		self.embedding = np.hstack([emb1, emb2])

	def similarity(self, u, v):
		v1 = self.embedding[u]
		v2 = self.embedding[v]
		
		return v1.dot(v2)/(norm(v1)*norm(v2))




