import networkx as nx 
import numpy as np
import random
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity

class NegativeSampler:
    def __init__(self, lengths, factor_M=10):
        self.lengths = np.cumsum(lengths)
        self.M = len(lengths) * factor_M
        self.table = self.creat_table()

    def fetch(self, u, v):
        while True:
            m = random.randint(0, self.M-1)
            node = self.table[m]
            if node != u and node != v:
                break

        return node

    def creat_table(self):
        table = []
        idx = 0
        ms = np.linspace(0, 1, self.M)
        for m in ms:
            if m >= self.lengths[idx]:
                idx += 1
            table.append(idx)

        return table


class TencentDataset(Dataset):
    def __init__(self, G, neg_sample_size=5):
        self.G = G
        self.edges = list(self.G.edges())
        self.neg_size = neg_sample_size
        if self.neg_size > 0:
            self.order = 2
            lengths = np.asarray([G.degree(node)**0.75 for node in G.nodes()])
            lengths = lengths / np.sum(lengths)
            self.neg_smapler = NegativeSampler(lengths)
        else:
            self.order = 1

    def __len__(self):
        return self.G.number_of_edges()

    def __getitem__(self, idx):
        u, v = self.edges[idx]
        if self.order == 1:
            return t.LongTensor([u]), t.LongTensor([v]), t.FloatTensor([1.0])

        us, vs, ws = [u], [v], [1.0]
        for _ in range(self.neg_size):
            v = self.neg_smapler.fetch(u, v)
            us.append(u)
            vs.append(v)
            ws.append(-1.0)
        return t.LongTensor(us), t.LongTensor(vs), t.FloatTensor(ws)

