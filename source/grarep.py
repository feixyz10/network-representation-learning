import networkx as nx 
import numpy as np 
import random
from numpy.linalg import norm
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
import scipy.sparse as sp

class GraRep:
    def __init__(self, adj_mat, emb_size_per_K=32, K=4, lamb=1):
        self.A = normalize(adj_mat, norm='l1', axis=1)
        self.N = self.A.shape[0]
        self.emb_size = emb_size_per_K * K 
        self.K = K
        self.lamb = lamb
        assert self.lamb > 0, 'lambda should be greater than zeor'
        self.emb = np.zeros([self.N, self.emb_size])


    def compute_X_matrix(self, Ak):
        Ak_norm = normalize(Ak, norm='l1', axis=0)
        Ak_norm.data = np.log(Ak_norm.data)
        Ak_norm.data -= np.log(1./self.N)
        
        X = Ak_norm.multiply(Ak_norm > 0)
        return X


    def train(self, verbose=True):
        if verbose: print('Start training...')
        d = self.emb_size // self.K
        A = self.A
        Ai_1 = sp.identity(A.shape[0])
        for i in range(self.K):
            if verbose: print('Training step %d' % (i+1))
            Ai = Ai_1 * A
            X = self.compute_X_matrix(Ai)
            U, S, V = svds(Ai, k=d)
            W = U * S**0.5
            self.emb[:, i*d:d+i*d] = W

            Ai_1 = Ai

        if verbose: print('Training done')

    def similarity(self, u, v):
        v1 = self.emb[u]
        v2 = self.emb[v]
        
        return v1.dot(v2)/(norm(v1)*norm(v2))




