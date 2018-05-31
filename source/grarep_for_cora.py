import networkx as nx 
import numpy as np 
import random
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import argparse

from data_utils_cora import *
from grarep import GraRep

parser = argparse.ArgumentParser(description='Train and test LINE on tencent dataset')
parser.add_argument('--embsize', default=50, type=int)
parser.add_argument('--K', default=6, type=int)
parser.add_argument('--C', default=2, type=float)
args = parser.parse_args()

random.seed(616)

X_, A, y = load_data(path='../cora/')
_, _, _, idx_train, idx_val, idx_test = get_splits(y)

grarep = GraRep(A, emb_size_per_K=args.embsize, K=args.K)
grarep.train(verbose=True)


pca = PCA(n_components=args.embsize * args.K)
X_pca = pca.fit_transform(X_)

X = np.hstack([normalize(X_pca), normalize(grarep.emb)])
X_trn, X_val, X_tst = X[idx_train], X[idx_val], X[idx_test]
y_trn, y_val, y_tst = y[idx_train], y[idx_val], y[idx_test]
y_trn = np.argmax(y_trn, axis=1)
y_val = np.argmax(y_val, axis=1)
y_tst = np.argmax(y_tst, axis=1)

print('Testing...')
clf = LogisticRegression(C=args.C)
clf.fit(X_trn, y_trn)
score = clf.score(X_val, y_val)
print('Validation accuracy: ', score)
score = clf.score(X_tst, y_tst)
print('Test accuracy: ', score)

print('Cora, test accuracy:', score)



