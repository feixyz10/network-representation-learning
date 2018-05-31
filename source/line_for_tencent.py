import networkx as nx 
import numpy as np 
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch as t
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse

from line import Line_1st, Line_2nd, Line
from utils import TencentDataset

parser = argparse.ArgumentParser(description='Train and test LINE on tencent dataset')
parser.add_argument('--load', default=0, type=int)
parser.add_argument('--train', default=0, type=int)
parser.add_argument('--gpuid', default='4', type=str)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batchsize', default=256, type=int)
args = parser.parse_args()

device = t.device('cuda:{}'.format(args.gpuid) if t.cuda.is_available() else "cpu")
NEG_SAMPLE_SIZE = 5

def train(model, data_loader, order, optimizer, num_epochs, x_val, y_val):
    data_size = len(data_loader.dataset)
    neg_size = 0 if order == 1 else NEG_SAMPLE_SIZE

    best_auc = 0.0
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_num = 0
        with tqdm(total=data_size) as pbar:
            for i, data in enumerate(data_loader):
                x1, x2, w = data
                x1 = x1.view(-1).to(device)
                x2 = x2.view(-1).to(device)
                w = w.view(-1).to(device)
                loss = model(x1, x2, w)
                loss *= (neg_size + 1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x1.shape[0]
                running_num += x1.shape[0]

                update_size = x1.shape[0] // (neg_size + 1)
                pbar.update(update_size)
                pbar.set_description('Epoch %d/%d Train loss: %.4f' % (epoch+1, num_epochs, running_loss/running_num))

        y_score = []
        for u, v in x_val:
            y_score.append(model.similarity(u, v))

        auc_val = roc_auc_score(y_val, y_score)
        print('Validation AUC: ', auc_val)

        if best_auc < auc_val:
            best_auc = auc_val
            t.save(model.state_dict(), '../models/line_{}st_best.pth'.format(order))


random.seed(616)
        
line_1st = Line_1st(169209, emb_size=128).to(device)
line_2st = Line_2nd(169209, emb_size=128).to(device)

if args.load != 0:
    print('Load pretrained models from file')
    line_1st.load_state_dict(t.load('../models/line_1st_best.pth'))
    line_2st.load_state_dict(t.load('../models/line_2st_best.pth'))
if args.train != 0:
    print('Load data and construct graph')
    edges = np.load('../tencent/train_edges.npy')
    G = nx.Graph()
    for i in range(169209):
        G.add_node(i)
    G.add_edges_from(edges)

    pos_val = np.load('../tencent/test_edges.npy')
    neg_val = np.load('../tencent/test_edges_false.npy')
    y_val = [True]*pos_val.shape[0] + [False]*neg_val.shape[0]
    x_val = np.vstack([pos_val, neg_val])

    print('Train Line first order similarity')
    data_set_1 = TencentDataset(G, 0)
    data_loaer_1 = DataLoader(data_set_1, shuffle=True, batch_size=args.batchsize, num_workers=4)
    optimizer = optim.Adam(line_1st.parameters(), lr=1e-3)
    train(line_1st, data_loaer_1, 1, optimizer, num_epochs=args.epochs, x_val=x_val, y_val=y_val)

    print('Train Line second order similarity')
    data_set_2 = TencentDataset(G, 5)
    data_loaer_2 = DataLoader(data_set_2, shuffle=True, batch_size=args.batchsize//6, num_workers=6)
    optimizer = optim.Adam(line_2st.parameters(), lr=1e-3)
    train(line_2st, data_loaer_2, 2, optimizer, num_epochs=args.epochs//2, x_val=x_val, y_val=y_val)


line = Line(line_1st, line_2st, alpha=2)

pos_test = np.load('../tencent/test_edges.npy')
neg_test = np.load('../tencent/test_edges_false.npy')
y_true = [True]*pos_test.shape[0] + [False]*neg_test.shape[0]
X = np.vstack([pos_test, neg_test])

print('Testing...')
y_score = []
for u, v in X:
    y_score.append(line.similarity(u, v))

auc_test = roc_auc_score(y_true, y_score)
print('Tencent, test AUC:', auc_test)