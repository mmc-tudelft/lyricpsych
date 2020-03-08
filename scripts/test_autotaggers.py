import os
os.environ['LIWC_PATH'] = "data/LIWC.json"
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '4'

from os.path import join, dirname
import sys
import argparse
sys.path.append(join(dirname(__file__), '..'))

import sys
sys.path.append('../')

import numpy as np
from scipy import sparse as sp
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score

from lyricpsych.tasks.fm import *
from lyricpsych.tasks.als_feat import *
from lyricpsych.utils import split_recsys_data, slice_row_sparse
from lyricpsych.metrics import ndcg



# load the data
X = sp.load_npz('../data/top1000_msd/labels.npz')
Y = np.load('../data/top1000_msd/mfcc.npy')


# splitting the data
# 1. split by tag
Xtr, Xvl, Xts = split_recsys_data(X)

# 2. TODO? split by song (for scaling)
rnd_idx = np.random.permutation(X.shape[0])
train_bound = int(len(rnd_idx) * 0.8)
valid_bound = train_bound + int(len(rnd_idx) * 0.1)

Ytr = Y[rnd_idx[:train_bound]]
Yvl = Y[rnd_idx[train_bound:valid_bound]]
Yts = Y[rnd_idx[valid_bound:]]

Xtrtr = X[rnd_idx[:train_bound]]
Xvlsd = Xtr[rnd_idx[train_bound:valid_bound]]
Xvlvl = Xvl[rnd_idx[train_bound:valid_bound]]
Xtssd = Xtr[rnd_idx[valid_bound:]]
Xtsts = Xts[rnd_idx[valid_bound:]]


train = sp.vstack([Xtrtr, Xvlsd])
feat = np.concatenate([Ytr, Yvl], axis=0)

als = ALSFeat(64, lmbda=1, l2=1e-6, alpha=10, dropout=0, dtype='float64')
als.fit(train.T.tocsr(), feat, verbose=True)


k = 10
song_factors = als.embeddings_['item']
tag_factors = als.embeddings_['user']

n_train = Ytr.shape[0]
n_trvl = n_train + Yvl.shape[0]
scores = []
for j, i in enumerate(range(n_train, n_trvl)):
    true, _ = slice_row_sparse(Xvlvl, j)
    if len(true) == 0:
        continue
    tr, _ = slice_row_sparse(train, i)
    s = song_factors[i] @ tag_factors.T
    s[tr] = -np.inf
    pred = argpart_sort(s, k, ascending=False)
    scores.append(ndcg(true, pred, k))  