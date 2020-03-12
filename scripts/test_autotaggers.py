import os
os.environ['LIWC_PATH'] = "data/LIWC.json"
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '6'

from os.path import join, dirname
import sys
import argparse
sys.path.append(join(dirname(__file__), '..'))

import sys
sys.path.append('../')

from itertools import chain

import numpy as np
from scipy import sparse as sp
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from lyricpsych.tasks.fm import *
from lyricpsych.tasks.als_feat import *
from lyricpsych.utils import split_recsys_data, slice_row_sparse, argpart_sort
from lyricpsych.metrics import ndcg


def load_data(data_root, feat_type='mfcc'):
    label_fn = join(data_root, 'X.npz')
    feat_fn = join(data_root, 'Xfeat_{}.npy'.format(feat_type))
    folds = [
        np.load(join(data_root, 'fold_idx_{:d}.npy'.format(i)))
        for i in range(10)
    ]
    seeds = np.load(join(data_root, 'seeds.npy'))

    # load the data
    X = sp.load_npz(label_fn)
    Y = np.load(feat_fn)
    return X, Y, folds, seeds


def build_sparse_seed_mat(seeds, n_songs, n_tags):
    cols_ = [list(s[s!=-1]) for s in seeds]
    rows_ = [[j] * len(cols_[j]) for j in range(len(cols_))]
    cols = list(chain.from_iterable(cols_))
    rows = list(chain.from_iterable(rows_))
    seeds_mat = sp.coo_matrix(
        ([1]*len(cols), (rows, cols)),
        shape=(n_songs, n_tags)
    )
    return seeds_mat


def split_data(test_fold, feat, label, folds, seeds, n_seeds=0, scale=False):
    """"""
    valid_fold = np.random.choice(
        [i for i in range(len(folds)) if i != test_fold]
    )
    train_folds = [
        i for i in range(len(folds))
        if i != test_fold and i != valid_fold
    ]
    train_idx = np.concatenate([folds[i] for i in train_folds])
    valid_idx = folds[valid_fold]
    test_idx = folds[test_fold]
    eval_idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
    label_cont = {}
    feat_cont = {}
    seed_cont = {}

    if n_seeds > 0:
        # build seed mat
        seed_mat = build_sparse_seed_mat(seeds[:, :n_seeds], *label.shape)
        seed_mat = seed_mat.tocsr()
        label = label.tocsr()
        for split, idx in eval_idx.items():
            if split == 'train':
                label_cont[split] = label[idx]
                feat_cont[split] = feat[idx]
            else:
                label_cont[split] = label[idx] - seed_mat[idx]
                feat_cont[split] = feat[idx]
                seed_cont[split] = seed_mat[idx]
    else:
        # this is the auto-tagging case
        for split, idx in eval_idx.items():
            label_cont[split] = label[idx]
            feat_cont[split] = feat[idx]

    return feat_cont, seed_cont, label_cont


def predict(model, data, indices, feat, UU=None):
    """"""
    if isinstance(model, ALSFeat):
        W = model.embeddings_['feat']
        U = model.embeddings_['user']

        if indices is None or len(indices) == 0:
            # just infer with the feature
            song_factor = feat @ W
        else:
            # get the partial learn
            if UU is None:
                UU = U.T @ U

            song_factor = partial_ALS_feat(
                data, indices, U, UU, feat, W, model.lmbda, model.l2
            )
        return song_factor @ U.T
    else:
        # FM
        return None


def evaluate(model, feats, seeds, labels, n_seeds=0, topk={1, 5, 10, 20}):
    """"""
    auc_res = None
    if isinstance(model, ALSFeat):
        U = model.embeddings_['user']
        UU = U.T @ U

    if n_seeds == 0:
        n_tests, n_tags = labels.shape
        p = feats @ model.embeddings_['feat']
        p = p @ model.embeddings_['user'].T
        y = labels.toarray()
        auc_res = {
            avg: roc_auc_score(y, p, average=avg)
            for avg in ['micro', 'macro', 'samples']
        }

    scores = {k:[] for k in topk}
    for i in range(labels.shape[0]):
        true, _ = slice_row_sparse(labels, i)
        if len(true) == 0:
            continue
        if seeds is not None:
            seed_ind, seed_val = slice_row_sparse(seeds, i)
        else:
            seed_ind, seed_val = np.array([]), np.array([])

        s = predict(model, seed_val, seed_ind, feats[i], UU=UU)
        if len(seed_ind) > 0:
            s[seed_ind] = -np.inf
        # true = np.append(true, seed_ind)
        pred = argpart_sort(s, max(topk), ascending=False)

        for k in topk:
            scores[k].append(ndcg(true, pred, k))

    ndcg_res = {k:np.mean(score) for k, score in scores.items()}

    return auc_res, ndcg_res


def find_best(model, k, space, feats, seeds, labels):
    """"""

    if model == 'ALSFeat':
        @use_named_args(space)
        def objective(**params):
            als = ALSFeat(k, dtype='float64', dropout=0, **params)
            als.fit(
                labels['train'].T.tocsr(),
                feats['train'],
                verbose=True
            )
            if seeds is not None: seed = seeds['valid']
            else:                 seed = None
            results = evaluate(
                als, feats['valid'], seed, labels['valid'], n_seeds
            )
            return -results[1][10]

        res = gp_minimize(objective, space, n_calls=50,
                          random_state=0, verbose=True)
        return res

    elif model == 'FactorizationMachine':
        return None


if __name__ == "__main__":
    data_root = '/Users/jaykim/Downloads/datasets/msd/msd_subset_top1000/splits/'
    eval_target = 'test'
    fold = 0
    n_seeds = 1
    k = 32
    lmbda = 1e+6
    l2 = 1e-6
    alpha = 1
    scale = False

    labels, feats, folds, seeds = load_data(data_root)
    feats, seeds, labels = split_data(fold, feats, labels, folds, seeds, n_seeds)

    if scale:
        # scale data
        sclr = StandardScaler()
        feats['train'] = sclr.fit_transform(feats['train'])
        feats['valid'] = sclr.transform(feats['valid'])
        feats['test'] = sclr.transform(feats['test'])

    als = ALSFeat(32, alpha=5, l2=1e-7, lmbda=1e+10, n_iters=15, dropout=0)
    als.fit(labels['train'].T.tocsr(), feats['train'], verbose=True)
    if n_seeds == 0:
        results = evaluate(als, feats['valid'], None, labels['valid'], n_seeds)
    else:
        results = evaluate(als, feats['valid'], seeds['valid'], labels['valid'], n_seeds)
    print(results)

    # The list of hyper-parameters we want to optimize. For each one we define the
    # bounds, the corresponding scikit-learn parameter name, as well as how to
    # sample values from that dimension (`'log-uniform'` for the learning rate)
    # space  = [Integer(10, 30, name='n_iters'),
    #           Real(1e+5, 1e+10, "log-uniform", name='lmbda'),
    #           Real(1e-4, 1e+4, "log-uniform", name='l2'),
    #           Real(1e-10, 1e+2, "log-uniform", name='alpha')]
    # res = find_best('ALSFeat', k, space, feats, labels)
    # best_model = ALSFeat(
    #     k, alpha=res.x[3], l2=res.x[2], lmbda=res.x[1], n_iters=res.x[0],
    #     dtype='float64', dropout=0
    # )

    # total_train_label = sp.vstack(
    #     [labels['train'], labels['valid'] + seeds['valid']]
    # )
    # total_train_feat = np.vstack([feats['train'], feats['valid']])

    # best_model.fit(
    #     total_train_label.T.tocsr(),
    #     total_train_feat,
    #     verbose=True
    # )
    # results = evaluate(
    #     best_model, feats['test'], seeds['test'], labels['test'], n_seeds
    # )

    # print(results)
