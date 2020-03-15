import os
os.environ['LIWC_PATH'] = "data/LIWC.json"
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['NUMBA_NUM_THREADS'] = '2'

from os.path import join, dirname
import sys
import argparse
sys.path.append(join(dirname(__file__), '..'))

import json
import time
from itertools import chain
import pickle as pkl

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
from lyricpsych.utils import slice_row_sparse, argpart_sort
from lyricpsych.metrics import ndcg


SEARCH_SPACE = {
    ALSFeat: {
        'noseed':[
            Integer(5, 20, name='n_iters'),
            Real(1e+5, 1e+10, "log-uniform", name='lmbda'),
            Real(1e-7, 1, "log-uniform", name='l2'),
            Real(1e-2, 1e+2, "log-uniform", name='alpha')
        ],
        'seed':[
            Integer(5, 20, name='n_iters'),
            Real(1, 10, "log-uniform", name='lmbda'),
            Real(1e-7, 1, "log-uniform", name='l2'),
            Real(1e-2, 1e+2, "log-uniform", name='alpha')
        ]
    }
}
DATASETS = {
    'msd50': 'msd_subset_top50',
    'msd1k': 'msd_subset_top1000',
    'mgt50': 'magnatag_subset_top50',
    'mgt188': 'magnatag_subset_top188'
}


def load_data(data_root, feat_type='mfcc', split_type='klmatch'):
    label_fn = join(data_root, 'X.npz')
    feat_fn = join(data_root, 'Xfeat_{}.npy'.format(feat_type))
    folds = [
        np.load(join(data_root, 'fold_idx_{:d}.npy'.format(i)))
        for i in range(10)
    ]
    split_fn_tmp = 'fold{:d}_split_{}.pkl'
    splits = [
        pkl.load(
            open(
                join(
                    data_root,
                    split_fn_tmp.format(i, split_type)
                ), 'rb'
            )
        )
        for i in range(10)
    ]

    # load the data
    X = sp.load_npz(label_fn)
    Y = np.load(feat_fn)
    return X, Y, folds, splits


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


def split_data(test_fold, feat, label, folds, splits, scale=False):
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

    for split, idx in eval_idx.items():
        if split == 'train':
            label_cont[split] = label[idx]
            feat_cont[split] = feat[idx]
        else:
            target_fold = valid_fold if split == 'valid' else test_fold
            feat_cont[split] = {}
            seed_cont[split] = {}
            label_cont[split] = {}
            for n_seed, case in splits[target_fold].items():
                label_cont[split][n_seed] = case['gt']
                seed_cont[split][n_seed] = case['seed']
                feat_cont[split][n_seed] = feat[case['idx']]

    # prepare the "final training dataset"
    total_train_idx = np.r_[train_idx, folds[valid_fold]]
    label_cont['total_train'] = label[total_train_idx]
    feat_cont['total_train'] = feat[total_train_idx]

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


def evaluate(model, feats, seeds, labels, topk={1, 5, 10, 20}):
    """"""
    auc_res = None
    if isinstance(model, ALSFeat):
        U = model.embeddings_['user']
        UU = U.T @ U

    ndcg_res = {}
    for n_seeds, gt in labels.items():

        if n_seeds == 0:
            n_tests, n_tags = gt.shape
            p = feats[n_seeds] @ model.embeddings_['feat']
            p = p @ model.embeddings_['user'].T
            y = gt.toarray()

            auc_res = {}
            for avg in ['micro', 'macro', 'samples']:
                if avg == 'samples':
                    safe_ix = np.where(y.sum(1) > 0)[0]
                    p = p[safe_ix]
                    y = y[safe_ix]
                elif avg == 'macro':
                    safe_ix = np.where(y.sum(0) > 0)[0]
                    p = p[:, safe_ix]
                    y = y[:, safe_ix]
                auc_res[avg] = roc_auc_score(y, p, average=avg)

        scores = {k:[] for k in topk}
        for i in range(gt.shape[0]):
            true, _ = slice_row_sparse(gt, i)
            if len(true) == 0:
                continue
            seed_ind, seed_val = slice_row_sparse(seeds[n_seeds], i)

            s = predict(model, seed_val, seed_ind, feats[n_seeds][i], UU=UU)
            if len(seed_ind) > 0:
                s[seed_ind] = -np.inf
            # true = np.append(true, seed_ind)
            pred = argpart_sort(s, max(topk), ascending=False)

            for k in topk:
                scores[k].append(ndcg(true, pred, k))

        ndcg_res[n_seeds] = {k:np.mean(score) for k, score in scores.items()}
    return auc_res, ndcg_res


def find_best(model, k, feats, seeds, labels, coldstart=False,
              n_calls=50, random_state=0, verbose=True):
    """"""
    isseed = 'noseed' if coldstart else 'seed'
    space = SEARCH_SPACE[model][isseed]

    if model == ALSFeat:

        @use_named_args(space)
        def objective(**params):
            als = model(k, dtype='float64', dropout=0, **params)
            als.fit(
                labels['train'].T.tocsr(),
                feats['train'],
                verbose=True
            )
            auc_, ndcg_ = evaluate(
                als, feats['valid'], seeds['valid'], labels['valid']
            )
            if coldstart:
                trg_meas = auc_['macro']
            else:
                trg_meas = np.mean([v[10] for k, v in ndcg_.items() if k > 0])

            return -trg_meas

        res = gp_minimize(objective, space, n_calls=n_calls,
                          random_state=random_state,
                          verbose=verbose)
        return res

    elif model == 'FactorizationMachine':
        return None


def save_result(dataset, feat_type, fold, k, scale,
                best_model, best_param, auc_, ndcg_,
                out_root):
    """"""
    result = {
        'dataset': dataset,
        'feat_type': feat_type,
        'fold': fold, 'k': k,
        'scale': scale, 'model': str(best_model),
        'auc': auc_, 'ndcg': ndcg_,
        'best_param': {
            k:int(v) if k=='n_iters' else v
            for k, v in best_param.items()
        }
    }
    fn = join(
        out_root,
        '{dataset}_fold{fold:d}_{model}.json'.format(**result)
    )
    json.dump(result, open(fn, 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', type=str,
                        help='path where all autotagging data is stored')
    parser.add_argument('out_root', type=str,
                        help='path where output is stored')
    parser.add_argument('dataset', type=str,
                        choices={'msd50', 'msd1k', 'mgt50', 'mgt188'},
                        help='type of dataset')
    parser.add_argument('feature', type=str, choices={'mfcc', 'rand'},
                        help='type of audio feature')
    parser.add_argument('fold', type=int, choices=set(range(10)),
                        help='target fold to be tested')
    parser.add_argument('k', type=int, choices={32, 64, 128},
                        help='model size')
    parser.add_argument('--sacle', dest='scale', action='store_true')
    parser.add_argument('--no-scale', dest='scale', action='store_false')
    parser.set_defaults(scale=True)
    parser.add_argument('--seed', dest='seed', action='store_true')
    parser.add_argument('--no-seed', dest='seed', action='store_false')
    parser.set_defaults(seed=True)
    args = parser.parse_args()

    # parse the input
    data_root = args.data_root
    out_root = args.out_root
    dataset = args.dataset
    feat_type = args.feature
    fold = args.fold
    k = args.k
    scale = args.scale
    seed = args.seed
    model = ALSFeat  # {ALSFeat, FactorizationMachine, MLPClassifier}

    # load the dataset and split (using pre-split data)
    data_loc = DATASETS[dataset]
    data_path = join(data_root, data_loc.split('_')[0], data_loc, 'splits')
    raw_labels, raw_feats, folds, splits = load_data(data_path, feat_type)
    feats, seeds, labels = split_data(fold, raw_feats, raw_labels, folds, splits)

    if scale:
        # scale data
        sclr = StandardScaler()
        feats['train'] = sclr.fit_transform(feats['train'])
        for split in ['valid', 'test']:
            for n_seed, feat in feats[split].items():
                feats[split][n_seed] = sclr.transform(feat)

    # find the best model using hyper-parameter tuner (GP)
    res = find_best(model, k, feats, seeds, labels, seed)
    best_param = {
        SEARCH_SPACE[model][i].name: res.x[i]
        for i in range(len(res.x))
    }
    best_model = model(k, dtype='float64', dropout=0, **best_param)
    best_model.fit(
        labels['total_train'].T.tocsr(),
        feats['total_train'],
        verbose=True
    )

    # final evaluation step
    auc_, ndcg_ = evaluate(best_model, feats['test'], seeds['test'], labels['test'])

    # save the result to the disk
    save_result(
        dataset, feat_type, fold, k, scale,
        best_model, best_param, auc_, ndcg_,
        out_root
    )
