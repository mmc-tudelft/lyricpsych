import os
os.environ['LIWC_PATH'] = "data/LIWC.json"

from os.path import join, dirname
import sys
import argparse
sys.path.append(join(dirname(__file__), '..'))

import json
import time
from itertools import chain
from functools import partial
import pickle as pkl

import numpy as np
from scipy import sparse as sp
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import skopt
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from lyricpsych.tasks.fm import *
from lyricpsych.tasks.als_feat import *
from lyricpsych.utils import slice_row_sparse, argpart_sort
from lyricpsych.metrics import ndcg


class BaseAutoTagger(object):
    def fit(self, song_feature, song_tag):
        pass

    def predict(self, song_feature, song_tag):
        pass

    def score(self, song_feature, song_tag):
        pass


class MFAutoTagger(BaseAutoTagger):
    def __init__(self, k, lmbda, l2, alpha, n_iters):
        super().__init__()
        self._mf = ALSFeat(
            k, lmbda=lmbda, l2=l2,
            alpha=alpha, n_iters=n_iters
        )

    def fit(self, song_feature, song_tag):
        self._mf.fit(song_tag.T.tocsr(), song_feature)
        U = self._mf.embeddings_['user']
        self._UU = U.T @ U

    def predict(self, song_feature, song_tag=None):
        W = self._mf.embeddings_['feat']
        U = self._mf.embeddings_['user']

        if song_tag is None:
            song_factor = song_feature @ W
        else:
            n = song_feature.shape[0]
            d = W.shape[-1]
            song_factor = np.empty((n, d), dtype=W.dtype)
            for i in range(song_feature.shape[0]): 
                ind, val = slice_row_sparse(song_tag, i)
                song_factor[i] = partial_ALS_feat(
                    val, ind, U, UU, song_feature[i], W,
                    self._mf.lmbda, self._mf.l2
                )
        return song_factor @ U.T

    def score(self, song_feature, song_tag, seed_song_tag=None,
              metric='auc', average=None, topk=None):
        average = 'samples' if average is None else average
        topk = 10 if topk is None else topk

        if metric == 'auc':
            p = self.predict(song_feature)
            y = song_tag.toarray()
            return safe_roc_auc(y, p, average)

        else:  # ndcg
            p = self.predict(song_feature, seed_song_tag)
            return compute_ndcg(y, p, seed_song_tag, topk)


class MLPAutoTagger(BaseAutoTagger):
    def __init__(self, k, n_hids, l2):
        super().__init__()
        self.k = k
        self.n_hids = n_hids
        self.l2 = l2

        self._mlp = MLPClassifier(
            (k,) * n_hids, alpha=l2, early_stopping=True
        )

    def fit(self, song_feature, song_tag):
        self._mlp.fit(song_feature, song_tag)

    def predict(self, song_feature, song_tag=None):
        return self._mlp.predict(song_feature)

    def score(self, song_feature, song_tag, seed_song_tag=None,
              metric='auc', average='samples'):
        p = self._mlp.predict_proba(song_feature)
        y = song_tag.toarray()
        return safe_roc_auc(y, p, average)
    
    
class FMAutoTagger(BaseAutoTagger):
    def __init__(self, k, lr, l2, n_iters, batch_sz=128, use_gpu=False):
        super().__init__()
        self.k = k
        self.lr = lr
        self.l2 = l2
        self.n_iters = n_iters
        self.user_gpu = use_gpu
        self.batch_sz = batch_sz
        
        self._fm = FactorizationMachine(
            k, init=0.001, n_iters=n_iters, l2=l2, learn_rate=lr,
            loss='bce', loss_agg='sum', use_gpu=use_gpu
        )
    
    def fit(self, song_feature, song_tag):
        # fit FM
        self._fm.fit(song_tag, song_feature,
                     batch_sz=self.batch_sz, n_jobs=2)
        self._fm.cpu()
        self.eval()
        
        # register factors for future
        self.zi, self.zi2 = self._model.zi, self._model.zi2
        self.zu, self.zu2 = self._model.zu, self._model.zu2
            
        # learn the feature -> item factor mapper
        U = self._fm.embeddings['user'].weight.cpu().data.numpy()
        self._mlp_v = MLPRegressor((128,), early_stopping=True) 
        self._mlp_v.fit(song_feature, U[:, :-1])
        
        self._mlp_w = MLPRegressor((128,), early_stopping=True) 
        self._mlp_w.fit(song_feature, U[:, -1])
        
    def predict(self, song_feature, song_tag=None):
        phi = self._fm.embeddings_['feat_user'].weight.data.numpy()
        
#         w = self._mlp_w.predict(song_feature)
#         v = self._mlp_v.predict(song_feature)
        
#         zfu = new_feat @ phi
#         zfu2 = new_feat**2 @ phi**2
        
#         zu = v + zfu[..., :-1]
#         zu2 = v**2 + zfu2[: :-1]
        
#         w = w + zfu[:, -1]
#         v = (zu)


class Optimizer:
    def __init__(self, n_calls=100):
        self.n_calls = n_calls

    def fit(self, model_class, space, song_feat, song_tag,
            valid_song_feat, valid_song_tag, valid_seed_song_tag,
            metric='ndcg', random_state=0, verbose=False):

        @use_named_args(space)
        def objective(**params):
            model = model_class(**params)
            model.fit(song_feat, song_tag)
            s = model.score(
                valid_song_feat, valid_song_tag, valid_seed_song_tag,
                metric=metric
            )
            return -s

        res = gp_minimize(
            objective, space, n_calls=self.n_calls,
            random_state=random_state, verbose=verbose
        )
        return res


DATASETS = {
    'msd50': 'msd_subset_top50',
    'msd1k': 'msd_subset_top1000',
    'mgt50': 'magnatag_subset_top50',
    'mgt188': 'magnatag_subset_top188'
}

SEARCH_SPACE = {
    MFAutoTagger: [
        Integer(5, 20, name='n_iters'),
        Real(1e-7, 1e+7, "log-uniform", name='lmbda'),
        Real(1e-7, 1e+7, "log-uniform", name='l2'),
        Real(1e-7, 1e+7, "log-uniform", name='alpha')
    ],
    MLPAutoTagger: [
        Integer(1, 3, name='n_hids'),
        Real(1e-7, 1, "log-uniform", name='l2')
    ]
}

MODELS = {
    'mf': MFAutoTagger,
    # 'fm': FactorizationMachine,
    'mlp': MLPAutoTagger
}


def safe_roc_auc(y, p, average='samples'):
    if average == 'samples':
        safe_ix = np.where(y.sum(1) > 0)[0]
        p = p[safe_ix]
        y = y[safe_ix]
    elif average == 'macro':
        safe_ix = np.where(y.sum(0) > 0)[0]
        p = p[:, safe_ix]
        y = y[:, safe_ix]
    return roc_auc_score(y, p, average=average)


def compute_ndcg(y, p, seed_mat, topk=10):
    scores = []
    for i in range(y.shape[0]):
        true, _ = slice_row_sparse(y, i)
        if len(true) == 0: continue

        seed, _ = slice_row_sparse(seed_mat, i)
        s = p[i].copy()

        if len(seed) > 0: s[seed] = -np.inf
        pred = argpart_sort(s, topk, ascending=False)
        scores.append(ndcg(true, pred, topk))
    return np.mean(scores)


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


def split_data(test_fold, feat, label, folds, splits,
               valid_fold=None):
    """"""
    if valid_fold is None:
        valid_fold = np.random.choice(
            [i for i in range(len(folds)) if i != test_fold]
        )
    else:
        if valid_fold == test_fold:
            raise ValueError(
                '[ERROR] valid fold should be different to' +
                ' the test fold!'
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


def save_best_param(res, out_root, n_seeds, model, k,
                    dataset, feat_type, split_type):
    out_fn = join(
        out_root,
        '{model}{k:d}_{dset}_seed{seeds}_{feat}_{split}.pkl'.format(
            seeds=n_seeds, model=model, k=k,
            dset=dataset, feat=feat_type, split=split_type
        )
    )
    skopt.dump(res, out_fn, store_objective=False)
    
    
def setup_argparser():
    # setup argparser
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', type=str,
                        help='path where all autotagging data is stored')
    parser.add_argument('out_root', type=str,
                        help='path where output is stored')
    parser.add_argument('model', type=str,
                        choices={'mf', 'mlp'},
                        help='target model to be tested')
    parser.add_argument('dataset', type=str,
                        choices={'msd50', 'msd1k', 'mgt50', 'mgt188'},
                        help='type of dataset')
    parser.add_argument('feature', type=str, choices={'mfcc', 'rand'},
                        help='type of audio feature')
    parser.add_argument('fold', type=int, choices=set(range(10)),
                        help='target fold to be tested')
    parser.add_argument('k', type=int, choices={32, 64, 128},
                        help='model size')
    parser.add_argument('--n-calls', type=int, default=50,
                        help='number of optimize stages')    
    parser.add_argument('--split-type', type=str, default='nomatch',
                        choices={'klmatch', 'nomatch'},
                        help='split method to be used')
    parser.add_argument('--valid-fold', type=int, default=None,
                        choices=set(range(10)),
                        help='target validation fold to be used')
    parser.add_argument('--n-seeds', type=int, default=0,
                        choices={0, 1, 2, 5},
                        help='set the number of seed tags used for the eval.')
    parser.add_argument('--sacle', dest='scale', action='store_true')
    parser.set_defaults(scale=False)
    args = parser.parse_args()
    return args

    
if __name__ == "__main__":
    # parse the input
    args = setup_argparser()
    model = MODELS[args.model]
    metric = 'auc' if args.n_seeds == 0 else 'ndcg'

    # load the dataset and split (using pre-split data)
    data_loc = DATASETS[args.dataset]
    data_path = join(args.data_root, data_loc.split('_')[0], data_loc, 'splits')
    raw_labels, raw_feats, folds, splits = load_data(
        data_path, args.feature, split_type=args.split_type
    )
    feats, seeds, labels = split_data(
        args.fold, raw_feats, raw_labels, folds, splits,
        valid_fold=args.valid_fold
    )

    if args.scale:
        # scale data
        sclr = StandardScaler()
        feats['train'] = sclr.fit_transform(feats['train'])
        for split in ['valid', 'test']:
            for n_seed, feat in feats[split].items():
                feats[split][n_seed] = sclr.transform(feat)

    opt = Optimizer(n_calls=10)
    res = opt.fit(
        partial(model, k=args.k), SEARCH_SPACE[model],
        feats['train'], labels['train'],
        feats['valid'][args.n_seeds], labels['valid'][args.n_seeds],
        seeds['valid'][args.n_seeds], metric=metric, verbose=True
    )
    save_best_param(
        res, args.out_root, args.n_seeds, args.model, args.k,
        args.dataset, args.feature, args.split_type
    )
    
