from os.path import join, dirname
import sys
import argparse

import pandas as pd
import numpy as np
import h5py
from scipy import sparse as sp

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  # eat up a lot of memory
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

from ..files import mxm2msd as mxm2msd_fn


def load_data(label_fn, feature_fn, row='tracks', col='tags'):
    """"""
    mxm2msd = dict(
        [line.strip('\n').split(',') for line in open(mxm2msd_fn())]
    )
    msd2mxm = {msd:mxm for mxm, msd in mxm2msd.items()}
    labels, tracks, tags = load_csr_data(label_fn, row, col)
    
    # prepare feature
    feature, track2id, pca, sclr = prepare_feature(feature_fn)
    
    reindex = [track2id[t] for t in tracks]
    X = {key:val[reindex] for key, val in feature.items()}
    y = labels
    y.data[:] = 1
    y = y.toarray()

    whitelist = np.where(y.sum(1) > 0)[0]
    X = {key:val[whitelist] for key, val in X.items()}
    y = y[whitelist]
    
    return X, y


def split_data(X, y, train_ratio=0.7, valid_ratio=0.15):
    """"""
    rnd_idx = np.random.permutation(y.shape[0])
    train_bound = int(len(rnd_idx) * train_ratio)
    valid_bound = train_bound + int(len(rnd_idx) * valid_ratio)
    Xtr, Xvl, Xts = X[:train_bound], X[train_bound:valid_bound], X[valid_bound:]
    ytr, yvl, yts = y[:train_bound], y[train_bound:valid_bound], y[valid_bound:]
    return (Xtr, Xvl, Xts), (ytr, yvl, yts)


def instantiate_model(model_class, model_size): 
    model_size = int(model_size)
    if model_class == 'LogisticRegression':
        model = OneVsRestClassifier(LogisticRegression(solver='lbfgs'))
    elif model_class == 'RandomForestClassifier':
        model = RandomForestClassifier(model_size, n_jobs=-1)
    elif model_class == 'MLPClassifier':
        model = MLPClassifier(
            (model_size,), learning_rate_init=0.001, early_stopping=True,
            learning_rate='adaptive'
        )
    else:
        raise ValueError(
            '[ERROR] only LogisticRegression, RandomForestClassifier, ' +
            'MLPClassifier are supported!'
        )
    return model


def get_model_instance(model_class, train_data, valid_data,
                       n_opt_calls=50, rnd_state=0):
    """"""
    search_spaces = {
        'RandomForestClassifier': [Real(1, 50, "log-uniform", name='model_size')],
        'MLPClassifier': [Real(1, 50, "log-uniform", name='model_size')]
    }
    
    if model_class in search_spaces:
        # setup objective func evaluated by optimizer
        @use_named_args(search_spaces[model_class])
        def _objective(**params):
            # instantiate a model
            model = instantiate_model(model_class, **params) 
            
            # evaluate the model
            scores = eval_model(model, train_data, valid_data)
            
            return -np.mean(scores)

        # search best model with gaussian process based
        res_gp = gp_minimize(
            _objective, search_spaces[model_class],
            n_calls=n_opt_calls, random_state=rnd_state
        )
        print(res_gp.fun) 
        model_size = res_gp.x[0]
        best_model = instantiate_model(model_class, model_size)
    else:
        best_model = instantiate_model(model_class, -1)  # model_size is not used

    return best_model
    
        
def eval_model(model, train_data, valid_data):
    """"""
    # train the model
    model.fit(*train_data)
    
    # test the model
    p = model.predict_proba(valid_data[0])
    if isinstance(model, RandomForestClassifier):
        p = np.concatenate([p_[:, 1][:, None] for p_ in p], axis=1)
        
    return roc_auc_score(valid_data[1], p, 'samples')  # AUC_t


def full_factorial_design(
    text_feats=['personality', 'linguistic', 'topic'],
    models=['LogisticRegression', 'RandomForestClassifier', 'MLPClassifier']):
    
    # 1. models / 2. feature combs
    cases = get_all_comb(text_feats)
    designs = []
    for case in chain.from_iterable(cases):
        feats = dict.fromkeys(text_feats, False)
        for feat in case:
            feats[feat] = True
        
        for model in models:
            row = {'model': model}
            row.update(feats)
            designs.append(row)
    return designs
            
        
def setup_argparse():
    # setup argparser
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('autotagging_fn', type=str,
                        help='filename of the auto-tagging label data')
    parser.add_argument('feature_fn', type=str,
                        help='filename to the lyrics feature file')
    parser.add_argument('--out-fn', type=str, default='test.csv',
                        help='filename for the test output')
    parser.add_argument('--n-rep', type=int, default=5,
                        help='number of testing users')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = setup_argparse()
    
    # load run specific packages
    from itertools import chain
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    
    from ..utils import (load_csr_data,
                         prepare_feature,
                         get_all_comb)
    
    # get full factorial design
    configs = full_factorial_design()
    
    # load the relevant data
    X, y = load_data(args.autotagging_fn, args.feature_fn)
    
    # run
    result = []
    for i, conf in enumerate(configs):
        print(conf)
        print('Running {:d}th / {:d} run...'.format(i+1, len(configs)))
        
        # prepare feature according to the design
        x = np.concatenate([v for k, v in X.items() if conf[k]], axis=1)
        for j in range(args.n_rep):
            # split data
            (Xtr, Xvl, Xts), (ytr, yvl, yts) = split_data(x, y)
            
            # find best model 
            model = get_model_instance(conf['model'], (Xtr, ytr), (Xvl, yvl))
            
            # prep test
            X_ = np.concatenate([Xtr, Xvl], axis=0)
            y_ = np.concatenate([ytr, yvl], axis=0)
            test = eval_model(model, (X_, y_), (Xts, yts))
            
            # register results
            metrics = {'trial': j, 'score': np.mean(test)}
            conf_copy = conf.copy()
            conf_copy.update(metrics)
            result.append(conf_copy)
            
    # save
    pd.DataFrame(result).to_csv(args.out_fn)