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
from sklearn.metrics import accuracy_score

from ..files import mxm2msd as mxm2msd_fn
from ..utils import prepare_feature


def load_data(label_fn, feature_fn, row='tracks', col='tags'):
    """"""
    mxm2msd = dict(
        [line.strip('\n').split(',') for line in open(mxm2msd_fn())]
    )
    msd2mxm = {msd:mxm for mxm, msd in mxm2msd.items()}
    with h5py.File(label_fn) as hf:
        genre = hf['genre'][:]
        tracks = hf['tracks'][:]
    
    # prepare feature
    feature, track2id = prepare_feature(feature_fn)
    
    reindex = [track2id[t] for t in tracks]
    X = {key:val[reindex] for key, val in feature.items()}
    
    return X, genre


def eval_clf(model, train_data, valid_data):
    """"""
    # train the model
    model.fit(*train_data)
    
    # test the model
    p = model.predict(valid_data[0])
    return accuracy_score(valid_data[1], p)


def instantiate_clf(model_class, model_size):
    """"""
    model_size = int(model_size)
    if model_class == 'LogisticRegression':
        model = LogisticRegression(max_iter=100,
                                   solver='lbfgs',
                                   multi_class='auto')
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


def setup_argparse():
    # setup argparser
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('classification_fn', type=str,
                        help='filename of the classification label data')
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
    from sklearn.model_selection import StratifiedShuffleSplit
    from .autotagging import (split_data,
                              get_model_instance,
                              full_factorial_design)
    
    # get full factorial design
    configs = full_factorial_design()
    
    # load the relevant data
    X, y = load_data(args.classification_fn, args.feature_fn)
    
    # run
    result = []
    spliter = StratifiedShuffleSplit(train_size=0.8)
    for i, conf in enumerate(configs):
        print(conf)
        print('Running {:d}th / {:d} run...'.format(i+1, len(configs))) 
        
        # prepare feature according to the design
        x = {k:v for k, v in X.items() if conf[k]}
        for j in range(args.n_rep):
            # split data
            (Xtr, Xvl, Xts), (ytr, yvl, yts) = split_data(x, y, spliter)
            
            # find best model 
            model = get_model_instance(
                conf['model'], (Xtr, ytr), (Xvl, yvl),
                model_init_f = instantiate_clf,
                eval_f = eval_clf
            )
            
            # prep test
            X_ = np.concatenate([Xtr, Xvl], axis=0)
            y_ = np.concatenate([ytr, yvl], axis=0)
            test = eval_clf(model, (X_, y_), (Xts, yts))
            
            # register results
            metrics = {'trial': j, 'score': np.mean(test)}
            conf_copy = conf.copy()
            conf_copy.update(metrics)
            result.append(conf_copy)
            
    # save
    pd.DataFrame(result).to_csv(args.out_fn) 