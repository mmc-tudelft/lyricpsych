from os.path import join, dirname
import sys
import argparse
from itertools import chain

import pandas as pd
import numpy as np
import h5py
from scipy import sparse as sp

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from ..files import mxm2msd as mxm2msd_fn
from ..utils import (get_all_comb,
                     preproc_feat,
                     load_csr_data,
                     prepare_feature,
                     split_data)


def load_data(label_fn, feature_fn, row='tracks', col='tags'):
    """"""
    mxm2msd = dict(
        [line.strip('\n').split(',') for line in open(mxm2msd_fn())]
    )
    msd2mxm = {msd:mxm for mxm, msd in mxm2msd.items()}
    labels, tracks, tags = load_csr_data(label_fn, row, col)

    # prepare feature
    feature, track2id = prepare_feature(feature_fn)

    reindex = [track2id[t] for t in tracks]
    X = {key:val[reindex] for key, val in feature.items()}
    y = labels
    y.data[:] = 1
    y = y.toarray()

    whitelist = np.where(y.sum(1) > 0)[0]
    X = {key:val[whitelist] for key, val in X.items()}
    y = y[whitelist]

    return X, y


def instantiate_model(model_class, model_size):
    model_size = int(model_size)
    if model_class == 'LogisticRegression':
        model = OneVsRestClassifier(
            LogisticRegression(solver='lbfgs', max_iter=300)
        )
    elif model_class == 'NaiveBayes':
        model = OneVsRestClassifier(GaussianNB(), n_jobs=-1)
    elif model_class == 'MLPClassifier':
        model = MLPClassifier(
            (model_size,), learning_rate_init=0.001, early_stopping=True,
            learning_rate='adaptive'
        )
    else:
        raise ValueError(
            '[ERROR] only LogisticRegression, NaiveBayes, ' +
            'MLPClassifier are supported!'
        )
    return model


def eval_model(model, train_data, valid_data):
    """"""
    # train the model
    model.fit(*train_data)

    # test the model
    p = model.predict_proba(valid_data[0])
    return roc_auc_score(valid_data[1], p, 'samples')  # AUC_t


def get_model_instance(model_class, train_data, valid_data,
                       n_opt_calls=50, rnd_state=0,
                       model_init_f=instantiate_model,
                       eval_f=eval_model):
    """"""
    search_spaces = {
        'MLPClassifier': [Real(50, 100, "log-uniform", name='model_size')]
    }

    if model_class in search_spaces:
        # setup objective func evaluated by optimizer
        @use_named_args(search_spaces[model_class])
        def _objective(**params):
            # instantiate a model
            model = model_init_f(model_class, **params)

            # evaluate the model
            scores = eval_f(model, train_data, valid_data)

            return -np.mean(scores)

        # search best model with gaussian process based
        res_gp = gp_minimize(
            _objective, search_spaces[model_class],
            n_calls=n_opt_calls, random_state=rnd_state,
            verbose=True
        )
        print(res_gp.fun, res_gp.x)
        model_size = res_gp.x[0]
        best_model = model_init_f(model_class, model_size)
    else:
        best_model = model_init_f(model_class, -1)  # model_size is not used

    return best_model


def full_factorial_design(
    text_feats=['audio', 'liwc', 'value', 'personality', 'linguistic', 'topic'],
    models=['LogisticRegression', 'NaiveBayes', 'MLPClassifier']):

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
    from sklearn.model_selection import ShuffleSplit

    # get full factorial design
    configs = full_factorial_design()

    # load the relevant data
    X, y = load_data(args.autotagging_fn, args.feature_fn)

    # run
    result = []
    spliter = ShuffleSplit(train_size=0.8)
    for i, conf in enumerate(configs):
        print(conf)
        print('Running {:d}th / {:d} run...'.format(i+1, len(configs)))

        # prepare feature according to the design
        x = {k:v for k, v in X.items() if conf[k]}
        for j in range(args.n_rep):
            # split data
            (Xtr, Xvl, Xts), (ytr, yvl, yts) = split_data(x, y, spliter)

            # find best model 
            model = get_model_instance(conf['model'], (Xtr, ytr), (Xvl, yvl))

            # prep test
            X_ = np.concatenate([Xtr, Xvl], axis=0)
            y_ = np.concatenate([ytr, yvl], axis=0)
            test = eval_model(model, (X_, y_), (Xts, yts))
            print(test)

            # register results
            metrics = {'trial': j, 'score': np.mean(test)}
            conf_copy = conf.copy()
            conf_copy.update(metrics)
            result.append(conf_copy)

    # save
    pd.DataFrame(result).to_csv(args.out_fn)
