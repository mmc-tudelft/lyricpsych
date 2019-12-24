import numpy as np
from implicit.als import AlternatingLeastSquares


class Recsys:
    def _recommend(self, user, user_item, n):
        """ Internal function that actually serves the rec
        """
        raise NotImplementedError('[ERROR] this should be implemented!')
        
    def recommend(self, user, user_item, n=100):
        """ Helper function that does the pre/post processing
        """
        slc = slice(user_item.indptr[user], user_item.indptr[user+1])
        gt = user_item.indices[slc]
        item = self._recommend(user, user_item, n, gt) 
        return item


class UserKNN(Recsys):
    def __init__(self, k, user_profiles, metric='cosine'):
        """
        """
        super().__init__()
        self.k = k
        self.user_profiles = user_profiles
        self.user_profiles /= np.linalg.norm(
            self.user_profiles, axis=1
        )[:, None]
        self.metric = metric
        
    def _recommend(self, user, user_item, n, gt=None):
        """
        """
        d = self.user_profiles @ self.user_profiles[user]
        idx = np.argpartition(d, self.k+1)[:self.k+1]
        neighbors = idx[np.argsort(d[idx])][1:]
        
        s = np.array(user_item[neighbors].sum(0)).ravel()
        s = s.astype(np.float32)
        if gt is not None:
            s[gt] = -np.inf
        idx = np.argpartition(-s, n)[:n]
        recs = idx[np.argsort(-s[idx])]
        return recs
    
    
class ItemKNN(Recsys):
    def __init__(self, k, item_profiles, metric='cosine'):
        """
        """
        super().__init__()
        self.k = k
        self.item_profiles = item_profiles
        self.item_profiles /= np.linalg.norm(
            self.item_profiles, axis=1
        )[:, None]
        self.metric = metric
        
    def _recommend(self, user, user_item, n, gt=None):
        """
        """
        items = user_item[user].indices
        d = self.item_profiles @ self.item_profiles[items].T
        d = d.mean(1).astype(np.float32)
        if gt is not None:
            d[gt] = np.inf
        idx = np.argpartition(d, n)[:n]
        recs = idx[np.argsort(d[idx])]
        return recs

    
class WRMF(Recsys):
    def __init__(self, k, reg=1e-4, n_iters=15):
        """"""
        super().__init__()
        self.k = k
        self.reg = reg
        self.n_iters = n_iters
        self.als = AlternatingLeastSquares(
            k, regularization=reg, iterations=n_iters
        )
    
    def _recommend(self, user, user_item, n, gt=None):
        return np.array([
            itemid for itemid, score 
            in self.als.recommend(user, user_item, n)
        ])

    
class Random(Recsys):
    def __init__(self):
        super().__init__()
        
    def _recommend(self, user, user_item, n, gt):
        item = np.random.permutation(user_item.shape[1])
        return item[~np.in1d(item, gt)][:n]
    
    
class MostPopular(Recsys):
    def __init__(self, user_item):
        super().__init__()
        self.user_item = user_item
        self.mp = np.argsort(-user_item.sum(0).A.flatten())
        
    def _recommend(self, user, user_item, n, gt):
        return self.mp[~np.in1d(self.mp, gt)][:n]

    
def build_user_profile(user_item, feature, track2id, items,
                       mixture=None, alpha=100):
    """ Build user profile by using feature set
    
    Inputs:
        user_item (scipy.sparse.csr_matrix): user-item matrix
        features (numpy.ndarray): song features
        track2id (dict[string] -> int): map between msd-tid to
                                        index in feature mat
        items (list of str): map of the items for user_item maps
        mixture (int or None): indicate whether the model is
                               unimodal Gaussian or mixture of k components
        alpha (int): controls the confidence of comsumption
                     ignored for GMM cases
    
    Returns:
        numpy.ndarray: computed user profiles
    """
    user_profile = []
    for u in range(user_item.shape[0]):
        idx, dat = slice_row_sparse(user_item, u)
        dat += alpha
        dat = dat / dat.sum()
        
        # TODO: non-weithed / GMM case
        x = dat @ feature[[track2id[items[i]] for i in idx]]
        
        user_profile.append(x)
        
    return np.array(user_profile)
    

def eval_model(model, train_data, test_data,
               n_test_users, topn, verbose=False):
    # train model
    test_scores = []
    if n_test_users <= 0:
        rng = np.arange(test_data.shape[0])
    else:
        rng = np.random.permutation(test_data.shape[0])[:n_test_users]
        
    with tqdm(total=len(rng), ncols=80, disable=not verbose) as p:
        for u in rng:
            test_gt, _ = slice_row_sparse(test_data, u)
            pred = model.recommend(u, train_data, topn)
            test_scores.append(ndcg(test_gt, pred, k=topn))
            p.update(1)
    return test_scores


def instantiate_model(model_class, k, alpha, Xtr, features, track2id, items):
    """"""
    k = int(k)  # should be cased to integer
    
    # instantiate a model
    if model_class == UserKNN:
        user_profile = build_user_profile(
            Xtr, features, track2id, items, alpha
        )
        model = model_class(k, user_profile)
    elif model_class == ItemKNN:
        item_profile = features[
            [track2id[items[i]] for i in range(Xtr.shape[1])]
        ]
        model = model_class(k, item_profile)
    elif model_class == MostPopular:
        model = model_class(Xtr)
    elif model_class == Random:
        model = model_class()
    elif model_class == WRMF:
        model = model_class(k)
        model.als.fit(Xtr.T * alpha, show_progress=False)
    return model
            

def get_model_instance(model_class, Xtr, Xvl, features,
                       track2id, items, n_test_users, topn,
                       n_opt_calls=50, rnd_state=0):
    """"""
    # hyper param search range
    search_spaces = {
        UserKNN: [
            Real(1, 1000, "log-uniform", name='k'),
            Real(1, 1000, "log-uniform", name='alpha')
        ],
        ItemKNN: [Real(1, 1000, "log-uniform", name='k')],
        WRMF: [
            Real(1, 500, "log-uniform", name='k'),
            Real(0.0001, 1000, "log-uniform", name='alpha')
        ],
    }
    
    if model_class in search_spaces:
        # setup objective func evaluated by optimizer
        @use_named_args(search_spaces[model_class])
        def _objective(**params):
            if model_class == ItemKNN:
                params['alpha'] = -1

            model = instantiate_model(
                model_class,
                Xtr=Xtr, features=features,
                track2id=track2id, items=items,
                **params
            )
            # evaluate the model
            scores = eval_model(model, Xtr, Xvl, n_test_users, topn)
            return -np.mean(scores)

        # search best model with gaussian process based
        res_gp = gp_minimize(
            _objective, search_spaces[model_class],
            n_calls=n_opt_calls, random_state=rnd_state
        )
        print(res_gp.fun)

        if model_class == ItemKNN:
            k = res_gp.x[0]
            alpha = -1
        else:
            k, alpha = res_gp.x
    else:
        k, alpha = 0, 0  # no parameter needed

    best_model = instantiate_model(
        model_class, k, alpha,
        Xtr=Xtr + Xvl, features=features,
        track2id=track2id, items=items
    )
    return best_model


def setup_argparser():
    # setup argparser
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('recsys_fn', type=str,
                        help='filename of the recsys interaction data')
    parser.add_argument('feature_fn', type=str,
                        help='filename to the lyrics feature file')
    parser.add_argument('--out-fn', type=str, default='test.csv',
                        help='filename for the test output')
    parser.add_argument('--topn', type=int, default=100,
                        help='truncate threshold for evaulation')
    parser.add_argument('--n-test-users', type=int, default=2000,
                        help='number of testing users')
    parser.add_argument('--n-rep', type=int, default=5,
                        help='number of testing users')
    args = parser.parse_args() 
    return args


def full_factorial_design(text_feats=['personality', 'linguistic', 'topic']):
    """""" 
    cases = get_all_comb(text_feats)
    designs = []
    feats = {key:False for key in text_feats}  
    for model in ['WRMF', 'Random', 'MostPopular']:
        row = {'model':model} 
        row.update(feats) 
        designs.append(row)
        
    for case in chain.from_iterable(cases):         
        feats = dict.fromkeys(text_feats, False)
        for feat in case:
            feats[feat] = True
            
        for model in ['UserKNN', 'ItemKNN']: 
            row = {'model': model} 
            row.update(feats) 
            designs.append(row) 
        
    return designs


if __name__ == "__main__":
    # build argparser
    args = setup_argparser()
    
    # load relevant packages
    import os
    os.environ['MKL_NUM_THREADS'] = '1'
    from itertools import chain
    import pandas as pd
    import h5py
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    from tqdm import tqdm
    
    from ..metrics import ndcg
    from ..utils import (load_csr_data,
                         split_recsys_data,
                         slice_row_sparse,
                         prepare_feature,
                         get_all_comb)

    # get full factorial design
    configs = full_factorial_design()
        
    # load the recsys data
    X, users, items = load_csr_data(args.recsys_fn)

    # prepare feature
    feature, track2id, pca, sclr = prepare_feature(args.feature_fn)
    
    # run
    result = []
    for i, conf in enumerate(configs):
        print(conf)
        print('Running {:d}th / {:d} run...'.format(i+1, len(configs)))

        # prepare feature according to the design
        if any([v for k, v in conf.items() if k != 'model']):
            Y = np.concatenate([v for k, v in feature.items() if conf[k]], axis=1)
        else:
            Y = None
            
        for j in range(args.n_rep):
            # split the data 
            Xtr, Xvl, Xts = split_recsys_data(
                X, train_ratio=0.7, valid_ratio=0.15
            )

            # get the model
            model_class = eval(conf['model'])
            model = get_model_instance(
                model_class, Xtr, Xvl, Y,
                track2id, items, args.n_test_users, args.topn
            )

            # evaluate the best model
            test = eval_model(
                model, Xtr + Xvl, Xts,
                args.n_test_users, args.topn
            )

            # register results
            metrics = {'trial': j, 'score': np.mean(test)}
            conf_copy = conf.copy()
            conf_copy.update(metrics)
            result.append(conf_copy)
        
    # save
    pd.DataFrame(result).to_csv(args.out_fn)