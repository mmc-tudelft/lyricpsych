import os
import logging

import numba as nb
import numpy as np
from scipy import sparse as sp

from tqdm import tqdm

from ..metrics import ndcg


class ALSFeat:
    def __init__(self, k, init=0.001, lmbda=1, l2=0.0001, n_iters=15,
                 alpha=5, eps=0.5, dtype='float32'):

        if dtype == 'float32':
            self.f_dtype = np.float32
        elif dtype == 'float64':
            self.f_dtype = np.float64
        else:
            raise ValueError('Only float32/float64 are supported!')
        
        self.k = k
        self.init = self.f_dtype(init)
        self.lmbda = self.f_dtype(lmbda)
        self.l2 = self.f_dtype(l2)
        self.alpha = self.f_dtype(alpha)
        self.eps = self.f_dtype(eps)
        self.dtype = dtype
        self.n_iters = n_iters

        check_blas_config()

    def _init_embeddings(self):
        for key, param in self.embeddings_.items():
            self.embeddings_[key] = param.astype(self.dtype) * self.init
    
    def fit(self, user_item, item_feat, valid_user_item=None,
            out_mat_val=True, verbose=False):
        """"""
        n_users, n_items = user_item.shape
        n_feats = item_feat.shape[1]
        self.embeddings_ = {
            'user': np.random.randn(n_users, self.k),
            'item': np.random.randn(n_items, self.k),
            'feat': np.random.randn(n_feats, self.k)
        }
        self._init_embeddings()

        # preprocess data
        user_item = user_item.copy().astype(self.dtype)
        user_item.data = self.f_dtype(1) + user_item.data * self.alpha

        item_user = user_item.T.tocsr()
        item_feat = item_feat.astype(self.dtype)

        # scale hyper-parameters
        # lmbda = self.lmbda * (item_user.sum() / n_items * n_feats)
        # l2 = self.l2 * (item_user.sum())
        lmbda = self.lmbda
        l2 = self.l2

        dsc_tmp = '[vacc={:.4f}]'
        with tqdm(total=self.n_iters, desc='[vacc=0.0000]',
                  disable=not verbose, ncols=80) as p:

            for n in range(self.n_iters):
                # update user factors
                update_user_factor(
                    user_item.data, user_item.indices, user_item.indptr,
                    self.embeddings_['user'], self.embeddings_['item'],
                    l2, self.alpha, self.eps
                )

                # update item factors
                update_item_factor(
                    item_user.data, item_user.indices, item_user.indptr,
                    self.embeddings_['user'], self.embeddings_['item'],
                    item_feat, self.embeddings_['feat'],
                    lmbda, l2, self.alpha, self.eps
                )

                # update feat factors
                update_feat_factor(
                    self.embeddings_['item'], item_feat,
                    self.embeddings_['feat'], lmbda, l2
                )

                if valid_user_item is not None:
                    score = self.validate(user_item, item_feat, valid_user_item)
                    p.set_description(dsc_tmp.format(score))
                p.update(1)

    def validate(self, user_item, item_feat, valid_user_item,
                 n_tests=2000, topk=100):
        """"""
        scores = []
        for u in np.random.choice(user_item.shape[0], n_tests, False):
            true = valid_user_item[u].indices
            if len(true) == 0:
                continue
            train = user_item[u].indices
            s = self.embeddings_['user'][u] @ self.embeddings_['item'].T
            s[train] = -np.inf
            idx = np.argpartition(-s, kth=topk)[:topk]
            pred = idx[np.argsort(-s[idx])]
            scores.append(ndcg(true, pred, topk))
        return np.mean(scores)


@nb.njit(nogil=True, parallel=True)
def update_user_factor(data, indices, indptr, U, V, lmbda, alpha, eps):
    """"""
    VV = V.T @ V  # precompute
    d = V.shape[1]
    I = np.eye(d, dtype=VV.dtype)
    rnd_idx = np.random.permutation(U.shape[0])
    
    # for n in range(U.shape[0]):
    for n in nb.prange(U.shape[0]):
        u = rnd_idx[n]
        u0, u1 = indptr[u], indptr[u + 1]
        ind = indices[u0:u1]
        c = data[u0:u1] + 0
        vv = V[ind]

        b = np.dot(c, V[ind])
        A = VV + vv.T @ np.diag(c - 1) @ vv + lmbda * I
        U[u] = np.linalg.solve(A, b.ravel())


@nb.njit(nogil=True, parallel=True)
def update_item_factor(data, indices, indptr, U, V, X, W, lmbda_x, lmbda, alpha, eps):
    """"""
    UU = U.T @ U
    XW = X @ W
    d = U.shape[1]
    I = np.eye(d, dtype=UU.dtype)
    rnd_idx = np.random.permutation(V.shape[0])
    
    for n in nb.prange(V.shape[0]):
    # for n in range(V.shape[0]):
        i = rnd_idx[n]
        i0, i1 = indptr[i], indptr[i+1]
        if i1 - i0 == 0:
            continue

        ind = indices[i0:i1]
        c = data[i0:i1] + 0
        xw = XW[i].copy()
        uu = U[ind].copy()
        
        b = np.dot(c, uu) + lmbda_x * xw
        A = UU + uu.T @ np.diag(c - 1) @ uu + (lmbda_x + lmbda) * I
        V[i] = np.linalg.solve(A, b.ravel())


def update_feat_factor(V, X, W, lmbda_x, lmbda):
    h = X.shape[1]
    A = X.T @ X + lmbda / lmbda_x * np.eye(h)
    B = X.T @ V
    W = np.linalg.solve(A, B)


def check_blas_config():
    """ checks if using OpenBlas/Intel MKL
        This function directly adopted from
        https://github.com/benfred/implicit/blob/master/implicit/utils.py
    """
    pkg_dict = {'OPENBLAS':'openblas', 'MKL':'blas_mkl'}
    for pkg, name in pkg_dict.items():
        if (np.__config__.get_info('{}_info'.format(name))
            and
            os.environ.get('{}_NUM_THREADS'.format(pkg)) != '1'):
            logging.warning(
                "{} detected, but using more than 1 thread. Its recommended "
                "to set it 'export {}_NUM_THREADS=1' to internal multithreading"
                .format(name, name)
            )