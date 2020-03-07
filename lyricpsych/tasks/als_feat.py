import os
import logging

import numba as nb
import numpy as np
from scipy import sparse as sp

from tqdm import tqdm

from ..metrics import ndcg


class ALS:
    def __init__(self, k, init=0.001, l2=0.0001, n_iters=15,
                 alpha=5, eps=0.5, dtype='float32'):

        if dtype == 'float32':
            self.f_dtype = np.float32
        elif dtype == 'float64':
            self.f_dtype = np.float64
        else:
            raise ValueError('Only float32/float64 are supported!')
        
        self.k = k
        self.init = self.f_dtype(init)
        self.l2 = self.f_dtype(l2)
        self.alpha = self.f_dtype(alpha)
        self.eps = self.f_dtype(eps)
        self.dtype = dtype
        self.n_iters = n_iters

        check_blas_config()

    def _init_embeddings(self):
        for key, param in self.embeddings_.items():
            self.embeddings_[key] = param.astype(self.dtype) * self.init
    
    def fit(self, user_item, valid_user_item=None, verbose=False):
        """"""
        n_users, n_items = user_item.shape
        self.embeddings_ = {
            'user': np.random.randn(n_users, self.k),
            'item': np.random.randn(n_items, self.k),
        }
        self._init_embeddings()

        # preprocess data
        user_item = user_item.copy().astype(self.dtype)
        user_item.data = self.f_dtype(1) + user_item.data * self.alpha
        item_user = user_item.T.tocsr()

        dsc_tmp = '[vacc={:.4f}]'
        with tqdm(total=self.n_iters, desc='[vacc=0.0000]',
                  disable=not verbose, ncols=80) as p:

            for n in range(self.n_iters):
                # update user factors
                update_user_factor(
                    user_item.data, user_item.indices, user_item.indptr,
                    self.embeddings_['user'], self.embeddings_['item'], self.l2
                )
                
                # update item factors
                update_user_factor(
                    item_user.data, item_user.indices, item_user.indptr,
                    self.embeddings_['item'], self.embeddings_['user'], self.l2
                )

                if valid_user_item is not None:
                    score = self.validate(user_item, valid_user_item)
                    p.set_description(dsc_tmp.format(score))
                p.update(1)

    def validate(self, user_item, valid_user_item, n_tests=2000, topk=100):
        """"""
        scores = []
        if n_tests >= user_item.shape[0]:
            targets = range(user_item.shape[0])
        else:
            targets = np.random.choice(user_item.shape[0], n_tests, False)
        for u in targets:
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


class ALSFeat:
    def __init__(self, k, init=0.001, lmbda=1, l2=0.0001, n_iters=15,
                 alpha=5, eps=0.5, dropout=0.5, dtype='float32'):

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
        self.dropout = dropout

        check_blas_config()

    def _init_embeddings(self):
        for key, param in self.embeddings_.items():
            self.embeddings_[key] = param.astype(self.dtype) * self.init
    
    def dropout_items(self, item_user):
        """"""
        if self.dropout > 0:
            n_items = item_user.shape[0]
            dropout_items = np.random.choice(
                n_items, int(n_items * self.dropout), False
            )
            for i in dropout_items:
                i0, i1 = item_user.indptr[i], item_user.indptr[i+1]
                item_user.data[i0:i1] = 0
            item_user.eliminate_zeros()                
        return item_user
    
    def fit(self, user_item, item_feat, valid_user_item=None,
            verbose=False):
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
        
        # pre-compute XX
        item_feat2 = item_feat.T @ item_feat

        # scale hyper-parameters
        lmbda = self.lmbda
        l2 = self.l2

        dsc_tmp = '[vacc={:.4f}]'
        with tqdm(total=self.n_iters, desc='[vacc=0.0000]',
                  disable=not verbose, ncols=80) as p:

            for n in range(self.n_iters):
                IU = self.dropout_items(item_user.copy())
                UI = IU.T.tocsr()
                
                # update user factors
                update_user_factor(
                    UI.data, UI.indices, UI.indptr,
                    self.embeddings_['user'], self.embeddings_['item'], l2
                )

                # update item factors
                update_item_factor(
                    IU.data, IU.indices, IU.indptr,
                    self.embeddings_['user'], self.embeddings_['item'],
                    item_feat, self.embeddings_['feat'], lmbda, l2
                )

                # update feat factors
                update_feat_factor(
                    self.embeddings_['item'], item_feat, item_feat2,
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
        if n_tests >= user_item.shape[0]:
            targets = range(user_item.shape[0])
        else:
            targets = np.random.choice(user_item.shape[0], n_tests, False)
        for u in targets:
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


# @nb.njit
@nb.njit(nogil=True, parallel=True)
def update_user_factor(data, indices, indptr, U, V, lmbda):
    """"""
    VV = V.T @ V  # precompute
    d = V.shape[1]
    I = np.eye(d, dtype=VV.dtype)
    # randomize the order so that scheduling is more efficient
    rnd_idx = np.random.permutation(U.shape[0])
    
    # for n in range(U.shape[0]):
    for n in nb.prange(U.shape[0]):
        u = rnd_idx[n]
        u0, u1 = indptr[u], indptr[u + 1]
        if u1 - u0 == 0:
            continue
        ind = indices[u0:u1]
        val = data[u0:u1]
        U[u] = partial_ALS(val, ind, V, VV, lmbda)

        
# @nb.njit
@nb.njit(nogil=True, parallel=True)
def update_item_factor(data, indices, indptr, U, V, X, W, lmbda_x, lmbda):
    """"""
    UU = U.T @ U
    d = U.shape[1]
    h = X.shape[1]
    I = np.eye(d, dtype=UU.dtype)
    # randomize the order so that scheduling is more efficient
    rnd_idx = np.random.permutation(V.shape[0])
    
    for n in nb.prange(V.shape[0]):
    # for n in range(V.shape[0]):
        i = rnd_idx[n]
        i0, i1 = indptr[i], indptr[i+1]
        # if i1 - i0 == 0:
        #     continue
        ind = indices[i0:i1]
        val = data[i0:i1]
        V[i] = partial_ALS_feat(val, ind, U, UU, X[i], W, lmbda_x, lmbda)

        
@nb.njit
def update_feat_factor(V, X, XX, W, lmbda_x, lmbda):
    h = X.shape[1]
    I = np.eye(h, dtype=V.dtype)
    # d = V.shape[1]
    # A = np.zeros((h, h))
    # B = np.zeros((h, d))
    
    A = XX + (lmbda / lmbda_x) * I
    # for f in range(h):
    #     for q in range(f, h):
    #         if f == q:
    #             A[f, q] += lmbda / lmbda_x 
    #         for j in range(X.shape[0]):
    #             A[f, q] += X[j, f] * X[j, q]
    # A = A + A.T - np.diag(A)
    
    B = X.T @ V
    # for f in range(h):
    #     for r in range(d):
    #         for j in range(X.shape[0]):
    #             B[f, r] += X[j, f] * V[j, r]
    
    # update feature factors
    W = np.linalg.solve(A, B)


@nb.njit
def partial_ALS(data, indices, V, VV, lmbda):
    d = V.shape[1]
    b = np.zeros((d,))
    A = np.zeros((d, d))
    c = data + 0
    vv = V[indices].copy()

    # b = np.dot(c, V[ind])
    for f in range(d):
        for j in range(len(c)):
            b[f] += c[j] * vv[j, f]
    
    # A = VV + vv.T @ np.diag(c - 1) @ vv + lmbda * I
    for f in range(d):
        for q in range(f, d):
            if q == f:
                A[f, q] += lmbda
            A[f, q] += VV[f, q]
            for j in range(len(c)):
                A[f, q] += vv[j, f] * (c[j] - 1) * vv[j, q]
                
    # copy the triu elements to the tril
    # A = A + A.T - np.diag(np.diag(A))
    for j in range(1, d):
        for k in range(j, d):
            A[k][j] = A[j][k]
    
    # update user factor
    return np.linalg.solve(A, b.ravel())
    
@nb.njit
def partial_ALS_feat(data, indices, U, UU, x, W, lmbda_x, lmbda):
    d = U.shape[1]
    b = np.zeros((d,))
    A = np.zeros((d, d))
    xw = np.zeros((d,))
    c = data + 0
    uu = U[indices].copy()

    # xw = x @ W
    for f in range(d):
        for h in range(len(x)):
            xw[f] += x[h] * W[h, f]

    # b = np.dot(c, uu) + lmbda_x * xw
    for f in range(d):
        b[f] += lmbda_x * xw[f]
        for j in range(len(c)):
            b[f] += c[j] * uu[j, f] 

    # A = UU + uu.T @ np.diag(c - 1) @ uu + (lmbda_x + lmbda) * I
    for f in range(d):
        for q in range(f, d):
            if q == f:
                A[f, q] += (lmbda + lmbda_x)
            A[f, q] += UU[f, q]
            for j in range(len(c)):
                A[f, q] += uu[j, f] * (c[j] - 1) * uu[j, q]

    # copy the triu elements to the tril
    # A = A + A.T - np.diag(np.diag(A))
    for j in range(1, d):
        for k in range(j, d):
            A[k][j] = A[j][k]

    return np.linalg.solve(A, b.ravel())


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