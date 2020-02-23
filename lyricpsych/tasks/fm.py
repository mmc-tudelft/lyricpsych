import numpy as np
import numba as nb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence

from tqdm import tqdm


# TODO: make decorator for checking the model is already fitted
class FactorizationMachine(nn.Module):
    def __init__(self, k, init=0.001, n_iters=10, learn_rate=0.001, l2=0.0001,
                 n_negs=10, use_gpu=False):
        super().__init__()
        self.k = k
        self.init = init
        self.learn_rate = learn_rate
        self.l2 = l2
        self.n_iters = n_iters
        self.n_negs = n_negs
        # TODO: generalization of the device selection (not only for cuda)
        self.target_device = 'cuda' if use_gpu else 'cpu'

    @property
    def device(self):
        return next(self.parameters()).device
        
    def _init_embeddings(self):
        """"""
        if not hasattr(self, 'embeddings_'):
            raise ValueError()

        for key, layer in self.embeddings_.items():
            layer.weight.data.copy_(torch.randn(*layer.weight.shape) * self.init)
        
    def _init_optimizer(self):
        """
        """
        if not hasattr(self, 'embeddings_'):
            raise ValueError('You should call .fit first!')
        
        params = {'sparse':[], 'dense':[]}
        for lyr in self.embeddings_.children():
            if lyr.sparse:
                params['sparse'].append(lyr.weight)
            else:
                params['dense'].append(lyr.weight)
                
        # determine if sparse optimizer needed
        if len(params['sparse']) == 0:
            self.opt = optim.Adam(
                params['dense'], lr=self.learn_rate, weight_decay=self.l2
            )
        elif len(params['dense']) == 0:
            self.opt = optim.SparseAdam(params['sparse'], lr=self.learn_rate)
        else:
            # init multi-optimizers
            # register it to the instance
            self.opt = MultipleOptimizer(
                optim.SparseAdam(params['sparse'], lr=self.learn_rate),
                optim.Adam(params['dense'], 
                           lr=self.learn_rate,
                           weight_decay=self.l2)
            )
            
    def _retrieve_factors(self, u, i, feats):
        """"""
        if not hasattr(self, 'embeddings_'):
            raise ValueError('You should call .fit first!')

        # retrieve embeddins
        user = self.embeddings_['user'](u)
        item = self.embeddings_['item'](i)
        if self.embeddings_['feat'].sparse:
            feat = self.embeddings_['feat'](feats)
        else:
            feat = self.embeddings_['feat'].weight[None] * feats[..., None]
        
        # post-process (devide factor and weight)
        w = torch.cat([user[..., -1], item[..., -1], feat[..., -1]], dim=1)
        v = torch.cat([user[..., :-1], item[..., :-1], feat[..., :-1]], dim=1)
        return w, v

    def _update_z(self, item_feature, verbose=True):
        """
        this method pre-compute & cache item-tag factor for prediction
        """
        if not hasattr(self, 'embeddings_'):
            raise ValueError('You should call .fit first!')

        n_items = item_feature.shape[0]
        Z = torch.zeros((item_feature.shape[0], self.k+1))
        Z2 = torch.zeros((item_feature.shape[0], self.k))
        for i in tqdm(range(n_items), disable=not verbose, ncols=80):
            
            item = torch.LongTensor([i])
            feat = item_feature[i]

            z = [self.embeddings_['item'](item).detach()]
            if self.embeddings_['feat'].sparse:
                z += [self.embeddings_['feat'](feat).detach()]
            else:
                z += [
                    (self.embeddings_['feat'].weight * feat[..., None]).detach()
                ]
            z = torch.cat(z)

            Z[i] = z.sum(0)
            Z2[i] = (z[..., :-1]**2).sum(0)
                
        # register
        self.register_buffer('Z_v', Z[..., :-1])
        self.register_buffer('Z_w', Z[..., -1][..., None])
        self.register_buffer('Z2_v', Z2)

    def predict_user(self, user):
        if not hasattr(self, 'embeddings_'):
            raise ValueError('You should call .fit first!')
        user = torch.LongTensor([user]).to(self.device)
        user = self.embeddings_['user'](user)
        u_w = user[..., -1][..., None]  # (bsz, 1)
        u_v = user[..., :-1]  # (bsz, k)
        
        # infer
        w = u_w[:, None] + self.Z_w[None]
        v = u_v[:, None] + self.Z_v[None]
        v2 = u_v[:, None]**2 + self.Z2_v[None]
        
        s = self.w0 + w[:, 0] + (v**2 - v2).sum(-1) * .5
        return s[0]
        
    def forward(self, u, i, feats):
        """
        u (torch.LongTensor): user ids
        i (torch.LongTensor): item ids
        feats (torch.FloatTensor): item feature tensor
        """
        if not hasattr(self, 'embeddings_'):
            raise ValueError('You should call .fit first!')
            
        w, v = self._retrieve_factors(u, i, feats)
        w_ = w.sum(1)
        v_ = (v.sum(1)**2 - (v**2).sum(1)).sum(-1) * .5
        s = self.w0 + w_ + v_
        return s, [w, v]

    def fit(self, user_item, item_feature, feature_sparse=False,
            report_every=2000, batch_sz=512, verbose=False, n_jobs=4):
        """"""
        # TODO: internal validation if specified

        # set some variables
        feats = torch.Tensor(item_feature)  # convert feature to the torch Tensor
        n_users, n_items = user_item.shape
        desc_tmp = '[tloss=0.0000]'

        # initialize the factors
        self.register_parameter('w0', nn.Parameter(torch.FloatTensor([0])))
        self.embeddings_ = nn.ModuleDict({
            'user': nn.Embedding(user_item.shape[0], self.k + 1, sparse=True),
            'item': nn.Embedding(user_item.shape[1], self.k + 1, sparse=True),
            'feat': nn.Embedding(item_feature.shape[1],
                                 self.k + 1, sparse=feature_sparse)
        })
        self._init_embeddings()
        if self.target_device != 'cpu':
            self.to(self.target_device)
        
        # prepare dataset
        db = RecFeatDataset(user_item, n_negs=self.n_negs,
                            item_feature=item_feature)
        dl = DataLoader(db, batch_sz, collate_fn=collate_triplets_with_feature,
                        shuffle=True, num_workers=n_jobs, drop_last=True)
        
        # init optimizer
        self._init_optimizer()

        # do the optimization
        try:
            for i in range(self.n_iters):
                with tqdm(total=len(dl), desc=desc_tmp,
                        ncols=80, disable=not verbose) as prog:
                    for batch in dl:
                        # send batch to the device
                        u, i, y, x = tuple([x.to(self.target_device) for x in batch])

                        # flush grads
                        self.opt.zero_grad()

                        # forward & compute loss
                        scores, weights = self.forward(u, i, x)

                        # compute the main loss
                        # TODO: generalize loss function
                        z = scores * y
                        loss = -F.logsigmoid(z).sum()

                        # adding L2 regularization term for sparse entries (user/item)
                        loss += self.l2 * sum([torch.sum(w[:,:2]**2) for w in weights])

                        # backward
                        loss.backward()

                        # update params
                        self.opt.step()

                        # update loss and counter
                        prog.set_description('[tloss={:.4f}]'.format(
                            loss.item()/batch_sz))
                        prog.update(1)
        except KeyboardInterrupt as e:
            print('User stopped the training!...')
        finally:
            # update the cached factors for faster inference
            self.cpu()  # since below process eats up lots of memory
            self._update_z(feats.to('cpu'), verbose=verbose)
            self.to(self.target_device)
            

# TODO: make decorator for checking the model is already fitted
class FactorizationMachine2(nn.Module):
    def __init__(self, k, init=0.001, n_iters=10, learn_rate=0.001, l2=0.0001,
                 n_negs=10, use_gpu=False):
        super().__init__()
        self.k = k
        self.init = init
        self.learn_rate = learn_rate
        self.l2 = l2
        self.n_iters = n_iters
        self.n_negs = n_negs
        # TODO: generalization of the device selection (not only for cuda)
        self.target_device = 'cuda' if use_gpu else 'cpu'

    @property
    def device(self):
        return next(self.parameters()).device
        
    def _init_embeddings(self):
        """"""
        if not hasattr(self, 'embeddings_'):
            raise ValueError()

        for key, layer in self.embeddings_.items():
            layer.weight.data.copy_(torch.randn(*layer.weight.shape) * self.init)
        
    def _init_optimizer(self):
        """
        """
        if not hasattr(self, 'embeddings_'):
            raise ValueError('You should call .fit first!')
        
        params = {'sparse':[], 'dense':[]}
        for lyr in self.embeddings_.children():
            if lyr.sparse:
                params['sparse'].append(lyr.weight)
            else:
                params['dense'].append(lyr.weight)
                
        # determine if sparse optimizer needed
        if len(params['sparse']) == 0:
            self.opt = optim.Adam(
                params['dense'], lr=self.learn_rate, weight_decay=self.l2
            )
        elif len(params['dense']) == 0:
            self.opt = optim.SparseAdam(params['sparse'], lr=self.learn_rate)
        else:
            # init multi-optimizers
            # register it to the instance
            self.opt = MultipleOptimizer(
                optim.SparseAdam(params['sparse'], lr=self.learn_rate),
                optim.Adam(params['dense'], 
                           lr=self.learn_rate,
                           weight_decay=self.l2)
            )
            
    def _retrieve_factors(self, u, i, feats):
        """"""
        if not hasattr(self, 'embeddings_'):
            raise ValueError('You should call .fit first!')

        # retrieve embeddins
        user = self.embeddings_['user'](u)
        item = self.embeddings_['item'](i)
        if self.embeddings_['feat'].sparse:
            feat = self.embeddings_['feat'](feats)
        else:
            feat = self.embeddings_['feat'].weight[None] * feats[..., None]
        
        # post-process (devide factor and weight)
        w = torch.cat([user[..., -1], item[..., -1], feat[..., -1]], dim=1)
        v = torch.cat([user[..., :-1], item[..., :-1], feat[..., :-1]], dim=1)
        return w, v

    def _update_z(self, item_feature, verbose=True):
        """
        this method pre-compute & cache item-tag factor for prediction
        """
        if not hasattr(self, 'embeddings_'):
            raise ValueError('You should call .fit first!')
            
        # update item z
        feats = torch.FloatTensor(item_feature).to(self.device)
        vi = self.embeddings_['item'].weight.detach()
        vf = self.embeddings_['feat'].weight.detach()
        zf = feats @ vf
        zf2 = feats**2 @ vf**2

        # raise one blank dimension for batch dim
        z = (vi + zf)[None]
        z2 = (vi**2 + zf2)[None]

        # register
        self.register_buffer('z', z)
        self.register_buffer('z2', z2)

    def predict_user(self, user):
        if not hasattr(self, 'embeddings_'):
            raise ValueError('You should call .fit first!')
        user = torch.LongTensor([user]).to(self.device)
        vu = self.embeddings_['user'](user)[:, None]
        w = self.z[..., -1] + vu[..., -1]
        v = (self.z[..., :-1] + vu[..., :-1])**2
        v -= (self.z2[..., :-1] + vu[..., :-1]**2)
        v = v.sum(-1) * .5
        s = self.w0 + w + v
        return s[0]
        
    def forward(self, u, i, feats):
        """
        u (torch.LongTensor): user ids
        i (torch.LongTensor): item ids
        feats (torch.FloatTensor): item feature tensor
        """
        if not hasattr(self, 'embeddings_'):
            raise ValueError('You should call .fit first!')
            
        w, v = self._retrieve_factors(u, i, feats)
        w_ = w.sum(1)
        v_ = (v.sum(1)**2 - (v**2).sum(1)).sum(-1) * .5
        s = self.w0 + w_ + v_
        return s, [w, v]

    def fit(self, user_item, item_feature, feature_sparse=False,
            report_every=2000, batch_sz=512, verbose=False, n_jobs=4):
        """"""
        # TODO: internal validation if specified

        # set some variables
        feats = torch.Tensor(item_feature)  # convert feature to the torch Tensor
        n_users, n_items = user_item.shape
        desc_tmp = '[tloss=0.0000]'

        # initialize the factors
        self.register_parameter('w0', nn.Parameter(torch.FloatTensor([0])))
        self.embeddings_ = nn.ModuleDict({
            'user': nn.Embedding(user_item.shape[0], self.k + 1, sparse=True),
            'item': nn.Embedding(user_item.shape[1], self.k + 1, sparse=False),
            'feat': nn.Embedding(item_feature.shape[1],
                                 self.k + 1, sparse=feature_sparse)
        })
        self._init_embeddings()
        if self.target_device != 'cpu':
            self.to(self.target_device)
            feats = feats.to(self.device)
        
        # init optimizer
        self._init_optimizer()

        # do the optimization
        try:
            for i in range(self.n_iters):
                rnd_usrs = np.random.permutation(n_users)
                itr_usrs = np.arange(0, n_users, batch_sz)
                with tqdm(total=len(itr_usrs), desc=desc_tmp,
                        ncols=80, disable=not verbose) as prog:
                    iter_loss = 0
                    for u0 in itr_usrs:
                        # retrieve user batch
                        u = rnd_usrs[u0:u0 + batch_sz]
                        
                        # retrieve interaction slice
                        y = user_item[u]
                        y.data[:] = 1
                        y = scisp2tchsp(y.tocoo()).to(self.device)
                        
                        # update item z
                        vu = self.embeddings_['user'](torch.LongTensor(u).to(self.device))
                        vi = self.embeddings_['item'].weight
                        vf = self.embeddings_['feat'].weight
                        zf = feats @ vf
                        zf2 = feats**2 @ vf**2
                        
                        # raise one blank dimension for batch dim
                        z = (vi + zf)[None]
                        z2 = (vi**2 + zf2)[None]
                        
                        # flush grads
                        self.opt.zero_grad()

                        # forward & compute loss
                        vu_ = vu[:, None]  # (batch_sz, k+1)
                        w = z[..., -1] + vu_[..., -1]
                        v = (z[..., :-1] + vu_[..., :-1])**2
                        v -= (z2[..., :-1] + vu_[..., :-1]**2)
                        v = v.sum(-1) * .5
                        s = self.w0 + w + v
                        
                        # compute the main loss
                        # TODO: generalize loss function
                        loss = F.binary_cross_entropy_with_logits(
                            s, y.to_dense(), reduction="sum"
                        )

                        # adding L2 regularization term for sparse entries (user/item)
                        loss += self.l2 * sum([
                            torch.sum(w**2) for w in [vu]
                        ])

                        # backward
                        loss.backward()

                        # update params
                        self.opt.step()

                        # update loss and counter
                        iter_loss += loss.item()
                        prog.set_description(
                            '[tloss={:.4f}]'.format(loss.item())
                        )
                        prog.update(1)

                    prog.set_description(
                        '[tloss={:.4f}]'.format(iter_loss / len(itr_usrs))
                    )
                    
        except KeyboardInterrupt as e:
            print('User stopped the training!...')
        finally:
            # update the cached factors for faster inference
            self.cpu()  # since below process eats up lots of memory
            self._update_z(feats.to('cpu'), verbose=verbose)
            self.to(self.target_device)
            
            
class MultipleOptimizer:
    """ Simple wrapper for the multiple optimizer scenario """
    def __init__(self, *op):
        """"""
        self.optimizers = op
    
    def zero_grad(self):
        """"""
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        """"""
        for op in self.optimizers:
            op.step()

    def state_dict(self):
        """"""
        return [op.state_dict() for op in self.optimizers]

    def load_state_dict(self, state_dict):
        """"""
        return [op.load_state_dict(state) for state in state_dict]
    

class RecFeatDataset(Dataset):
    def __init__(self, user_item, user_feature=None, item_feature=None,
                 n_negs=10, device='cpu'):
        super().__init__()
        self.user_item = user_item
        self.user_feature = user_feature
        self.item_feature = item_feature
        self.n_negs = n_negs
        self.device = device

    def _draw_data(self, u, user_item):
        i0, i1 = user_item.indptr[u], user_item.indptr[u+1]
        pos = user_item.indices[i0:i1]
        j0 = np.random.choice(len(pos))
        negs = negsamp_vectorized_bsearch(
            pos, user_item.shape[1], n_samp=self.n_negs
        )
        return pos[j0][None], negs
    
    def _preproc(self, u, pos, negs):
        """"""
        u = torch.full((self.n_negs + 1,), u).long().to(self.device)
        i = torch.LongTensor(np.r_[pos, negs]).to(self.device)
        y = torch.LongTensor([1] + [-1] * self.n_negs).to(self.device)
        return u[:, None], i[:, None], y

    def __len__(self):
        return self.user_item.shape[0]
    
    def __getitem__(self, u_idx):
        pos, negs = self._draw_data(u_idx, self.user_item)
        u, i, y = self._preproc(u_idx, pos, negs)
        x = torch.FloatTensor(self.item_feature[i[:, 0]]).to(self.device)
        return u, i, y, x


def collate_triplets_with_feature(samples):
    """"""
    return tuple(map(torch.cat, zip(*samples)))


@nb.njit("i8[:](i4[:], i8, i8)")
def negsamp_vectorized_bsearch(pos_inds, n_items, n_samp=32):
    """ Pre-verified with binary search
    `pos_inds` is assumed to be ordered
    """
    raw_samp = np.random.randint(0, n_items - len(pos_inds), size=n_samp)
    pos_inds_adj = pos_inds - np.arange(len(pos_inds))
    ss = np.searchsorted(pos_inds_adj, raw_samp, side='right')
    neg_inds = raw_samp + ss
    return neg_inds


def scisp2tchsp(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def sparse_dense_mul(s, d):
    i = s._indices()
    v = s._values()
    dv = d[i[0,:], i[1,:]]  # get values from relevant entries of dense matrix
    return torch.sparse.FloatTensor(i, v * dv, s.size())