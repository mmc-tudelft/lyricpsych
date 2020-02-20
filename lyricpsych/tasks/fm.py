import numpy as np
import numba as nb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.utils.rnn import pad_packed_sequence
from tqdm import tqdm


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
        self.target_device = 'gpu' if use_gpu else 'cpu'
        
    def _init_embeddings(self):
        """"""
        for key, layer in self.embeddings_.items():
            layer.weight.data.copy_(torch.randn(*layer.weight.shape) * self.init)
        
    def _init_optimizer(self):
        """
        """
        if not hasattr(self, 'embeddings_'):
            raise ValueError()
        
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
        user = self.embeddings_['user'](u)
        item = self.embeddings_['item'](i)
        if self.embeddings_['feat'].sparse:
            feat = self.embeddings_['feat'](feats)
        else:
            feat = self.embeddings_['feat'].weight[None] * feats[..., None]
        
        w = torch.cat([user[..., -1], item[..., -1], feat[..., -1]], dim=1)
        v = torch.cat([user[..., :-1], item[..., :-1], feat[..., :-1]], dim=1)
        return w, v
    
    def _draw_data(self, u, user_item):
        i0, i1 = user_item.indptr[u], user_item.indptr[u+1]
        pos = user_item.indices[i0:i1]
        j0 = np.random.choice(len(pos))
        negs = negsamp_vectorized_bsearch(
            pos, user_item.shape[1], n_samp=self.n_negs
        )
        return pos[j0][None], negs
    
    def preproc(self, u, pos, negs):
        """"""
        u = torch.full((self.n_negs + 1,), u).long().to(self.target_device)
        i = torch.LongTensor(np.r_[pos, negs]).to(self.target_device)
        y = torch.LongTensor([1] + [-1] * self.n_negs)
        return u[:, None], i[:, None], y
    
    def _draw_batch(self, user_item, item_feature, batch_sz):
        """"""
        U, I, Y, X = [], [], [], []
        for u in np.random.choice(user_item.shape[0], batch_sz, False): 
            pos, negs = self._draw_data(u, user_item)
            u, i, y = self.preproc(u, pos, negs)
            x = item_feature[i[:, 0]].to(self.target_device)

            U.append(u)
            I.append(i)
            Y.append(y)
            X.append(x)
            
        return (torch.cat(U, dim=0), torch.cat(I, dim=0),
                torch.cat(Y, dim=0), torch.cat(X, dim=0))
        
    def forward(self, u, i, feats):
        """
        u (torch.LongTensor): user ids
        i (torch.LongTensor): item ids
        feats (torch.FloatTensor): item feature tensor
        """
        if not hasattr(self, 'embeddings_'):
            raise ValueError()
            
        w, v = self._retrieve_factors(u, i, feats)
        w = w.sum(1)
        v = (v.sum(1)**2 - (v**2).sum(1)).sum(-1) * .5
        s = self.w0 + w + v
        return s

    def fit(self, user_item, item_feature, feature_sparse=False,
            report_every=2000, batch_sz=512, verbose=False):
        """"""
        # initialize the factors
        self.register_parameter('w0', nn.Parameter(torch.FloatTensor([0])))
        self.embeddings_ = nn.ModuleDict({
            'user': nn.Embedding(user_item.shape[0], self.k + 1, sparse=True),
            'item': nn.Embedding(user_item.shape[1], self.k + 1, sparse=True),
            'feat': nn.Embedding(item_feature.shape[1],
                                 self.k + 1, sparse=feature_sparse)
        })
        self._init_embeddings()
        
        # init optimizer
        self._init_optimizer()
                
        # do the optimization
        self.losses_ = []
        feats = torch.Tensor(item_feature)
#         with tqdm(total=self.n_iters, ncols=80, disable=not verbose) as p:
        for i in range(self.n_iters):
            loss_sum = 0.
            j = 0
            n_users, n_items = user_item.shape
            desc_tmp = '[tloss=0.00000000]'
            rng = np.arange(0, n_users, batch_sz)
            with tqdm(total=len(rng), desc=desc_tmp,
                      ncols=80, disable=not verbose) as prog:
                for i in rng:
                    # draw batch
                    u, i, y, x = self._draw_batch(user_item, feats, batch_sz)

#                     # flush grads
#                     self.opt.zero_grad()

#                     # forward & compute loss
#                     s = self.forward(u, i, x) * y
#                     loss = -F.logsigmoid(s).sum()

#                     # backward
#                     loss.backward()

#                     # update params
#                     self.opt.step()

#                     # update loss and counter
#                     prog.set_description('[tloss={:.8f}]'.format(
#                         loss.item()/batch_sz))
#                     loss_sum += loss
#                     j += 1
#                     if (j + 1) % report_every == 0:
#                         self.losses_.append(
#                             (loss_sum.data.numpy() / report_every)
#                         )
#                         loss_sum = 0
#                         j = 0
                    prog.update(1)
        
    
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