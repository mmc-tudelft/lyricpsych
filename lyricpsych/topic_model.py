import numba
import numpy as np


class PLSA:
    def __init__(self, k, n_iters=30):
        self.k = k
        self.n_iters = n_iters

    def fit(self, X):
        coo = X.tocoo()
        theta, phi = self._init_params(coo)

        plsa_numba(
            coo.row, coo.col, coo.data,
            theta, phi, self.n_iters
        )

        # assign
        self.doc_topic = theta
        self.topic_term = phi

    def _init_params(self, X, init_theta=True, init_phi=True):
        theta, phi = None, None

        if init_theta:
            theta = np.random.rand(X.shape[0], self.k)
            theta = theta / theta.sum(1)[:, None]
            theta = theta.astype(np.float32)

        if init_phi:
            phi = np.random.rand(X.shape[1], self.k).T
            phi = phi / phi.sum(0)[None]
            phi = phi.astype(np.float32)

        return theta, phi

    def transform(self, X):
        if not hasattr(self, 'topic_term'):
            raise Exception('[ERROR] .fit should be called before!')

        X = X.tocoo()
        theta = self._init_params(X, init_phi=False)[0]

        _learn_doc_topic(
            X.row, X.col, X.data,
            theta, self.topic_term, self.n_iters
        )
        return theta

    def score(self, X):
        return _perplexity(X, self, self.n_iters)


@numba.jit(nopython=True)
def plsa_numba(dt_row, dt_col, dt_val, topic_doc, term_topic, n_iter):
    """ PLSA numba. this function is directly employed from:

    https://github.com/henryre/numba-plsa
    """
    n_docs, n_topics = topic_doc.shape
    n_terms = term_topic.shape[1]

    nnz = len(dt_val)
    topic_full = np.zeros((nnz, n_topics))

    term_sum = np.zeros((n_topics))
    doc_sum = np.zeros((n_docs))

    for i in range(n_iter):
        ### Expectation ###
        for idx in range(nnz):
            p = np.zeros((n_topics))
            d, t = dt_row[idx], dt_col[idx]
            s = 0
            for z in range(n_topics):
                p[z] = topic_doc[d, z] * term_topic[z, t]
                s += p[z]
            if s == 0:
                s = 1e-10
            for z in range(n_topics):
                topic_full[idx, z] = p[z] / s
        ### Maximization ###
        topic_doc[:] = 0
        term_topic[:] = 0
        term_sum[:] = 1e-10
        doc_sum[:] = 1e-10
        for idx in range(nnz):
            for z in range(n_topics):
                q = dt_val[idx] * topic_full[idx, z]
                term_topic[z, dt_col[idx]] += q
                term_sum[z] += q
                topic_doc[dt_row[idx], z] += q
                doc_sum[dt_row[idx]] += q
        # Normalize P(topic | doc)
        for d in range(n_docs):
            for z in range(n_topics):
                topic_doc[d, z] /= doc_sum[d]
        # Normalize P(term | topic)
        for z in range(n_topics):
            for t in range(n_terms):
                term_topic[z, t] /= term_sum[z]


@numba.jit(nopython=True)
def _learn_doc_topic(dt_row, dt_col, dt_val, topic_doc, term_topic, n_iter):
    """ PLSA numba. this function is directly employed from:

    https://github.com/henryre/numba-plsa
    """
    n_docs, n_topics = topic_doc.shape
    n_terms = term_topic.shape[1]

    nnz = len(dt_val)
    topic_full = np.zeros((nnz, n_topics))

    term_sum = np.zeros((n_topics))
    doc_sum = np.zeros((n_docs))

    for i in range(n_iter):
        ### Expectation ###
        for idx in range(nnz):
            p = np.zeros((n_topics))
            d, t = dt_row[idx], dt_col[idx]
            s = 0
            for z in range(n_topics):
                p[z] = topic_doc[d, z] * term_topic[z, t]
                s += p[z]
            if s == 0:
                s = 1e-10
            for z in range(n_topics):
                topic_full[idx, z] = p[z] / s
        ### Maximization ###
        term_topic[:] = 0
        term_sum[:] = 1e-10
        doc_sum[:] = 1e-10
        for idx in range(nnz):
            for z in range(n_topics):
                q = dt_val[idx] * topic_full[idx, z]
                term_topic[z, dt_col[idx]] += q
                term_sum[z] += q
                doc_sum[dt_row[idx]] += q

        # Normalize P(term | topic)
        for z in range(n_topics):
            for t in range(n_terms):
                term_topic[z, t] /= term_sum[z]


def _perplexity(test_docs, plsa, n_iter=30):
    """ Compute perplexity of given topic model

    Inputs:
        test_docs (scipy.sparse.csr_matrix): hold-out document
        plsa (PLSA): trained pLSA model

    Returns:
        float: perplexity
    """
    new_phi = plsa.transform(test_docs)
    log_p_w_theta = np.zeros(test_docs.shape[1])
    for doc_idx in range(test_docs.shape[0]):
        phi = new_phi[doc_idx][None]
        internal_idx = slice(
            test_docs.indptr[doc_idx],
            test_docs.indptr[doc_idx+1]
        )
        idx = test_docs.indices[internal_idx]
        val = test_docs.data[internal_idx]

        log_p_w_theta[idx] += (
            np.log(np.maximum(phi @ plsa.topic_term[:, idx], 1e-14))[0] * val
        )

    perplexity = np.exp(-log_p_w_theta.sum() / test_docs.sum())
    return perplexity


def get_top_terms(plsa, id2word, topk=20):
    return [
        [
            id2word[t] for t
            in np.argsort(-plsa.topic_term[kk])[:topk]
        ]
        for kk in range(plsa.k)
    ]
