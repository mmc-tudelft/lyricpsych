import numba
import numpy as np


class PLSA:
    def __init__(self, k, n_iters=30):
        self.k = k
        self.n_iters = n_iters

    def fit(self, X):
        coo = X.tocoo()
        U = np.random.rand(coo.shape[0], self.k)
        V = np.random.rand(coo.shape[1], self.k).T
        U, V = U / U.sum(1)[:, None], V / V.sum(0)[None]
        self.doc_topic = U.astype(np.float32)
        self.topic_term = V.astype(np.float32)
        
        plsa_numba(
            coo.row, coo.col, coo.data,
            self.doc_topic, self.topic_term,
            self.n_iters
        )

    
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