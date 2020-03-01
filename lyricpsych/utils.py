from os.path import basename, dirname, splitext, join
import string
from itertools import chain, combinations
from functools import partial

import numpy as np
from scipy import sparse as sp

import gensim
from gensim import corpora

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.util import ngrams

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import h5py

from tqdm import tqdm

from .topic_model import PLSA
from .files import mxm2msd as mxm2msd_fn

# intantiate lemmatizer / stopwords
ENGLISH_VOCAB = set(w.lower() for w in nltk.corpus.words.words())
SW = set(stopwords.words('english'))
LEMMA = WordNetLemmatizer()


def compute_non_englishness(dataset, verbose=True):
    """ Detects unusual words from lyrics dataset
    """
    unusuals = []
    for tid, lyrics in tqdm(dataset, ncols=80, disable=not verbose):
        if lyrics != '':
            text_vocab = set(
                w for w
                in nltk.word_tokenize(lyrics.lower())
                if w.isalpha()
            )
            unusual = text_vocab.difference(ENGLISH_VOCAB)
            unusuals.append((tid, len(unusual)))
        else:
            unusuals.append((tid, len(ENGLISH_VOCAB)))
    return unusuals


def _process_word(word, filter_stopwords=False):
    """ Pre-process word. this sub-function is for consistency.
    
    1) Stopwords (English) are filtered (optional)
    2) Lemmatize (WordNet lemmatizer from nltk)
    
    Inputs:
        word (string): target word to be processed
        filter_stopwords (bool): indicates whether stopwords are filtered
        
    Returns:
        string (or None): processed word (or not a good entry)
    """
    
    if not set(string.ascii_letters + " -").issuperset(word):
        return None

    word = LEMMA.lemmatize(word)
    
    if filter_stopwords and word in SW:
        return None
    
    return word


def represent_ngram(data, unit='unigram', filter_stopwords=True):
    """ Represent given texts as given n-gram type
    
    Inputs:
        data (list of string): raw data read from the disk
        unit (string): determine which kind of representation is used
                       {'unigram', 'bigram', 'trigram'}
        filter_stopwords (bool): indicates stop-words filtering explicitly
                       
    Returns:
        list of list of string: ngram text data
    """
    # do the chunking first
    new_text = [
        [w for w in nltk.wordpunct_tokenize(text.lower())]
        for text in data
    ]

    # lemmatize
    # -> build lemmatized dictionary for all unique words
    unique_words = set(chain.from_iterable(new_text))
    filt_sw = True if (unit == 'unigram') and filter_stopwords else False
    lemma_dict = {
        w:_process_word(w, filter_stopwords=filt_sw)
        for w in unique_words
    }
    
    # -> replace words to their lemmatized version
    new_text = [
        [lemma_dict[w] for w in sent if lemma_dict[w] is not None]
        for sent in new_text
    ]
    
    if unit == 'unigram':
        return new_text  # just output the unigram data
    else:
        # prepare container
        ngram_text = []
        
        # prepare processor
        chunk_size = 2 if unit == 'bigram' else 3
        _ngrams = partial(
            ngrams, n=chunk_size, pad_left=True, pad_right=True,
            left_pad_symbol='<s>', right_pad_symbol='</s>'
        )
        
        # process ngram for given unigram chunks
        for text in new_text:
            sentence = []
            for token in _ngrams(text):
                sentence.append('_'.join(token))
            ngram_text.append(sentence)
        return ngram_text


def preprocessing(data, unit='unigram', 
                  doc_frq_filt=[5, 0.8], filt_stopwords=True):
    """ Pre-processing raw-data (list of tuples of (id, string)) to corpus
    
    Inputs:
        data (list of string): raw data read from the disk
        unit (string): determine which kind of representation is used
                       {'unigram', 'bigram', 'trigram'}
        doc_frq_filt (list of int): doc-freq percentile to filter out the terms
    
    Returns:
        list of bow: corbus in bag-of-word format
        gensim.corpora.dictionary.Dictionary: dictionary contains unique words
        scipy.sparse.coo_matrix: song-words assignment matrix
    """       
    # preparing the data
    texts = represent_ngram(data, unit, filt_stopwords)
    
    # build matrix
    id2word = corpora.Dictionary(texts)
    
    # filter terms if it's extremely prevalent or scarce
    if doc_frq_filt is not None:
        id2word.filter_extremes(doc_frq_filt[0], doc_frq_filt[1])
    corpus = [id2word.doc2bow(text) for text in texts]
    
    rows, cols, vals = [], [], []
    for i, word_freq in enumerate(corpus):
        for j, v in word_freq:
            rows.append(i)
            cols.append(j)
            vals.append(v)
    X = sp.coo_matrix((vals, (rows, cols)))    
    return texts, corpus, id2word, X
    

def filter_english_plsa(dataset, k=20, thresh=15, top_k_words=50,
                        n_iter=1, verbose=False):
    """ Filter out non-english entries using topic-modeling
    
    Inputs:
        dataset (list of tuple of strings): contains lyrics data (index, lyrics)
        k (int): number of topics used 
        thresh (int): determines the topic is non-english one
                      if the number of non-english words larger than thresh
        top_k_words (int): number of top-k words for filtering
        n_iter (int): number of iteration of this procedure
        verbose (bool): verbosity. If True, compute total non-englishness every epoch 
    """
    # pre-process the data
    texts, corpus, id2word, X = preprocessing([r[1] for r in dataset])

    for j in range(n_iter): 
        
        # instantiate a PLSA and fit
        plsa = PLSA(k, 30)
        plsa.fit(X)
        
        # aliasing (for readability)
        theta = plsa.topic_term
        phi = plsa.doc_topic
        
        # process
        blacklist = []  # for tracks to be filtered
        for topic in range(k):
            
            # get topic related terms
            topic_terms = [
                id2word[i] for i
                in np.argsort(-theta[topic])[:top_k_words]
            ]
            
            # get number of unusual words (in English)
            num_unusuals = len([
                w for w in topic_terms
                if w not in ENGLISH_VOCAB
            ])
            
            if num_unusuals >= thresh:
                blacklist.append(np.where(phi.argmax(1) == topic)[0])
        
        # get the final list to be filtered out
        blacklist = set(np.concatenate(blacklist))
                
        # filter out
        dataset = [d for i, d in enumerate(dataset) if i not in blacklist]
        
        if verbose:
            total_non_englishness = sum(
                r[1] for r in compute_non_englishness(data)
            )
            print(
                '[iter / {:d}] - non_Englishness: {:d}'.format(
                    j, total_non_englishness
                )
            )
        
    return dataset
    
    
def dump_embeddings(fn, embeddings, metadata, metadata_columns):
    """ Dump embedding matrix and corresponding metadata to disk
    
    Inputs:
        fn (string): filename where embedding data is saved
        embeddings (numpy.ndarray): embedding matrix
        metadata (list of tuple): metadata
        metadata_columns (tuple): metadata namespaces
    """
    # infer metadata filename
    metadata_fn = join(
        dirname(fn),
        splitext(basename(fn))[0] + '_metadata.txt'
    )
    
    # save metadata
    with open(metadata_fn, 'w') as f:
        f.write('\t'.join(metadata_columns) + '\n')
        for row in metadata:
            f.write('\t'.join(row) + '\n')
    
    # save embedding mat
    np.savetxt(fn, embeddings, delimiter='\t')
    
    
def split_docs(X, train_ratio=0.8, keep_format=False, return_idx=True):
    """ Split given documents in train / test

    Inputs:
        X (scipy.sparse.csr_matrix): sparse matrix of document-term relationship
        train_ratio (float): ratio of training samples
        keep_format (bool): whether keeping its original format
        return_idx (bool): whether returning the indices
    
    Returns:
        scipy.sparse.csr_matrix: train matrix
        scipy.sparse.csr_matrix: test matrix
    """
    if not sp.isspmatrix_csr(X):
        org_fmt = X.format
        X = X.tocsr()        
    
    # split the data
    N = X.shape[0]
    idx = np.random.permutation(N)
    train_idx = idx[:int(train_ratio * N)]
    test_idx = idx[int(train_ratio * N):]
    
    Xtr = X[train_idx]
    Xts = X[test_idx]
    
    if keep_format:
        output = (Xtr.asformat(org_fmt), Xts.asformat(org_fmt))
    else:
        output = (Xtr, Xts)
    
    if return_idx:
        return output + (train_idx, test_idx)
    else:
        return output
    
    
def load_csr_data(h5py_fn, row='users', col='items'):
    """ Load recsys data stored in hdf format
    
    Inputs:
        fn (str): filename for the data
        
    Returns:
        scipy.sparse.csr_matrix: user-item matrix
        numpy.ndarray: user list
        numpy.ndarray: item list
    """
    import h5py
    with h5py.File(h5py_fn, 'r') as hf:
        data = (hf['data'][:], hf['indices'][:], hf['indptr'][:])
        X = sp.csr_matrix(data)
        rows = hf[row][:]
        cols = hf[col][:]
    return X, rows, cols


def split_recsys_data(X, train_ratio=0.8, valid_ratio=0.1):
    """ Split given user-item matrix into train/test.
    
    This split is to check typical internal ranking accracy.
    (not checking cold-start problem)
    
    Inputs:
        X (scipy.sparse.csr_matrix): user-item matrix
        train_ratio (float): ratio of training records per user
        test_ratio (float): ratio of validation records per user
    
    Returns:
        scipy.sparse.csr_matrix: training matrix
        scipy.sparse.csr_matrix: validation matrix
        scipy.sparse.csr_matrix: testing matrix
    """
    def _store_data(cur_i, container, indices, data, rnd_idx, start, end):
        container['I'].extend(np.full((end - start,), cur_i).tolist())
        container['J'].extend(indices[rnd_idx[start:end]].tolist())
        container['V'].extend(data[rnd_idx[start:end]].tolist())
        
    def _build_mat(container, shape):
        return sp.coo_matrix(
            (container['V'], (container['I'], container['J'])),
            shape=shape
        ).tocsr()
    
    # prepare empty containers
    train = {'V': [], 'I': [], 'J': []}
    valid = {'V': [], 'I': [], 'J': []}
    test = {'V': [], 'I': [], 'J': []}
    for i in range(X.shape[0]):
        idx, dat = slice_row_sparse(X, i)
        rnd_idx = np.random.permutation(len(idx))
        n = len(idx)
        train_bound = int(train_ratio * n)
        if np.random.rand() > 0.5:
            valid_bound = int(valid_ratio * n) + train_bound
        else:
            valid_bound = int(valid_ratio * n) + train_bound + 1

        _store_data(i, train, idx, dat, rnd_idx, 0, train_bound)
        _store_data(i, valid, idx, dat, rnd_idx, train_bound, valid_bound)
        _store_data(i, test, idx, dat, rnd_idx, valid_bound, n)
    
    return tuple(
        _build_mat(container, X.shape)
        for container in [train, valid, test]
    )


def slice_row_sparse(csr, i):
    slc = slice(csr.indptr[i], csr.indptr[i+1])
    return csr.indices[slc], csr.data[slc]


def prepare_feature(feature_fn):
    """"""
    # for some pre-processors
    pca = PCA(whiten=True)
    sclr = StandardScaler()
    
    # getting mxm->msd map
    mxm2msd = dict(
        [line.strip('\n').split(',') for line in open(mxm2msd_fn())]
    )
    
    # load the feature data and concatenate
    with h5py.File(feature_fn, 'r') as hf: 
        
        X = []
        bounds = [0]
        feature_sets = [
            k.split('_cols')[0]
            for k in hf['features'].keys()
            if 'cols' in k
        ]
        
        for feat in feature_sets:
            # fetch features per set
            x = hf['features'][feat][:]
            if feat == 'topic':
                x = pca.fit_transform(x) 
            else:
                if feat == 'audio':
                    xsum = x.sum(1)
                    non_zero_idx = np.where(xsum > 0)[0]
                    zero_idx = np.where(xsum == 0)[0]
                    sclr.fit(x[non_zero_idx])
                    x = sclr.transform(x)
                    x[zero_idx] = np.random.randn(len(zero_idx), x.shape[1])
                else:
                    x = sclr.fit_transform(x)
                    
            X.append(x) 
            bounds.append(bounds[-1] + x.shape[1])
        feature = np.concatenate(X, axis=1)
        track2id = {
            mxm2msd[t]:i for i, t
            in enumerate(hf['features']['ids'][:])
        }
        
    # split back the features with the bound
    features = {}
    for i, feat in enumerate(feature_sets):
        features[feat] = feature[:, bounds[i]:bounds[i+1]]  
    
    return features, track2id, pca, sclr, feature_sets


def get_all_comb(cases, include_null=False):
    combs = [
        combinations(cases, j)
        for j in range(1, len(cases) + 1)
    ]
    if include_null:
        combs.append(None)
    return combs 