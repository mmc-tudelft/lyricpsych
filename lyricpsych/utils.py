from itertools import chain
from functools import partial

import numpy as np
from scipy import sparse as sp

import gensim
from gensim import corpora

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

import h5py

from tqdm import tqdm

from .extractors.base import TopicFeature
from .extractors.topic_model import PLSA
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

    X = sp.coo_matrix((vals, (rows, cols)),
                      shape=(len(texts), len(id2word.token2id)))
    return texts, corpus, id2word, X


def filter_english_plsa(dataset, k=20, thresh=15, top_k_words=50,
                        n_iter=1, preproc=True, verbose=False):
    """ Filter out non-english entries using topic-modeling

    Inputs:
        dataset (list of tuple of strings): contains lyrics data (index, lyrics)
        k (int): number of topics used
        thresh (int): determines the topic is non-english one
                      if the number of non-english words larger than thresh
        top_k_words (int): number of top-k words for filtering
        n_iter (int): number of iteration of this procedure
        preproc (bool): determine whether filtering the extreme words or not
        verbose (bool): verbosity. If True, compute total non-englishness every epoch
    """
    # pre-process the data
    texts, corpus, id2word, X = preprocessing(
        [r[1] for r in dataset], doc_frq_filt=[5, 0.8] if preproc else None
    )

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
        if len(blacklist) > 0:
            blacklist = np.concatenate(blacklist)
            blacklist = set(blacklist)
        else:
            break

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
    with h5py.File(h5py_fn, 'r') as hf:
        data = (hf['data'][:], hf['indices'][:], hf['indptr'][:])
        X = sp.csr_matrix(data)
        rows = hf[row][:]
        cols = hf[col][:]
    return X, rows, cols


def save_feature_h5(features, out_fn):
    """ Save extracted feature to the disk in HDF format

    Inputs:
        features (dict[string] -> TextFeature): extracted features
        out_fn (string): filename to dump the extracted features
    """
    if len(features) == 0:
        raise ValueError('[ERROR] No features found!')

    ids = list(features.values())[0].ids  # anchor
    with h5py.File(out_fn, 'w') as hf:
        hf.create_group('features')
        for key, feat in features.items():
            hf['features'].create_dataset(
                key, data=feat.features[[feat.inv_ids[i] for i in ids]]
            )
            hf['features'].create_dataset(
                key + '_cols',
                data=np.array(feat.columns, dtype=h5py.special_dtype(vlen=str))
            )

            if isinstance(feat, TopicFeature):
                id2token = {token:i for i, token in feat.id2word.items()}
                hf['features'].create_dataset(
                    'topic_terms', data=feat.topic_terms
                )
                hf['features'].create_dataset(
                    'id2word',
                    data=np.array(
                        [id2token[i] for i in range(len(id2token))],
                        dtype=h5py.special_dtype(vlen=str)
                    )
                )
        hf['features'].create_dataset(
            'ids', data=np.array(ids, dtype=h5py.special_dtype(vlen=str))
        )


def save_feature_csv(features, out_fn, delim=','):
    """ Save extracted feature to the disk in csv format

    Inputs:
        features (dict[string] -> TextFeature): extracted features
        out_fn (string): filename to dump the extracted features
        delim (string): delimiter for the separation
    """
    # aggregation process
    ids = list(features.values())[0].ids  # anchor
    # first aggregate the data
    agg_feature = []
    agg_feat_cols = []
    for key, feat in features.items():
        x = feat.features[[feat.inv_ids[i] for i in ids]]
        agg_feature.append(x)
        agg_feat_cols.extend([key + '_' + col for col in feat.columns])
    agg_feature = np.hstack(agg_feature)

    with open(out_fn, 'w') as f:
        f.write(delim.join(agg_feat_cols) + '\n')
        for row in agg_feature:
            f.write(delim.join(['{:.8f}'.format(y) for y in row]) + '\n')


def normalize_matrix(a, order=2, axis=1, eps=1e-12):
    """
    """
    norm = np.linalg.norm(a, order, axis=axis)
    norm = np.maximum(norm, eps)
    if axis == 0:
        return a / norm[None]
    elif axis == 1:
        return a / norm[:, None]
    else:
        raise ValueError('[ERROR] axis should be either 0 or 1!')


def integrate_audio_feat(features, audio_h5, mxm2msd):
    """
    """
    # TODO: this part should be moved to MFCC feature extraction
    #       and stored in the feature file for better integrity
    n_coeffs = 40
    audio_feat_cols = (
        ['mean_mfcc{:d}'.format(i) for i in range(n_coeffs)] +
        ['var_mfcc{:d}'.format(i) for i in range(n_coeffs)] +
        ['mean_dmfcc{:d}'.format(i) for i in range(n_coeffs)] +
        ['var_dmfcc{:d}'.format(i) for i in range(n_coeffs)] +
        ['mean_ddmfcc{:d}'.format(i) for i in range(n_coeffs)] +
        ['var_ddmfcc{:d}'.format(i) for i in range(n_coeffs)]
    )
    with h5py.File(audio_h5, 'r') as hf:
        tid2row = {tid:i for i, tid in enumerate(hf['tids'][:])}
        feats = []
        for mxmid in corpus.ids:
            tid = mxm2msd[mxmid]
            if tid in tid2row:
                feats.append(hf['feature'][tid2row[tid]][None])
            else:
                feats.append(np.zeros((1, len(audio_feat_cols))))
        audio_feat = np.concatenate(feats, axis=0)
        # idx = [tid2row[mxm2msd[mxmid]] for mxmid in corpus.ids]
        # audio_feat = hf['feature'][idx]
        features['audio'] = TextFeature(
            'mfcc', corpus.ids, audio_feat, audio_feat_cols
        )

    return features
