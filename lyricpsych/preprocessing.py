import string
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

from tqdm import tqdm

from .extractors.topicmodel import PLSA


# intantiate lemmatizer / stopwords
ENGLISH_VOCAB = set(w.lower() for w in nltk.corpus.words.words())
SW = set(stopwords.words('english'))
LEMMA = WordNetLemmatizer()


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
