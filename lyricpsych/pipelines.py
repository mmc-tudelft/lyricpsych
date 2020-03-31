from itertools import product, chain
from collections import OrderedDict, Counter
import argparse
import logging

import numpy as np
import pandas as pd
import gensim
import gensim.downloader
import h5py

from scipy.spatial.distance import cdist
from tqdm import tqdm

from .topic_model import PLSA
from .data import (load_personality_adj,
                   load_value_words,
                   load_liwc_dict,
                   load_lyrics_db,
                   load_mxm2msd,
                   Corpus)
from .utils import (represent_ngram,
                    preprocessing,
                    _process_word,
                    save_feature_h5)
from .linguistic_features import (get_document_frequency,
                                  get_extreme_words,
                                  _num_words,
                                  _num_rep_phrase,
                                  _num_unique_words,
                                  _num_stop_words,
                                  _num_extreme_words)
from .feature import TextFeature, TopicFeature


class TextFeatureExtractor:
    def __init__(self, gensim_w2v=None, compute_similarity=False):
        """
        """
        # this can take minutes if the model if big one
        if gensim_w2v is None:
            self.w2v = None
        else:
            self.w2v = gensim.downloader.load(gensim_w2v)

        self.psych_inventories = {
            'personality': load_personality_adj(),
            'value': load_value_words()
        }
        self._init_liwc()
        self.compute_similarity = compute_similarity

    @staticmethod
    def _build_liwc_trie(liwc_dict):
        """
        Build a character-trie from the plain string value -> key map

        it is adopted from the `https://github.com/chbrown/liwc-python/blob/master/liwc/trie.py

        Inputs:
            dict[str] -> str: word -> cat map

        Outputs:
            dict[char] -> dict: trie for LIWC
        """
        trie = {}
        for word, cat in liwc_dict.items():
            cursor = trie
            for char in word:
                if char == "*":
                    cursor["*"] = cat
                    break
                if char not in cursor:
                    cursor[char] = {}
                cursor = cursor[char]
            cursor["$"] = cat
        return trie

    @staticmethod
    def _search_trie(trie, token, token_i=0):
        """
        Search the given character-trie for paths match the `token` string

        it is adopted from the `https://github.com/chbrown/liwc-python/blob/master/liwc/trie.py
        """
        if "*" in trie:
            return trie["*"]
        if "$" in trie and token_i == len(token):
            return trie["$"]
        if token_i < len(token):
            char = token[token_i]
            if char in trie:
                return (
                    TextFeatureExtractor
                    ._search_trie(trie[char], token, token_i+1)
                )
        return []

    @property
    def is_liwc(self):
        return hasattr(self, '_liwc_trie')

    def _init_liwc(self):
        """"""
        self.liwc_dict = {}
        self._raw_liwc_dict = load_liwc_dict()
        if self._raw_liwc_dict is None:
            logging.warning('No LIWC dictionary found.'
                            'LIWC feature is not extracted')
        else:
            self._liwc_cat_map = {k:i for i, k
                                  in enumerate(self._raw_liwc_dict.keys())}
            for cat, words in self._raw_liwc_dict.items():
                for word in words:
                    if word not in self.liwc_dict:
                        self.liwc_dict[word] = []
                    self.liwc_dict[word].append(cat)

            # learn the trie
            self._liwc_trie = (
                TextFeatureExtractor
                ._build_liwc_trie(self.liwc_dict)
            )

    def _inventory_scores(self, corpus, inventory):
        """
        """
        # pre-process inventories
        scores = {}
        for cat, words in inventory.items():
            scores[cat] = [_process_word(w) for w in words]

        feature, cols = _compute_coherence(
            self.w2v, corpus.ngram_corpus, scores,
            compute_similarity=self.compute_similarity
        )
        return feature, cols

    def psych_scores(self, corpus):
        """
        """
        scores = {}
        for key, adjs in self.psych_inventories.items():
            s, c = self._inventory_scores(corpus, adjs)
            feat = TextFeature(key, corpus.ids, s, c)
            scores[key] = feat
        return scores

    def liwc(self, corpus):
        """
        TODO: deal with special cases (i.e. `(i) like*`)
        """
        if hasattr(self, '_liwc_trie'):
            feats = []
            for words in corpus.ngram_corpus:
                # extract liwc registers
                cnt = Counter(chain.from_iterable([
                    TextFeatureExtractor._search_trie(self._liwc_trie, word)
                    for word in words
                ]))
                feats.append(dict(cnt))

            # convert to the dataframe -> text feature
            feats = pd.DataFrame(feats).fillna(0.)
            feats = TextFeature('LIWC', corpus.ids, feats.values, feats.columns)
            return feats
        else:
            return None

    def linguistic_features(self, corpus, verbose=False):
        """
        """
        feature = _compute_linguistic_features(corpus.ngram_corpus, verbose)
        return TextFeature(
            'linguistic', corpus.ids, feature.values, feature.columns
        )

    def topic_distributions(self, corpus, k=25):
        """
        """
        # run plsa
        plsa = PLSA(k, 30)
        plsa.fit(corpus.doc_term)

        return TopicFeature(
            k, corpus.ids, plsa.doc_topic, plsa.topic_term,
            corpus.id2word.token2id
        )

#     def get_deep_features(self, corpus):
#         """
#         """
#         pass


def _compute_coherence(word2vec, words_corpus, target_words,
                       compute_similarity=False, show_progress=False):
    """ Compute the coherence score between given texts and the target inventory

    Inputs:
        word2vec (gensim.models.Word2Vec or dict): key-value container that holds
                                                   word-vector association
        word_corpus (list of list of string): corpus to be compared
        target_words (dict[string] -> list of string): target words from each inventory
        compute_similarity (bool): compute 1 - cosine distance
        show_progress (bool): indicates verbosity of the process

    Returns:
        np.ndarray: computed coherence score (#obs, #target_words)
    """
    target_words = OrderedDict(target_words)

    # 1. pre-compute distances for all unique words pairs
    def preproc(corpus):
        uniques = list(set(
            w for w in chain.from_iterable(corpus)
            if w in word2vec
        ))
        id_map = {w:i for i, w in enumerate(uniques)}
        embs = np.array([word2vec[w] for w in uniques])
        return uniques, id_map, embs

    unique_words, words_hash, uniq_wv = preproc(words_corpus)
    unique_targs, targs_hash, uniq_tv = preproc(target_words.values())
    X = cdist(uniq_wv, uniq_tv, metric='cosine')

    # 2. pre-aggregate scores by their categories
    S = np.zeros((X.shape[0], len(target_words)))
    cats = []  # categories
    for cat, targs in target_words.items():
        targ_ids = [targs_hash[t] for t in targs if t in targs_hash]
        S[:, len(cats)] = X[:, targ_ids].mean(1)
        cats.append(cat)

    # 3. compute scores per docs and all the target words
    coherences = np.zeros((len(words_corpus), len(cats)))
    with tqdm(total=len(words_corpus),
              disable=not show_progress, ncols=80) as p:
        for i, words in enumerate(words_corpus):
            w_i = [words_hash[w] for w in words if w in words_hash]
            if len(w_i) > 0:
                coherences[i] = S[w_i].mean(0)
            p.update(1)

    # if specified, get the similarity rather than distance
    if compute_similarity:
        coherences = 1 - coherences

    return coherences, cats


def _compute_linguistic_features(words_corpus, show_progress=True,
                                 extreme_thresh=[2, 0.75]):
    """ Compute all the linguistic features

    Inputs:
        words_corpus (list of list of string):

    """
    # pre-compute some entities
    doc_freq = get_document_frequency(words_corpus)
    rare_words, common_words = get_extreme_words(doc_freq, extreme_thresh)

    # compute all features
    feats = []
    with tqdm(total=len(words_corpus), ncols=80,
              disable=not show_progress) as p:

        for i, words in enumerate(words_corpus):
            feat = {}
            feat['num_words'] = N = _num_words(words)
            # feat['num_rep_phrase'] = _num_rep_phrase(words, phrase_dict)
            feat['num_unique_words'] = _num_unique_words(words)
            feat['num_stop_words'] = _num_stop_words(words)
            feat['num_rare_words'] = _num_extreme_words(words, rare_words)
            feat['num_common_words'] = _num_extreme_words(words, common_words)
            if N is not None:
                feat['ratio_unique_words'] = feat['num_unique_words'] / N
                feat['ratio_stop_words'] = feat['num_stop_words'] / N
                feat['ratio_rare_words'] = feat['num_rare_words'] / N
                feat['ratio_common_words'] = feat['num_common_words'] / N
            else:
                feat['ratio_unique_words'] = None
                feat['ratio_stop_words'] = None
                feat['ratio_rare_words'] = None
                feat['ratio_common_words'] = None
            feats.append(feat)
            p.update(1)
    feats = pd.DataFrame(feats)
    return feats


def extract(corpus, extractor, filt_non_eng=True,
            filter_thresh=[5, .3], num_topics=25,
            features={
                'liwc', 'linguistic', 'personality',
                'value', 'topic'
            }):
    """ Extract text related features from lyrics database

    Inputs:
        corpus (lyricpsych.data.Corpus): data contains all the texts
        extractor (lyricpsych.pipelines.TextFeatureExtractor): lyrics feature extractor
        filt_non_eng (bool): indicates whether filter non-English lyrics
        filter_thresh (list of float): filtering threshold for rare / popular words
        num_topics (int): number of topics to be extracted
        config (set of string): extraction configuration

    Returns:
        dict[string] -> TextFeature: extracted text feature
    """
    # extract the feature
    output = {}

    if 'liwc' in features and extractor.is_liwc:
        output['liwc'] = extractor.liwc(corpus)

    if 'linguistic' in features:
        output['linguistic'] = extractor.linguistic_features(corpus)

    # redo the preprocessing with the thresholds
    corpus.filter_thresh = filter_thresh
    corpus._preproc()

    if extractor.w2v is None:
        logging.warning('No word2vec model is specified.'
                        ' inventory estimation is not computed')
    else:
        for key in ['personality', 'value', 'inventory']:
            if key in features:
                s, c = extractor._inventory_scores(
                    corpus, extractor.psych_inventories[key]
                )
                output[key] = TextFeature(key, corpus.ids, s, c)

    if 'topic' in features:
        output['topic'] = extractor.topic_distributions(corpus, k=num_topics)

    return output


def integrate_audio_feat(features, audio_h5):
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


if __name__ == "__main__":
    # setup argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('mxm_dir', type=str,
                        help='path where all lyrics are stored')
    parser.add_argument('w2v', type=str,
                        help='codename to the word2vec pretrained model (for gensim)')
    parser.add_argument('out_fn', type=str,
                        help='path to dump processed h5 file')
    parser.add_argument('--audio-h5', default=None,
                        help='filename of the audio feature `hdf` file')
    # in this case, we filter out non-english entries.
    parser.add_argument('--eng-filt', dest='is_eng', action='store_true')
    # which is not in this case.
    parser.add_argument('--no-eng-filt', dest='is_eng', action='store_false')

    parser.set_defaults(is_eng=True)
    args = parser.parse_args()

    # 1. loading the data
    mxm2msd = load_mxm2msd()
    data = load_lyrics_db(args.mxm_dir)
    ids, texts = tuple(zip(*data))
    corpus = Corpus(ids, texts, filter_thresh=None,
                    filt_non_eng=filt_non_eng)

    # 2. initiate the extractor
    extractor = TextFeatureExtractor(args.w2v)

    # 3. extract lyrics features
    features = extract(corpus, extractor, args.audio_h5, args.is_eng)

    # 4. integrate audio feature
    if exists(args.audio_h5):
        features = integrate_audio_feat(features, args.audio_h5)

    # 5. save to the .h5 file
    save_feature_h5(features, args.out_fn)
