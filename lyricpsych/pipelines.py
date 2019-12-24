from itertools import product, chain
from collections import OrderedDict
import argparse

import numpy as np
import pandas as pd
import gensim
import h5py

from scipy.spatial.distance import cdist
from tqdm import tqdm

from .topic_model import PLSA
from .data import load_personality_adj, load_lyrics_db, Corpus
from .utils import represent_ngram, preprocessing, _process_word
from .linguistic_features import (get_document_frequency,
                                  get_extreme_words,
                                  _num_words,
                                  _num_rep_phrase,
                                  _num_unique_words,
                                  _num_stop_words,
                                  _num_extreme_words)
from .feature import TextFeature, TopicFeature


class TextFeatureExtractor:
    def __init__(self, w2v_fn=None, w2v_type='word2vec'):
        """
        """
        # this can take minutes if the model if big one
        if w2v_fn is None:
            self.w2v = None
        else:
            self.w2v = (
                gensim.models.KeyedVectors
                .load_word2vec_format(
                    w2v_fn,
                    binary=(
                        True
                        if w2v_type == 'word2vec'
                        else False # {'glove', 'word2vec'}
                    )
                )
            )
            
        self.psych_adjs = {
            'personality': load_personality_adj()
        }
    
    def _inventory_scores(self, corpus, inventory):
        """
        """
        # pre-process inventories
        pers_adj = {}
        for cat, words in inventory.items():
            pers_adj[cat] = [_process_word(w) for w in words]
        feature, cols = _compute_coherence(self.w2v, corpus.ngram_corpus, pers_adj)
        return feature, cols
    
    def psych_scores(self, corpus):
        """
        """
        scores = []
        for key, adjs in self.psych_adjs.items():
            s, c = self._inventory_scores(corpus, adjs)
            feat = TextFeautre(key, corpus.ids, s, c)
            scores.append(feat)
        return scores

    def linguistic_features(self, corpus,
                            artist_song_map=None,
                            verbose=False):
        """
        """
        feature = _compute_linguistic_features(
            corpus.ngram_corpus, artist_song_map, verbose
        )
        
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
                       show_progress=False):
    """ Compute the coherence score between given texts and the target inventory
    
    Inputs:
        word2vec (gensim.models.Word2Vec or dict): key-value container that holds
                                                   word-vector association
        word_corpus (list of list of string): corpus to be compared
        target_words (dict[string] -> list of string): target words from each inventory
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

    return coherences, cats


def _compute_linguistic_features(words_corpus, artists=None,
                                 show_progress=True, extreme_thresh=[2, 0.75]):
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
    
    # if map is given, aggregate (mean) the result by the artist
    if artists is not None:
        feats['artist'] = artists
        return feats.dropna(axis=0).groupby('artist').mean()
    else:
        return feats
    
    
def extract(mxm_dir, w2v_fn=None,
            filter_thresh=[5, .3], num_topics=25,
            config={
                'liwc': False, 'linguistic': True,
                'personality': True, 'value': False,
                'topic': True, 'deep': False
            }):
    """ Extract text related features from lyrics database
    
    Inputs:
        mxm_dir (string): path where all lyrics data stored
        w2v_fn (string or None): path to the word2vec model
        config (dict[string] -> bool): extraction configuration 
    
    Returns:
        dict[string] -> TextFeature: extracted text feature
    """
    # 1. loading the data
    data = load_lyrics_db(mxm_dir)
    ids, texts = tuple(zip(*data))
    corpus = Corpus(ids, texts, filter_thresh=None)
    
    # 2. initiate the extractor
    extractor = TextFeatureExtractor(w2v_fn)
    
    # 3. extract the feature
    features = {}
    
    if config['liwc']:
        # features['liwc'] = 
        pass
    
    if config['linguistic']:
        features['linguistic'] = extractor.linguistic_features(corpus)
    
    # redo the preprocessing with the thresholds
    corpus.filter_thresh = filter_thresh
    corpus._preproc()
    
    if config['personality']:
        pers_feat, pers_cols = extractor._inventory_scores(
            corpus, extractor.psych_adjs['personality']
        )
        features['personality'] = TextFeature(
            'personality', corpus.ids, pers_feat, pers_cols
        )
    
    if config['value']:
        val_feat, val_cols = extractor._inventory_scores(
            corpus, extractor.psych_adjs['value']
        )
        features['value'] = TextFeature(
            'value', corpus.ids, pers_feat, pers_cols
        )
    
    if config['topic']:
        features['topic'] = extractor.topic_distributions(corpus, k=num_topics)
        
    if config['deep']:
        pass
    
    return features
    
    
def save(features, out_fn):
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
        
        
if __name__ == "__main__":
    # setup argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('mxm_dir', type=str,
                        help='path where all lyrics are stored')
    parser.add_argument('w2v_fn', type=str,
                        help='filename to the word2vec pretrained model (for gensim)')
    parser.add_argument('out_fn', type=str,
                        help='path to dump processed h5 file')
    args = parser.parse_args()
    
    # run
    save(extract(args.mxm_dir, args.w2v_fn), args.out_fn)