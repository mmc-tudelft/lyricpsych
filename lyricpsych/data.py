from os.path import basename, join
from collections import OrderedDict
import glob
import json

from tqdm import tqdm
import numpy as np
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer

from .files import personality_adj, value_words, liwc_dict, mxm2msd
from .utils import preprocessing, filter_english_plsa


class Corpus:
    def __init__(self, ids, texts, filt_non_eng=True,
                 filter_stopwords=True, filter_thresh=[5, .3]):
        """"""
        self.ids = ids
        self.texts = texts
        self.filt_non_eng = filt_non_eng
        self.filter_stopwords = filter_stopwords
        self.filter_thresh = filter_thresh

        if filt_non_eng:
            self.ids, self.texts = tuple(zip(
                *filter_english_plsa(
                    list(zip(self.ids, self.texts)),
                    preproc=(
                        False if filter_thresh is None
                        else filter_thresh
                    )
                )
            ))
        self._preproc()

    def _preproc(self):
        output = preprocessing(
            self.texts, 'unigram',
            self.filter_thresh, self.filter_stopwords
        )
        self.ngram_corpus = output[0]
        self.corpus = output[1]
        self.id2word = output[2]
        self.doc_term = output[3]


def parse_row(line):
    cells = line.split(',')
    tid = cells[0]
    mxm_tid = cells[1]
    wc = [[int(c) for c in cell.split(':')] for cell in cells[2:]]
    return tid, mxm_tid, wc


def build_mat(lines, verbose=True):
    with tqdm(total=len(lines[18:]), disable=not verbose) as prog:
        tids = {}
        mxm_tids = {}
        rows, cols, vals = [], [], []
        for i, line in enumerate(lines[18:]):
            tid, mxm_tid, wc = parse_row(line)
            col, val = zip(*wc)
            tids[tid] = i
            mxm_tids[mxm_tid] = i
            rows.append(np.full((len(wc),), i))
            cols.append(col)
            vals.append(val)
            prog.update()
    rows, cols, vals = tuple(map(np.concatenate, (rows, cols, vals)))

    X = sp.coo_matrix(
        (vals, (rows, cols - 1)),
        shape=(len(lines[18:]), 5000)
    ).tocsr()
    return X, tids, mxm_tids


def load_mxm_bow(fn, tfidf=True):
    """ Load MusixMatch BOW data matched to MSD

    Inputs:
        fn (string): filename to the MxM BoW

    Returns:
        scipy.sparse.csr_matrix: corpus bow matrix
        list of string: words
        dict: map from MxM tids to MSD tids
        sklearn.feature_extraction.text.TfidfTransformer: tfidf object
    """
    with open(fn) as f:
        lines = [line.strip('\n') for line in f]

    words = lines[17][1:].split(',')
    X, tids, mxm_tids = build_mat(lines)
    if tfidf:
        tfidf_ = TfidfTransformer(sublinear_tf=True)
        X = tfidf_.fit_transform(X)
    else:
        tfidf_ = None

    tid_map = dict(zip(tids, mxm_tids))
    return X, words, tid_map, tfidf_


def load_mxm_lyrics(fn):
    """ Load a MusixMatch api response

    Read API (track_lyrics_get_get) response.

    Inputs:
        fn (str): filename

    Returns:
        list of string: lines of lyrics
        string: musixmatch tid
    """
    d = json.load(open(fn))['message']
    header, body = d['header'], d['body']

    status_code = header['status_code']
    lyrics_text = []
    tid = basename(fn).split('.json')[0]

    if status_code == 200.:
        if body['lyrics']:
            lyrics = body['lyrics']['lyrics_body'].lower()
            if lyrics != '':
                lyrics_text = [
                    l for l in lyrics.split('\n') if l != ''
                ][:-3]

    return tid, ' '.join(lyrics_text)


def load_mxm2msd():
    """ Load the id-map between MxM and MSD

    Inputs:
        fn (str): filename

    Returns:
        dict[str] -> str: MxM to MSD tid
    """
    res = {}
    with open(mxm2msd()) as f:
        for line in f:
            mxm, msd = line.strip().split(',')
            res[mxm] = msd
    return res


def load_lyrics_db(path, fmt='json', verbose=True):
    """ Load loyrics db (crawled) into memory

    Inputs:
        path (string): path where all the api responses are stored
        fmt (string): format of which lyrics are stored
        verbose (bool): indicates whether progress is displayed

    Returns:
        list of tuple: lyrics data
    """
    db = [
        load_mxm_lyrics(fn)
        for fn in tqdm(
            glob.glob(join(path, '*.{}'.format(fmt))),
            disable=not verbose, ncols=80
        )
    ]
    return [(tid, lyrics) for tid, lyrics in db if lyrics != '']


def load_personality_adj():
    """ Load personality adjective from Saucier, Goldbberg 1996

    Returns:
        dict[string] -> list of strings: personality adjectives
    """
    return json.load(open(personality_adj()))


def load_value_words():
    """ Load value words from Wilson et al. 2018

    Returns:
        dict[string] -> list of strings: value words
    """
    return json.load(open(value_words()))


def load_liwc_dict():
    """ Load value LIWC dictionary

    Returns:
        dict[string] -> list of strings: value words
    """
    liwc_fn = liwc_dict()
    if liwc_fn is None:
        return None

    return json.load(open(liwc_fn), object_pairs_hook=OrderedDict)
