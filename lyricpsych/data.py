from os.path import basename, join
import glob
import json

from tqdm import tqdm
import numpy as np
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer

from .files import hexaco

INVENTORIES = {
    'hexaco': hexaco()
}


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


def load_lyrics_db(path, fmt='json', verbose=True):
    """ Load loyrics db (crawled) into memory
    
    Inputs:
        path (string): path where all the api responses are stored
        fmt (string): format of which lyrics are stored
        verbose (bool): indicates whether progress is displayed
    
    Returns:
        list of tuple: lyrics data
    """
    return [
        load_mxm_lyrics(fn)
        for fn in tqdm(
            glob.glob(join(path, '*.{}'.format(fmt))),
            disable=not verbose, ncols=80
        )
    ]


def load_inventory(inventory='hexaco'):
    """ Load psych inventory to be used as target
    
    Inputs:
        inventory (string): type of inventory to be loaded {'hexaco', 'value'}
    
    Outputs:
        list of tuple: inventory data
    """
    if inventory not in INVENTORIES:
        raise ValueError('[ERROR] {} is not supported!'.format(inventory))
        
    y = json.load(open(INVENTORIES[inventory]))['inventory']
    return list(y.items()) 