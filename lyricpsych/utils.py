from os.path import basename, dirname, splitext, join
from itertools import chain

import numpy as np
import nltk
from tqdm import tqdm


def compute_non_englishness(dataset, verbose=True):
    """ Detects unusual words from lyrics dataset
    """
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusuals = []
    for tid, lyrics in tqdm(dataset, ncols=80, disable=not verbose):
        if lyrics != '':
            text_vocab = set(
                w for w
                in nltk.word_tokenize(lyrics.lower())
                if w.isalpha()
            )
            unusual = text_vocab.difference(english_vocab)
            unusuals.append((tid, len(unusual)))
        else:
            unusuals.append((tid, len(english_vocab)))
    return unusuals


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
    