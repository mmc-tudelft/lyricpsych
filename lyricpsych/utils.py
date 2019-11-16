import nltk
from itertools import chain
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


def dump_embeddings(fn, embeddings, metadata):
    """ Dump embedding matrix and corresponding metadata to disk
    
    Inputs:
        fn (string): filename where embedding data is saved
        embeddings (numpy.ndarray): embedding matrix
        metadata (list of tuple): metadata
    """
    pass
    