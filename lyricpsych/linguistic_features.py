from itertools import chain
import numpy as np
from .utils import SW as stop_words


def words_sanity_check(ling_feat):
    def wrapper(words, *args, **kwargs):
        # series of sanity checks
        if len(words) == 0:
            # raise Exception('[ERROR] No lyrics found!')
            return None
        else:
            return ling_feat(words, *args, **kwargs)
    return wrapper


@words_sanity_check
def _num_words(words):
    """ Count number of words per songs
    """
    return len(words)


@words_sanity_check
def _num_rep_phrase(words, phrase_dict):
    """ count appearance of phrase give in the phrase dict
    """
    pass


@words_sanity_check
def _num_unique_words(words):
    """ Count unique number of words per song
    """
    return len(set(words))


@words_sanity_check
def _num_stop_words(words):
    """ Count the number of stop words included
    """
    return len([w for w in set(words) if w in stop_words])


@words_sanity_check
def _num_extreme_words(words, extreme_words, average=True):
    """ Count the number of common words
    
    Inputs:
        words (list of string): to be checked
        extreme_words (set of string or dict[string] -> float): common words set
        
    Returns:
        tuple or list of int: # of extreme words in each extreme polars
    """
    if not isinstance(extreme_words, (dict, set)):
        raise Exception('[ERROR] common/rare word list should be set!')
    elif isinstance(extreme_words, list):
        extreme_words = set(extreme_words)
         
    if not len(extreme_words) > 0:
        raise Exception('[ERROR] no words found!!')
         
    res = 0
    for word in words:
        if word in extreme_words:
            res += 1
            
    if average:
        res /= len(extreme_words)
        
    return res


def get_extreme_words(df, thresh=[2, .95]):
    """ Extract extreme words in each polars
    
    Inputs:
        idf (dict[string] -> float): contains words and
                                     their document raw frequency (in count)
        thresh ([int, float]): threshold to determine extreme words.
                               the first element is the threshold for rare words.
                               words appeared less than this number are treated as rare
                               the second element is the threshold for common words.
                               words appeared more than this ratio (to the entire corpus)
                               considered as the common word.
        
    Returns:
        tuple of string: extreme words.
    """
    df_arr = np.array(list(df.values()))
    com_thrs = np.percentile(df_arr, thresh[1] * 100)
    
    rar_words = set(w for w, freq in df.items() if freq < thresh[0])
    com_words = set(w for w, freq in df.items() if freq > com_thrs)
    return rar_words, com_words


def get_document_frequency(corpus):
    """ Get document frequency from given corpus
    
    Inputs:
        texts (list of list of string): corpus
        
    Returns:
        dict[string] -> float: list of vocabulary and their document frequency
    """
    unique_words = set(chain.from_iterable(corpus))
    df = dict.fromkeys(unique_words, 0)
    for words in corpus:
        for word in words:
            df[word] += 1
    return df