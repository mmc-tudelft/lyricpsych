from collections import Counter
import re

import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

STOPWORDS = set(stopwords.words('english'))


class NaiveText2Vec:
    def __init__(self, w2v, tfidf=None):
        self.w2v = w2v
        if isinstance(tfidf, TfidfVectorizer) or tfidf is None:
            self.tfidf = tfidf
        else:
            raise Exception(
                '[ERROR] tfidf should be instance of ' +
                'sklearn.feature_extraction.text.TfidfVectorizer or' +
                'None.'
            )
    
    def _compute_lyrics_vector(self, lyrics, tfidf=True):
        # make the alphabets lower
        line = ' '.join(lyrics).lower()
        
        # remove non-alphabetic chars
        line = re.sub(
            "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t:])|(\w+:\/\/\S+)",
            " ", line
        )
        
        # tokenization and filtering stop-words
        words = [
            word for word
            in nltk.word_tokenize(line)
            if word not in STOPWORDS
        ]
        word_count = Counter(words)
        
        count = 0
        d = self.w2v.vectors.shape[-1]  # dimensionality of w2v
        vector = np.zeros((d,))
        for word, freq in word_count.items():
            if word in self.w2v:
                vec = self.w2v[word]
                weight = 1 
                if tfidf and (word in self.tfidf.vocabulary_):
                    word_idx = self.tfidf.vocabulary_[word]
                    weight = self.tfidf.idf_[word_idx] 
                vector += vec * weight * freq
                count += 1
                
        # dealing with conner case (no words found in w2v)
        if count == 0:
            vector = np.random.randn(d)
        else:
            vector /= count
            
        return vector

    def get_vectors(self, all_lyrics, tfidf=True, verbose=True):
        self._fit_tfidf(all_lyrics)
        
        shape = (len(all_lyrics), self.w2v.vectors.shape[-1])
        vectors = np.zeros(shape, dtype=self.w2v.vectors.dtype)
        
        with tqdm(total=len(all_lyrics), ncols=80, disable=not verbose) as prog: 
            for i, lyrics in enumerate(all_lyrics):
                vectors[i] = self._compute_lyrics_vector(lyrics, tfidf)
                prog.update(1)
            
        return vectors
        
    def _fit_tfidf(self, all_lyrics):
        if self.tfidf is None:
            self.tfidf = TfidfVectorizer(stop_words=STOPWORDS)
            self.tfidf.fit([r[1] for r in all_lyrics])