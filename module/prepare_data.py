"""
Tools to load and preprocess data
"""
import re
import logging
from string import punctuation

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import RussianStemmer
from sklearn.base import TransformerMixin

logger = logging.getLogger(__name__)
STEMMER = RussianStemmer(False)
STOPWORDS = stopwords.words("russian")


def load_dataset(path):
    logger.info('Loading dataset from %s.', path)
    return pd.read_csv(path)


def nltk_preprocess_text(
        text,
        stemmer,
        stopwords,
        tokenize_type='nltk'
):
    if tokenize_type == 'nltk':
        tokens = [word for sent in nltk.sent_tokenize(text)
                  for word in nltk.word_tokenize(sent)]
    else:
        tokens = [word for sent in nltk.sent_tokenize(text)
                  for word in re.split(r'[\W\d_]+', sent)]

    tokens = [stemmer.stem(token) for token in tokens
              if token not in stopwords
              and token != " "
              and token.strip() not in punctuation]

    text = " ".join(tokens)

    return text


class Preprocessor(TransformerMixin):
    def __init__(self,
                 text_features,
                 _stopwords=STOPWORDS,
                 stemmer=STEMMER,
                 tokenizer='nltk',
                 ):
        self.text_feature = text_feature
        self.tokenizer = tokenizer
        self.stopwords = _stopwords
        self.stemmer = stemmer

    def fit(self, *args):
        return self

    def transform(self, data):
        return data[self.text_feature].apply(lambda x: nltk_preprocess_text(
            x, self.stemmer, self.stopwords, self.tokenizer))
