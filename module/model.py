"""
Tools to deal with model
"""
import pickle
import gzip
import logging

from catboost import CatBoostClassifier

from .prepare_data import Preprocessor
from sklearn.pipeline import Pipeline


logger = logging.getLogger(__name__)
TEXT_PROCESSING = {
    "tokenizers": [{
        "tokenizer_id": "Space",
        "separator_type": 'ByDelimiter',
        "delimiter": " ",
        "lowercasing": "true",
        "sub_tokens_policy": 'SingleToken',
    },{
        "tokenizer_id": "Sense",
        "separator_type": 'BySense',
        "lowercasing": "true",
        "sub_tokens_policy": 'SingleToken',
    }],
    "dictionaries": [{
            "dictionary_id": "Word",
            "gram_order": "1",
            "max_dictionary_size": "50000",
            "occurrence_lower_bound": "3",
        }, {
            "dictionary_id": "BiGram",
            "gram_order": "2",
            "max_dictionary_size": "50000",
            "occurrence_lower_bound": "3",
        }, {
            "dictionary_id": "TriGram",
            "gram_order": "3",
            "max_dictionary_size": "50000",
            "occurrence_lower_bound": "3",
        }, {
            "dictionary_id": "Symbols_1",
            "token_level_type": "Letter",
            "gram_order": "1",
            "max_dictionary_size": "500",
            "occurrence_lower_bound": "1",
        }, {
            "dictionary_id": "Symbols_2",
            "token_level_type": "Letter",
            "gram_order": "2",
            "max_dictionary_size": "2500",
            "occurrence_lower_bound": "2",
        }, {
            "dictionary_id": "Symbols_3",
            "token_level_type": "Letter",
            "gram_order": "3",
            "max_dictionary_size": "10000",
            "occurrence_lower_bound": "10",
        }],
    "feature_processing": {
        "default": [{
                "dictionaries_names": [
                    "Symbols_1",
                    "Symbols_2",
                    "Symbols_3",
                    'Word',
                    "BiGram",
                    "TriGram",
                ],
                "feature_calcers": ["BoW"],
                "tokenizers_names": ["Space"]
            }, {
                "dictionaries_names": ["Symbols_1", 'Word'],
                "feature_calcers": ["NaiveBayes"],
                "tokenizers_names": ["Space"]
        }],
    }
}
CB_PARAMS = {
    'text_features': ['text'],
    'text_processing': TEXT_PROCESSING,
    'iterations': 100,
}


def get_cb_pipeline(catboost_args):
    return Pipeline([
        ('preproc', Preprocessor('text')),
        ('clf', CatBoostClassifier(**catboost_args))
    ])


def save_pipeline(pipeline, filename):
    logger.info('Model is saving as %s', filename)
    with gzip.open(filename, "wb") as file:
        pickle.dump(pipeline, file)


def load_pipeline(filename):
    logger.info('Model is loading from %s', filename)
    with gzip.open(filename, "rb") as file:
        return pickle.load(file)