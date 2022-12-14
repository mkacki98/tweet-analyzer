""" A helper module with language processing functions. """

import spacy

from transformers import pipeline

nlp = spacy.load("en_core_web_sm")
classifier = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")


def get_clean_tweets(docs):
    """
    Clean Tweets from URLs/mentiones.

    Parameters
    ----------
    docs : SpaCy's Doc Object
        all the tokens passed through SpaCy pipeline that contains their features
        (PoS etc.)

    Returns
    -------
    list of list
        each list contains nouns extracted from a tweet as an element
    """

    tokens_filtered = [
        [token.text for token in tweet if filter_tweets(token)] for tweet in docs
    ]
    return [" ".join(tokens) for tokens in tokens_filtered]


def get_nouns(docs):
    """Get nouns and proper nouns from all the Tweets for visualization purposes."""

    return [
        [token.text for token in tweet if filter_non_nouns(token)] for tweet in docs
    ]


def get_spectrum_scores(model_output):
    """Get polarity scores in [-1, 1] range."""

    if model_output["label"] == "POSITIVE":
        return -1 * model_output["score"]
    if model_output["label"] == "NEGATIVE":
        return model_output["score"]
    return 0


def get_polarity_scores(docs):
    """Return a list of floats where each float corresponds to a polarity of a tweet."""

    scores = [get_spectrum_scores(classifier(tweet)[0]) for tweet in docs]

    max_score = max(scores)
    min_score = min(scores)

    return [(score - min_score) / (max_score - min_score) for score in scores]


def filter_tweets(token):
    """Return False is a token is an URl or a mention."""

    if token.like_url == True:
        return False
    if "@" in token.text:
        return False
    return True


def filter_non_nouns(token):
    """Return False if a token is not a noun or a proper noun, is a mention or a link."""

    if token.pos_ not in ["NOUN", "PROPN"]:
        return False
    if token.like_url:
        return False
    if "@" in token.text:
        return False

    return True
