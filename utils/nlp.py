import spacy 
from transformers import pipeline

nlp = spacy.load("en_core_web_sm")
classifier = pipeline("sentiment-analysis", model = "roberta-large-mnli", framework = 'pt')

def get_clean_tweets(docs):
    """ Clean Tweets from URLs/mentiones."""

    tokens_filtered = [[token.text for token in tweet if filter_tweets(token)] for tweet in docs]
    return [" ".join(tokens) for tokens in tokens_filtered]

def get_nouns(docs):
    """ Get nouns and proper nouns from all the Tweets for visualization purposes. """

    return [[token.text for token in tweet if filter_non_nouns(token)] for tweet in docs] 

def get_sentiments(docs):
    """ Return a list of floats where each float corresponds to a sentiment of a tweet. """
    return [classifier(tweet)[0]['score'] for tweet in docs]

def filter_tweets(token):
    """ Return False is a token is an URL, a mention or something weird/irrelevant. """

    if token.like_url == True:
        return False
    if '@' in token.text:
        return False
    return True

def filter_non_nouns(token):
    """ Return False if a token is not a noun or a proper noun. """

    if token.pos_ not in ["NOUN", "PROPN"]:
        return False
    return True