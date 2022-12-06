import pandas as pd
import snscrape.modules.twitter as sntwitter
import streamlit as st

from datetime import datetime, timedelta

def get_start_date(end_date, days_offset = 7):
    """ Given an end_date and number of days to offse, return the string of a start_date. """
    
    start_date = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days = days_offset)
    return datetime.strftime(start_date, "%Y-%m-%d")

def make_space(n):
    """ Create blank space between lines in the app. """
    for i in range(n):
        st.write("\n")

def remove_spaces(input_string):
    """ Remove \n in strings to avoid formatting problems. """
    return input_string.replace("\n", " ")
    
def get_virality_score(x, _min, _max):
    """ Normalise average number of likes per tweet to get a [0,1] score. """
    return (x - _min)/(_max - _min)
    
def compute_features_to_plot(df):
    """ Compute features for time series barplots. """

    df['day'] = df.date.apply(lambda x: x.day)
    df = df.groupby(by = 'day').agg(lambda x: x.to_list())

    df['tweet_count'] = df.user.apply(lambda x: len(x))
    df['avg_likes_per_day'] = df.likes.apply(lambda x: sum(x)/len(x))
    df['avg_likes_per_tweet'] = df.apply(lambda x: int(sum(x.likes)/x.tweet_count), axis = 1)

    min_likes_per_tweet = min(df.avg_likes_per_tweet)
    max_likes_per_tweet = max(df.avg_likes_per_tweet)
    df['virality_score'] = df.apply(lambda x: get_virality_score(x.avg_likes_per_tweet, min_likes_per_tweet, max_likes_per_tweet), axis = 1)

    return df[['tweet_count', 'virality_score']]

@st.experimental_memo()
def get_tweets(user_name, start_date, end_date, limit = 500):
    """ Get tweets from the last § using the package."""

    query = f"(from:{user_name}) until:{end_date} since:{start_date}"

    tweet_generator = sntwitter.TwitterSearchScraper(query).get_items()

    tweets = [[tweet for tweet in tweet_generator] for _ in range(limit)]
    tweets = [item for sublist in tweets for item in sublist]

    tweets_data = [[tweet.date, tweet.user.username, tweet.content, tweet.likeCount, tweet.retweetCount, tweet.quoteCount] for tweet in tweets]
    
    return pd.DataFrame(tweets_data, columns = ["date", "user", "tweet", "likes", "retweets", "quotes"])

@st.experimental_memo()
def get_user_info(user_name):
    """ Get information from user profile. """
    
    user_generator = sntwitter.TwitterProfileScraper(user_name).get_items()
    user = list(next(user_generator) for _ in range(1))[0].user

    return [user.profileImageUrl, user.followersCount]


def check_format(date):
    """ Check if the format is as desired. """

    try:
        date = datetime.strptime(date, "%Y-%m-%d")
        if datetime.today() < date:
            return False
        return True
    except ValueError:
        return False