""" General helper module with useful functions. """

import pandas as pd
import snscrape.modules.twitter as sntwitter
import streamlit as st

from datetime import datetime, timedelta


def get_start_date(end_date, days_offset=7):
    """
    Given an end_date and number of days to offset,
    return the string of a start_date.

    Parameters
    ----------
    end_date : str
        end date for the analysis
    days_offset : int, optional
        used to set the start date given end date, by default 7

    Returns
    -------
    str
        start date of the analysis
    """

    start_date = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=days_offset)
    return datetime.strftime(start_date, "%Y-%m-%d")


def remove_spaces(input_string):
    """Remove `\n` in strings to avoid formatting problems."""
    return input_string.replace("\n", " ")


def get_virality_score(x, _min, _max):
    """Normalise average number of likes per tweet to get a [0,1] score."""
    return (x - _min) / (_max - _min)


def compute_features_to_plot(df):
    """
    Compute features for time series barplots.

    Returns
    -------
    pd.DataFrame
        dataframe with computed features, `tweet_count` and `virality_score`
    """

    df["day"] = df.date.apply(lambda x: x.day)
    df = df.groupby(by="day").agg(lambda x: x.to_list())

    df["tweet_count"] = df.user.apply(lambda x: len(x))
    df["avg_likes_per_day"] = df.likes.apply(lambda x: sum(x) / len(x))
    df["avg_likes_per_tweet"] = df.apply(
        lambda x: int(sum(x.likes) / x.tweet_count), axis=1
    )

    min_likes_per_tweet = min(df.avg_likes_per_tweet)
    max_likes_per_tweet = max(df.avg_likes_per_tweet)
    df["virality_score"] = df.apply(
        lambda x: get_virality_score(
            x.avg_likes_per_tweet, min_likes_per_tweet, max_likes_per_tweet
        ),
        axis=1,
    )

    return df[["tweet_count", "virality_score"]]


@st.experimental_memo()
def get_tweets(user_name, start_date, end_date, limit=500):
    """
    Get tweets from the last § using the scraper. Save it up in a
    Streamlit memo.

    Parameters
    ----------
    user_name : str
        Twitter username
    start_date : str
        start date of the analysis
    end_date : str
        end date of the analysis
    limit : int, optional
        limit of tweets to be extracted, can cause problems if it's unbounded,
        by default 500

    Returns
    -------
    pd.DataFrame
        dataframe containing tweets' content and information (e.g. likes or quotes number)
    """

    query = f"(from:{user_name}) until:{end_date} since:{start_date}"

    tweet_generator = sntwitter.TwitterSearchScraper(query).get_items()

    tweets = [[tweet for tweet in tweet_generator] for _ in range(limit)]
    tweets = [item for sublist in tweets for item in sublist]

    tweets_data = [
        [
            tweet.date,
            tweet.user.username,
            tweet.content,
            tweet.likeCount,
            tweet.retweetCount,
            tweet.quoteCount,
        ]
        for tweet in tweets
    ]

    return pd.DataFrame(
        tweets_data, columns=["date", "user", "tweet", "likes", "retweets", "quotes"]
    )


@st.experimental_memo()
def get_user_info(user_name):
    """
    Get information by scraping user's profile.

    Parameters
    ----------
    user_name : str
        Twitter username

    Returns
    -------
    list
        image URL and the number of followers
    """

    user_generator = sntwitter.TwitterProfileScraper(user_name).get_items()
    user = list(next(user_generator) for _ in range(1))[0].user

    return [user.profileImageUrl, user.followersCount]


def check_format(date):
    """
    Check if the entered format of a date is valid.

    Parameters
    ----------
    date : str
        date (should be in %Y-%m-%d format)

    Returns
    -------
    bool
    """

    try:
        date = datetime.strptime(date, "%Y-%m-%d")
        if datetime.today() < date:
            return False
        return True
    except ValueError:
        return False
