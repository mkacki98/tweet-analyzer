import pandas as pd
import snscrape.modules.twitter as sntwitter
import streamlit as st

from datetime import datetime, timedelta

@st.experimental_memo()
def get_tweets(user_name, end_date, limit = 500):
    """ Get tweets from the last § using the package."""

    tweets = []

    start_date = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days = 28)
    start_date = datetime.strftime(start_date, "%Y-%m-%d")

    query = f"(from:{user_name}) until:{end_date} since:{start_date}"

    for tweet in sntwitter.TwitterSearchScraper(query).get_items():

        if len(tweets) == limit:
            break

        tweets.append([tweet.date, tweet.user.username, tweet.rawContent, tweet.likeCount])

    return pd.DataFrame(tweets, columns = ["Date", "User", "Tweet", "Likes"])

def check_format(date):
    """ Check if the format is as desired. """

    try:
        date = datetime.strptime(date, "%Y-%m-%d")
        if datetime.today() < date:
            return False
        return True
    except ValueError:
        return False