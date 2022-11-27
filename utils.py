import datetime
import pandas as pd
import snscrape.modules.twitter as sntwitter
import streamlit as st

@st.cache(allow_output_mutation=True)
def get_tweets(user_name, start_date, end_date, limit = 500):
    """ Get tweets using the library."""

    tweets = []
    query = f"(from:{user_name}) until:{end_date} since:{start_date}"

    for tweet in sntwitter.TwitterSearchScraper(query).get_items():

        if len(tweets) == limit:
            break

        tweets.append([tweet.date, tweet.user.username, tweet.rawContent, tweet.likeCount])

    return pd.DataFrame(tweets, columns = ["Date", "User", "Tweet", "Likes"])
