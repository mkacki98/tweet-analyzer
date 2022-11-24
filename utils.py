import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import pandas as pd
import snscrape.modules.twitter as sntwitter

def get_tweets(user_name, start_date, end_date, limit = 500):
    """ Get tweets using the library."""

    tweets = []
    query = f"(from:{user_name}) until:{end_date} since:{start_date}"

    for tweet in sntwitter.TwitterSearchScraper(query).get_items():

        if len(tweets) == limit:
            break

        tweets.append([tweet.date, tweet.user.username, tweet.rawContent, tweet.likeCount])

    df = pd.DataFrame(tweets, columns = ["Date", "User", "Tweet", "Likes"])
    df.to_csv(f"tweets/{user_name}.csv")

def process_dates_to_weeks(df):
    """ Change strings of datetime to datetime objects in a df. """

    format = "%Y-%m-%d %H:%M:%S"
    df['Day'] = df.Date.apply(lambda x: datetime.datetime.strptime(x[:-6], format).day)

    return df

def plot_day_tweets_count(df):
    """ Plot number of tweets per day. """

    df = process_dates_to_weeks(df)
    df = df.groupby(by='Day').agg(lambda x: x.to_list()).reset_index()
    df['Count'] = df.Date.apply(lambda x: len(x))
    df = df.sort_values(by='Day')

    fig = plt.figure(figsize=(10, 4))
    sns.barplot(data = df, x = 'Day', y = 'Count')
    plt.title("Number of Tweets published in January.")
    
    return fig

def plot_day_likes_count(df):
    """ Plot number of likes the tweets get per day. """

    df = process_dates_to_weeks(df)
    df = df.groupby(by='Day').agg(lambda x: x.to_list()).reset_index()
    df['Count'] = df.Likes.apply(lambda x: sum(x))
    df = df.sort_values(by='Day')

    fig = plt.figure(figsize=(10, 4))
    sns.barplot(data = df, x = 'Day', y = 'Count')
    plt.title("Number of likes tweets received in January.")
    
    return fig