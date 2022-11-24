import datetime
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

