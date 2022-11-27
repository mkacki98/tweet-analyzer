import seaborn as sns
import matplotlib.pyplot as plt

from utils import process_dates_to_weeks

def plot_day_tweets_count(df):
    """ Plot number of tweets per day. """

    df['Day'] = df.Date.apply(lambda x: x.day)
    df = df.groupby(by='Day').agg(lambda x: x.to_list()).reset_index()
    df['Count'] = df.Date.apply(lambda x: len(x))
    df = df.sort_values(by='Day')

    fig = plt.figure(figsize=(10, 4))
    sns.barplot(data = df, x = 'Day', y = 'Count')
    plt.title("Number of Tweets published in January.")
    
    return fig

def plot_day_likes_count(df):
    """ Plot number of likes the tweets get per day. """

    df['Day'] = df.Date.apply(lambda x: x.day)
    df = df.groupby(by='Day').agg(lambda x: x.to_list()).reset_index()
    df['Count'] = df.Likes.apply(lambda x: sum(x))
    df = df.sort_values(by='Day')

    fig = plt.figure(figsize=(10, 4))
    sns.barplot(data = df, x = 'Day', y = 'Count')
    plt.title("Number of likes tweets received in January.")
    return fig
    