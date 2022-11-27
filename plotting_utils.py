import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.figure_factory as ff

def plot_sentiment_distribution(tweets_sentiment):
    """ Plot distribution of sentiments for this user. """
    
    fig = ff.create_distplot(
        [tweets_sentiment], group_labels = ["Sentiment score"], bin_size=[0.02])

    st.plotly_chart(fig, use_container_width=True)
    
def plot_day_tweets_count(df):
    """ Plot number of tweets per day. """

    df['Day'] = df.Date.apply(lambda x: x.day)
    df = df.groupby(by='Day').agg(lambda x: x.to_list()).reset_index()
    df['Count'] = df.Date.apply(lambda x: len(x))
    df = df.sort_values(by='Day')

    fig = plt.figure(figsize=(10, 4))
    sns.barplot(data = df, x = 'Day', y = 'Count')
    plt.title("Number of Tweets published in January.")
    st.pyplot(fig)

def plot_day_likes_count(df):
    """ Plot number of likes the tweets get per day. """

    df['Day'] = df.Date.apply(lambda x: x.day)
    df = df.groupby(by='Day').agg(lambda x: x.to_list()).reset_index()
    df['Count'] = df.Likes.apply(lambda x: sum(x))
    df = df.sort_values(by='Day')

    fig = plt.figure(figsize=(10, 4))
    sns.barplot(data = df, x = 'Day', y = 'Count')
    plt.title("Number of likes tweets received in January.")
    st.pyplot(fig)
    