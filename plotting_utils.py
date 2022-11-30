import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px

def plot_sentiment_distribution(tweets_sentiment):
    """ Plot distribution of sentiments for this user. """
    
    fig = ff.create_distplot(
        [tweets_sentiment], group_labels = ["Sentiment score"], bin_size=[0.02])

    st.plotly_chart(fig, use_container_width=True)

def plot_timeseries_barplot(df, column, title):

    fig = px.bar(df.reset_index(), x='day', 
            y=column, 
            template='plotly_dark', 
            title = title)

    st.plotly_chart(fig)