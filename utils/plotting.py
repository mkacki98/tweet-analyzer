import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px

from PIL import Image
import urllib.request
    
def display_profile_image(profile_image_url, user_name):
    """ Display profile image of the user. """

    urllib.request.urlretrieve(profile_image_url, 'profile_image.jpg')

    return st.image(Image.open('profile_image.jpg'), caption= f"{user_name}", use_column_width = 'never', width = 200)
                    
def plot_sentiment_distribution(tweets_sentiment):
    """ Plot distribution of sentiments for this user. """
    
    fig = ff.create_distplot(
        [tweets_sentiment], group_labels = ["Sentiment score"], bin_size=[0.02])

    st.plotly_chart(fig, use_container_width=True)

def plot_timeseries_barplot(df, column, title):
    """ Plot pre-computed time-series features of the account. """

    fig = px.bar(df.reset_index(), x='day', 
            y=column, 
            template='plotly_dark', 
            title = title)

    st.plotly_chart(fig)