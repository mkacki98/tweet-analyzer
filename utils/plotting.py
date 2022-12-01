import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.express as px

from PIL import Image
import urllib.request

from utils.general import make_space

def plot_correlation(corr):
    """ Plot correlation between features. """

    fig = px.imshow(corr)
    st.write(fig)

def display_profile_polarity(avg_polarity):
    """ Return the judgement on the profile's polarity based on the average score. """
    
    judgement = "neutral"

    if avg_polarity < 0.25:
        judgement = "negative"
    if avg_polarity > 0.75:
        judgement = "positive"

    st.subheader(f"This accounts' tweets are generally `{judgement}`, the average polarity score is `{avg_polarity}`.")


def plot_likes_distribution(df):
    """ Plot distribution of likes per tweet. """

    fig = px.histogram(df, x="likes", marginal="box", title='Distribution of number of likes per tweet.')
    st.plotly_chart(fig, use_container_width=True)

def display_profile_image(profile_image_url, user_name):
    """ Display profile image of the user. """

    urllib.request.urlretrieve(profile_image_url, 'profile_image.jpg')

    return st.image(Image.open('profile_image.jpg'), caption= f"{user_name}", use_column_width = 'never', width = 200)
                    
def plot_polarity_distribution(tweets_polarity):
    """ Plot distribution of polarity scores for this user. """
    
    fig = ff.create_distplot(
        [tweets_polarity], group_labels = ["Polarity score"], bin_size=[0.02])

    st.plotly_chart(fig, use_container_width=True)

def plot_timeseries_barplot(df, column, title):
    """ Plot pre-computed time-series features of the account. """

    fig = px.bar(df.reset_index(), x='day', 
            y=column, 
            template='plotly_dark', 
            title = title)

    st.plotly_chart(fig)