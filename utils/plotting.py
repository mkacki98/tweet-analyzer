""" Helper file for plotting functions and displaying prompts. """

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.express as px
import urllib.request

from wordcloud import WordCloud
from PIL import Image


def plot_nouns_wordcloud(tweets_nouns):
    """Plot a Wordlcloud given nouns extracted from tweets.

    Parameters
    ----------
    tweets_nouns : list of lists
        each list contains nouns extracted from a given tweet
    """

    text = " ".join([item for sublist in tweets_nouns for item in sublist])

    wordcloud = WordCloud().generate(text)
    fig, x = plt.subplots()

    x.imshow(wordcloud, interpolation="bilinear")
    x.axis("off")

    st.pyplot(fig)


def get_correlation_judgement(corr):
    """
    Return the verbal evaluation of correlation between polarity
    and other variables.

    Parameters
    ----------
    corr : float
        correlation score

    Returns
    -------
    str
        subjective judgement of how strong the correlation is
    """

    corr = abs(corr)

    if corr > 0 and corr < 0.2:
        return "weakly"
    if corr >= 0.2 and corr < 0.5:
        return "moderately"
    if corr >= 0.5:
        return "strongly"


def display_correlation_prompts(corr):
    """
    Display information on how strongly correlated polarity
    is with another variables.

    Parameters
    ----------
    corr : pd.DataFrame
        dataframe of correlations between polarity and other variables
    """

    polarity_corr = corr.iloc[0]

    st.subheader(
        f"Number of likes is `{get_correlation_judgement(polarity_corr.likes)}` correlated with polarity. \n"
    )
    st.subheader(
        f"Number of retweets is `{get_correlation_judgement(polarity_corr.retweets)}` correlated with polarity. \n"
    )
    st.subheader(
        f"Number of quotes is `{get_correlation_judgement(polarity_corr.quotes)}` correlated with polarity."
    )


def plot_correlation(corr):
    """Plot correlation between features."""

    fig = px.imshow(corr)
    st.write(fig)


def display_profile_polarity(avg_polarity):
    """Return the judgement on the profile's polarity based on the average score."""

    judgement = "neutral"

    if avg_polarity < 0.25:
        judgement = "very positive"
    if avg_polarity >= 0.25 and avg_polarity < 0.4:
        judgement = "positive"

    if avg_polarity > 0.6 and avg_polarity <= 0.75:
        judgement = "negative"
    if avg_polarity > 0.75:
        judgement = "very negative"

    st.subheader(
        f"This accounts' tweets are generally `{judgement}`, the average polarity score is `{avg_polarity}`."
    )


def plot_likes_distribution(df):
    """Plot distribution of likes per tweet."""

    fig = px.histogram(
        df,
        x="likes",
        marginal="box",
        title="Distribution of number of likes per tweet.",
    )
    st.plotly_chart(fig, use_container_width=True)


def display_profile_image(profile_image_url, user_name):
    """Retrieve and display profile image of the user."""

    urllib.request.urlretrieve(profile_image_url, "profile_image.jpg")

    return st.image(
        Image.open("profile_image.jpg"),
        caption=f"{user_name}",
        use_column_width="never",
        width=125,
    )


def plot_polarity_distribution(tweets_polarity):
    """Plot distribution of polarity scores for this user."""

    fig = ff.create_distplot(
        [tweets_polarity], group_labels=["Polarity score"], bin_size=[0.02]
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_timeseries_barplot(df, column, title):
    """Plot pre-computed time-series features of the account."""

    fig = px.bar(
        df.reset_index(), x="day", y=column, template="plotly_dark", title=title
    )

    st.plotly_chart(fig)
