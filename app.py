import streamlit as st
import pandas as pd
import spacy 
import plotly.figure_factory as ff
import os 

from utils import get_tweets
from nlp_utils import get_clean_tweets, get_nouns, get_sentiments
from plotting_utils import plot_day_tweets_count, plot_day_likes_count

nlp = spacy.load("en_core_web_sm")

def app():

    with st.sidebar:
        st.subheader("Tweeter Analyzer")
        st.markdown("This Streamlit application analyses tweets published by a given user together with some of their linguistic feautres like sentiment.")

        st.markdown("@mkacki98 on Github")

    st.title("T")
    default_user_name = st.selectbox(
            "Select one of the sample Twitter profiles.",
            (
                "elonmusk",
                "DonaldTrump",
            ),
        )

    user_input_name = st.text_input(
            "Input other Twitter account name."
        )

    if not user_input_name:
        user_input_name = default_user_name

    # Currently just one month (to be extended)
    start_date = "2022-01-01"
    end_date = "2022-01-31"

    df = get_tweets(user_input_name, start_date, end_date)

    tweets = df.Tweet.values
    docs = list(nlp.pipe(tweets))

    tweets_clean = get_clean_tweets(docs) 
    tweets_nouns = get_nouns(docs)

    tweets_sentiment = get_sentiments(tweets_clean)

    st.markdown(f"Average sentiment score for `{user_input_name}` is {round(sum(tweets_sentiment)/len(df),3)}")

    st.markdown("""---""")

    max_idx = tweets_sentiment.index(max(tweets_sentiment))
    min_idx = tweets_sentiment.index(min(tweets_sentiment))

    st.markdown(f"The most hateful Tweet is: `{tweets[max_idx]}` \n with score of {round(tweets_sentiment[max_idx],3)}")
    st.markdown(f"The least hateful Tweet is: `{tweets[min_idx]}` \n with score of {round(tweets_sentiment[min_idx],3)}")

    st.markdown("""---""")

    fig = ff.create_distplot(
        [tweets_sentiment], group_labels = ["Sentiment score"], bin_size=[0.02])

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""---""")
    
    st.subheader(f"User `{user_input_name}` has published `{len(df)}` Tweets in January 2022.")
    fig = plot_day_tweets_count(df)
    st.pyplot(fig)

    fig = plot_day_likes_count(df)
    st.pyplot(fig)




if __name__ == "__main__":
    st.set_page_config(layout="wide")
    app()