import streamlit as st
import pandas as pd
import spacy 
import plotly.figure_factory as ff

from utils import plot_day_tweets_count, plot_day_likes_count, get_tweets
from nlp_utils import get_clean_tweets, get_nouns, get_sentiments

nlp = spacy.load("en_core_web_sm")

def app():

    with st.sidebar:
        st.subheader("Tweeter Analyzer")
        st.markdown("This Streamlit application analyses tweets published by a given user together with some of their linguistic feautres like sentiment.")

        st.markdown("@mkacki98 on Github")

    st.title("Tweeter Analyzer")
    user_name = st.text_input("Pass a Twitter account name that you'd like to investigate", "elonmusk")

    # Currently just one month (to be extended)
    start_date = "2022-01-01"
    end_date = "2022-01-31"

    confirm_button = st.button("Apply the nickname.")

    if 'tweets' not in st.session_state or confirm_button:
        get_tweets(user_name, start_date, end_date)
        df = pd.read_csv(f"tweets/{user_name}.csv")
        st.session_state['df'] = df 

        tweets = df.Tweet.values
        docs = list(nlp.pipe(tweets))

        st.session_state['tweets'] = get_clean_tweets(docs) 
        st.session_state['nouns'] = get_nouns(docs)

        st.session_state['sentiments'] = get_sentiments(st.session_state['tweets'])

    if 'tweets' in st.session_state:
        st.subheader(f"Average sentiment score for `{user_name}` is {round(sum(st.session_state['sentiments'])/len(df),3)}")

        st.markdown("""---""")

        max_idx = st.session_state['sentiments'].index(max(st.session_state['sentiments']))
        min_idx = st.session_state['sentiments'].index(min(st.session_state['sentiments']))

        st.markdown(f"The most hateful Tweet is: `{tweets[max_idx]}` \n with score of {round(st.session_state['sentiments'][max_idx],3)}")
        st.markdown(f"The least hateful Tweet is: `{tweets[min_idx]}` \n with score of {round(st.session_state['sentiments'][min_idx],3)}")

        st.markdown("""---""")

        fig = ff.create_distplot(
            [st.session_state['sentiments']], group_labels = ["Sentiment score"], bin_size=[0.02])

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""---""")
        
        st.dataframe(pd.read_csv(f"tweets/{user_name}.csv"))

        st.subheader(f"User `{user_name}` has published `{len(df)}` Tweets in January 2022.")
        fig = plot_day_tweets_count(df)
        st.pyplot(fig)

        fig = plot_day_likes_count(df)
        st.pyplot(fig)

        st.markdown(st.session_state['tweets'])



if __name__ == "__main__":
    st.set_page_config(layout="wide")
    app()