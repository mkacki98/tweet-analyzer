import streamlit as st
import spacy 
from datetime import datetime

from utils import make_space, remove_spaces, get_tweets, check_format, compute_features_to_plot
from nlp_utils import get_clean_tweets, get_nouns, get_sentiments
from plotting_utils import plot_timeseries_barplot, plot_sentiment_distribution

nlp = spacy.load("en_core_web_sm")

def app():

    with st.sidebar:
        st.subheader("Tweeter Analyzer")
        st.markdown("This Streamlit application analyses tweets published by a given user together with some of their linguistic feautres like sentiment.")

        st.markdown("@mkacki98 on Github")

    col1, col2 = st.columns(2)
    
    with col1:
        default_user_name = st.selectbox(
                "Select one of the sample Twitter profiles.",
                (
                    "elonmusk",
                    "BarackObama", 
                    "justinbieber", 
                    "Cristiano"
                ),
            )

        user_input_name = st.text_input(
                "Input other Twitter account name."
            )

        end_date = st.text_input("Input the date in format `YYYY-MM-DD`.", "2022-01-31")
        
    if not check_format(end_date):
        with col1:
            st.markdown("Please make sure you use the right format for the input end date.")
    
    else:

        if not user_input_name:
            user_input_name = default_user_name

        df = get_tweets(user_input_name, end_date)

        if len(df) == 0:
            with col1:
                st.markdown("Sorry, I can'f find this user name, can you change it?")
        else:
            tweets = df.tweet.values
            docs = list(nlp.pipe(tweets))

            tweets_clean = get_clean_tweets(docs) 
            tweets_nouns = get_nouns(docs)

            tweets_sentiment = get_sentiments(tweets_clean)

            df_features = compute_features_to_plot(df)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:    
            make_space(2)
            plot_timeseries_barplot(df_features, column = 'tweet_count', title = 'How many tweets this user posted each day?')
            plot_timeseries_barplot(df_features, column = 'virality_score', title = "Which days were the most viral for this user?")

        with col2:
            most_liked = df[df.likes == max(df.likes)]

            st.subheader(f"`{user_input_name}` has published `{len(df)}` tweets a month before {end_date}.")
            st.subheader(f"This tweet went viral and gained `{most_liked.likes.values[0]}` likes: \n `{most_liked.tweet.values[0]}` \n") 
            st.subheader(f"It was also retweeted `{most_liked.retweets.values[0]}` times and quoted `{most_liked.quotes.values[0]}` times!")
            
        col1, col2 = st.columns(2)

        with col1:
            
            plot_sentiment_distribution(tweets_sentiment)
        
        with col2:
            st.subheader(f"Average polarity score for `{user_input_name}` is {round(sum(tweets_sentiment)/len(df),3)}")

            max_idx = tweets_sentiment.index(max(tweets_sentiment))
            min_idx = tweets_sentiment.index(min(tweets_sentiment))

            st.markdown(f"The most hateful tweet {user_input_name} has published has a polarity score of {round(tweets_sentiment[max_idx],3)}: \n")
            st.subheader(f"*{remove_spaces(tweets[max_idx])}*")
            
            st.markdown(f"The least hateful tweet {user_input_name} has published has a polarity score of {round(tweets_sentiment[min_idx],3)}: \n ")
            st.subheader(f"*{remove_spaces(tweets[min_idx])}*")

        st.markdown("""---""")



if __name__ == "__main__":
    st.set_page_config(layout="wide")
    app()