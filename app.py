import streamlit as st
import spacy 

from utils import get_tweets, check_format, compute_features_to_plot
from nlp_utils import get_clean_tweets, get_nouns, get_sentiments
from plotting_utils import plot_timeseries_barplot, plot_sentiment_distribution

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
        st.markdown("Please make sure you use the right format for the input end date.")
    else:

        if not user_input_name:
            user_input_name = default_user_name

        df = get_tweets(user_input_name, end_date)

        if len(df) == 0:
            st.markdown("Sorry, I can'f find this user name, can you change it?")
        else:
            tweets = df.tweet.values
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

            df_features = compute_features_to_plot(df)
            plot_sentiment_distribution(tweets_sentiment)

            st.markdown("""---""")
            
            st.subheader(f"User `{user_input_name}` has published `{len(df)}` Tweets in January 2022.")
            
            plot_timeseries_barplot(df_features, column = 'tweet_count', title = 'Number of tweets per day.')
            plot_timeseries_barplot(df_features, column = 'avg_likes_per_day', title = 'Average number of likes received per day.')
            plot_timeseries_barplot(df_features, column = 'virality_score', title = 'Virality of tweets per day.')




if __name__ == "__main__":
    st.set_page_config(layout="wide")
    app()