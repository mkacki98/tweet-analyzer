import streamlit as st
import spacy 

from utils.general import make_space, remove_spaces, get_tweets, check_format, compute_features_to_plot, get_user_info
from utils.nlp import get_clean_tweets, get_nouns, get_polarity_scores
from utils.plotting import plot_timeseries_barplot, plot_polarity_distribution, display_profile_image, plot_correlation, plot_likes_distribution, display_profile_polarity

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
        user_info = get_user_info(user_input_name)

        if len(df) == 0:
            with col1:
                st.markdown(f"Sorry, I can'f find `{user_input_name}`, can you change it?")
        else:

            with col2: 
                display_profile_image(user_info[0], user_input_name)

            tweets = df.tweet.values
            docs = list(nlp.pipe(tweets))

            tweets_clean = get_clean_tweets(docs) 
            tweets_nouns = get_nouns(docs)

            tweets_polarity = get_polarity_scores(tweets_clean)
            df['polarity'] = tweets_polarity

            df_features = compute_features_to_plot(df)

            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:    
                plot_timeseries_barplot(df_features, column = 'tweet_count', title = f'How many tweets {user_input_name} posted each day?')
                plot_timeseries_barplot(df_features, column = 'virality_score', title = f"Which days were the most viral for {user_input_name}?")

            with col2:
                plot_likes_distribution(df)
                st.markdown("---")

                most_liked = df[df.likes == max(df.likes)]

                st.markdown(f"User `{user_input_name}` has published `{len(df)}` that month time.")

                st.markdown(f"This tweet went viral and gained `{most_liked.likes.values[0]}` likes: \n") 
                st.subheader(f"*{remove_spaces(most_liked.tweet.values[0])}*")
                st.markdown(f"It was also retweeted `{most_liked.retweets.values[0]}` times and quoted `{most_liked.quotes.values[0]}` times.")

            st.markdown("""---""")

            col1, col2 = st.columns(2)

            with col1:
                
                plot_polarity_distribution(tweets_polarity)
            
            with col2:
                avg_polarity  = round(sum(tweets_polarity)/len(df),3)
                display_profile_polarity(avg_polarity)

                max_idx = tweets_polarity.index(max(tweets_polarity))
                min_idx = tweets_polarity.index(min(tweets_polarity))

                st.markdown(f"The most hateful tweet `{user_input_name}` has published: \n")
                st.subheader(f"*{remove_spaces(tweets[min_idx])}*")
                
                st.markdown(f"The least hateful tweet `{user_input_name}` has published: \n ")
                st.subheader(f"*{remove_spaces(tweets[max_idx])}*")

            st.markdown("""---""")

            col1, col2 = st.columns(2)

            corr = df[['polarity', 'likes', 'retweets', 'quotes']].corr()

            with col1:
                plot_correlation(corr)

            with col2:
                
                st.markdown("Correlations between *polarity* and *likes* with other feaures: \n ")
                st.dataframe(corr[['polarity', 'likes']])
        
            st.markdown("---")

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    app()