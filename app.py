""" Main module running the Streamlit app. """

import streamlit as st
import spacy

from utils.general import (
    get_start_date,
    remove_spaces,
    get_tweets,
    check_format,
    compute_features_to_plot,
    get_user_info,
)
from utils.nlp import get_clean_tweets, get_nouns, get_polarity_scores
from utils.plotting import (
    plot_timeseries_barplot,
    plot_nouns_wordcloud,
    plot_polarity_distribution,
    display_correlation_prompts,
    display_profile_image,
    plot_correlation,
    plot_likes_distribution,
    display_profile_polarity,
)


nlp = spacy.load("en_core_web_sm")


def app():
    """Main function running the app."""

    with st.sidebar:
        st.subheader("Tweeter Analyzer")
        st.markdown(
            "This Streamlit application analyses tweets published by a given user together with some of their linguistic feautres like sentiment/polarity."
        )

        st.markdown("@mkacki98 on Github")

    col1, col2 = st.columns(2)

    with col1:
        default_user_name = st.selectbox(
            "Select one of the sample Twitter profiles.",
            (
                "bbclaurak",
                "elonmusk",
                "AJEnglish",
                "bbc",
                "FoxNews",
                "BarackObama",
                "OwenJones84",
                "ScottishLabour",
                "ScotTories",
            ),
        )

        user_input_name = st.text_input("Input other Twitter account name.")

        end_date = st.text_input("Input the date in format `YYYY-MM-DD`.", "2022-01-31")
        day_offset = st.number_input("How many days back from the end_date you want to investigate? Pick a number from 1 to 28.", 7)
    
    if day_offset > 28 or day_offset < 1:
        with col1:
            st.markdown("Please make sure the number you put is in [1, 28] range.")

    else:
        if not check_format(end_date):
            with col1:
                st.markdown(
                    "Please make sure you use the right format for the input end date."
                )


        else:
            
            if not user_input_name:
                user_input_name = default_user_name

            start_date = get_start_date(end_date, days_offset=day_offset)
            df = get_tweets(user_input_name, start_date, end_date)

            if len(df) == 0:
                with col1:
                    st.markdown(
                        f"Sorry, I can'f find `{user_input_name}`, can you change it?"
                    )
            else:
                user_info = get_user_info(user_input_name)
                with col1:
                    st.markdown(f"I am displaying the analysis for the period `{start_date}` to `{end_date}`.")
                
                with st.sidebar:
                    st.markdown("---")
                    st.markdown("You are seeing the Tweeter analysis of user:")

                    _, center_column, _, _ = st.columns(4)
                    with center_column:
                        display_profile_image(user_info[0], user_input_name)
                    
                    st.markdown(f"This user has `{user_info[1]}` followers.")

                    st.markdown("---")

                    st.markdown(
                        "The model used for text classification (sentiment analysis), called DestilBERT, was introduced by HuggingFace in [Sahn et al., 2019]"
                    )
                    st.markdown(
                        "The architecture of DestilBERT is a student/teacher one where the student (DestilBERT) is trained in a self-supervised way to replicate what the teacher (BERT) is doing."
                    )
                    st.markdown(
                        "The resulting model is 40% smaller, 60% faster and retains 97% of the language understanding capabilities."
                    )

                    st.markdown(
                        "It is uses knowledge destilation, a technique in which a bigger model is used as a teacher to train a lighter version of the teacher."
                    )

                    st.markdown("---")

                tweets = df.tweet.values
                docs = list(nlp.pipe(tweets))

                tweets_clean = get_clean_tweets(docs)
                tweets_nouns = get_nouns(docs)

                tweets_polarity = get_polarity_scores(tweets_clean)
                df["polarity"] = tweets_polarity

                df_features = compute_features_to_plot(df)

                with col2:
                    plot_nouns_wordcloud(tweets_nouns)

                st.markdown("---")
                col1, col2 = st.columns(2)

                with col1:
                    plot_timeseries_barplot(
                        df_features,
                        column="tweet_count",
                        title=f"How many tweets {user_input_name} posted each day?",
                    )
                    plot_timeseries_barplot(
                        df_features,
                        column="virality_score",
                        title=f"Which days were the most viral for {user_input_name}?",
                    )

                with col2:
                    plot_likes_distribution(df)
                    st.markdown("---")

                    most_liked = df[df.likes == max(df.likes)]

                    st.markdown(
                        f"User `{user_input_name}` has published `{len(df)}` tweets between {start_date} and {end_date}."
                    )

                    st.markdown(
                        f"This tweet went viral and gained `{most_liked.likes.values[0]}` likes: \n"
                    )
                    st.subheader(f"*{remove_spaces(most_liked.tweet.values[0])}*")
                    st.markdown(
                        f"It was also retweeted `{most_liked.retweets.values[0]}` times and quoted `{most_liked.quotes.values[0]}` times."
                    )

                st.markdown("""---""")

                col1, col2 = st.columns(2)

                with col1:

                    plot_polarity_distribution(tweets_polarity)

                with col2:
                    avg_polarity = round(sum(tweets_polarity) / len(df), 3)
                    display_profile_polarity(avg_polarity)

                    max_idx = tweets_polarity.index(max(tweets_polarity))
                    min_idx = tweets_polarity.index(min(tweets_polarity))

                    st.markdown(
                        f"Tweet with the highest polarity that `{user_input_name}` has published: \n"
                    )
                    st.subheader(f"*{remove_spaces(tweets[max_idx])}*")

                    st.markdown(
                        f"Tweet with the lowest polarity that `{user_input_name}` has published: \n "
                    )
                    st.subheader(f"*{remove_spaces(tweets[min_idx])}*")

                st.markdown("""---""")

                col1, col2 = st.columns(2)
                corr = df[["polarity", "likes", "retweets", "quotes"]].corr()

                with col1:
                    plot_correlation(corr)

                with col2:
                    display_correlation_prompts(corr)

                st.markdown("---")


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    app()
