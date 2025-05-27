import streamlit as st
import praw
import pandas as pd
import numpy as np
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ==============================================
# Configuration
# ==============================================
CRYPTO_TICKERS = ["BTC", "ETH", "SOL", "XRP", "USDT", "USDC", "DOGE", "ADA"]
SUBREDDITS = ["CryptoCurrency", "CryptoMarkets", "Bitcoin", "ethereum", "solana"]
POST_LIMIT = 100
REQUEST_DELAY = 0.5

# Load crypto slang lexicon
with open(Path(__file__).parent / 'config' / 'crypto_slang.json') as f:
    CRYPTO_SLANG = json.load(f)["sentiment_modifiers"]

# Initialize sentiment analyzers
VADER = SentimentIntensityAnalyzer()
for word, score in CRYPTO_SLANG["positive"].items():
    VADER.lexicon[word] = score
for word, score in CRYPTO_SLANG["negative"].items():
    VADER.lexicon[word] = score

# ==============================================
# Core Functions
# ==============================================
def init_reddit_client():
    """Initialize PRAW Reddit client"""
    return praw.Reddit(
        client_id=st.secrets["REDDIT_CLIENT_ID"],
        client_secret=st.secrets["REDDIT_SECRET"],
        user_agent="crypto-sentiment/1.0"
    )

def clean_reddit_text(text):
    """Clean and normalize Reddit text"""
    if not isinstance(text, str):
        return ""
    
    # Remove URLs and special characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#|\n', ' ', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text.strip()

def extract_crypto_mentions(text):
    """Detect crypto tickers in text"""
    text = text.lower()
    mentions = set()
    
    for ticker in CRYPTO_TICKERS:
        if re.search(rf'\b{ticker.lower()}\b', text):
            mentions.add(ticker)
        if ticker == "BTC" and "bitcoin" in text:
            mentions.add(ticker)
        if ticker == "ETH" and "ethereum" in text:
            mentions.add(ticker)
    
    return list(mentions)

def analyze_sentiment(text):
    """Hybrid TextBlob + VADER sentiment analysis with crypto adjustments"""
    if not text or len(text.split()) < 3:
        return 0.0
    
    processed_text = clean_reddit_text(text)
    
    # TextBlob analysis
    tb_score = TextBlob(processed_text).sentiment.polarity
    
    # VADER analysis with crypto lexicon
    vd_score = VADER.polarity_scores(processed_text)["compound"]
    
    # Weighted hybrid score
    hybrid_score = (0.4 * tb_score) + (0.6 * vd_score)
    
    # Crypto-specific adjustments
    if "btc" in text.lower() or "bitcoin" in text.lower():
        hybrid_score *= 1.1
        
    if any(term in text.lower() for term in ["scam", "rug pull"]):
        hybrid_score -= 0.3
        
    return np.clip(hybrid_score, -1.0, 1.0)

@st.cache_data(ttl=3600)
def get_reddit_data():
    """Fetch and process Reddit posts"""
    reddit = init_reddit_client()
    all_posts = []
    
    for subreddit in SUBREDDITS:
        try:
            for submission in reddit.subreddit(subreddit).new(limit=POST_LIMIT):
                if submission.stickied:
                    continue
                    
                all_posts.append({
                    'id': submission.id,
                    'title': submission.title,
                    'content': submission.selftext,
                    'created_utc': submission.created_utc,
                    'score': submission.score,
                    'subreddit': subreddit,
                    'url': submission.url,
                    'tickers': extract_crypto_mentions(
                        f"{submission.title} {submission.selftext}"
                    )
                })
                time.sleep(REQUEST_DELAY)
        except Exception as e:
            st.error(f"Error fetching from r/{subreddit}: {str(e)}")
    
    df = pd.DataFrame(all_posts)
    if df.empty:
        return df
    
    df['created_dt'] = pd.to_datetime(df['created_utc'], unit='s')
    df['sentiment'] = df['title'].apply(analyze_sentiment)
    return df

# ==============================================
# Streamlit UI Components
# ==============================================
def render_sidebar():
    """Sidebar filters"""
    with st.sidebar:
        st.title("🔍 Filters")
        selected_ticker = st.selectbox(
            "Select Cryptocurrency",
            CRYPTO_TICKERS,
            index=0
        )
        
        time_filter = st.select_slider(
            "Time Range",
            options=["6h", "12h", "24h", "3d"],
            value="24h"
        )
        
        st.markdown("---")
        st.info(f"Analyzing {len(SUBREDDITS)} subreddits")
    
    return selected_ticker, time_filter

def render_metrics(df, ticker):
    """Key metrics display"""
    ticker_posts = df[df['tickers'].apply(lambda x: ticker in x)]
    avg_sentiment = ticker_posts['sentiment'].mean()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg. Sentiment", f"{avg_sentiment:.2f}")
    with col2:
        st.metric("Total Mentions", len(ticker_posts))
    with col3:
        st.metric("Most Active Sub", ticker_posts['subreddit'].mode()[0])

def render_sentiment_trend(df, ticker, time_range):
    """Interactive sentiment chart"""
    time_map = {
        "6h": timedelta(hours=6),
        "12h": timedelta(hours=12),
        "24h": timedelta(days=1),
        "3d": timedelta(days=3)
    }
    
    filtered = df[
        (df['tickers'].apply(lambda x: ticker in x)) &
        (df['created_dt'] > datetime.now() - time_map[time_range])
    ]
    
    fig = px.line(
        filtered.groupby(pd.Grouper(key='created_dt', freq='1h'))['sentiment'].mean(),
        title=f"Sentiment Trend for {ticker}",
        labels={'value': 'Sentiment Score'}
    )
    st.plotly_chart(fig, use_container_width=True)

def render_word_cloud(df, ticker):
    """Generate word cloud"""
    text = " ".join(df[df['tickers'].apply(lambda x: ticker in x)]['title'])
    wordcloud = WordCloud(width=800, height=400).generate(text)
    
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# ==============================================
# Main App
# ==============================================
def main():
    st.set_page_config(
        page_title="Crypto Sentiment Dashboard",
        layout="wide"
    )
    st.title("📊 Crypto Market Sentiment Dashboard")
    
    # Load data
    with st.spinner("Fetching latest Reddit data..."):
        df = get_reddit_data()
    
    # Sidebar
    selected_ticker, time_filter = render_sidebar()
    
    # Dashboard
    render_metrics(df, selected_ticker)
    
    tab1, tab2, tab3 = st.tabs(["Trend Analysis", "Top Posts", "Word Cloud"])
    
    with tab1:
        render_sentiment_trend(df, selected_ticker, time_filter)
    
    with tab2:
        st.dataframe(
            df[df['tickers'].apply(lambda x: selected_ticker in x)]
              .sort_values('sentiment', ascending=False)
              .head(10)[['title', 'sentiment', 'subreddit']],
            column_config={
                "sentiment": st.column_config.ProgressColumn(
                    min_value=-1,
                    max_value=1
                )
            }
        )
    
    with tab3:
        render_word_cloud(df, selected_ticker)

if __name__ == "__main__":
    main()