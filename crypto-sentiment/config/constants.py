# Cryptocurrency tickers and aliases
CRYPTO_TICKERS = ["BTC", "ETH", "SOL", "XRP", "USDT", "USDC", "DOGE", "ADA"]

# Target subreddits
SUBREDDITS = [
    "CryptoCurrency",
    "CryptoMarkets",
    "Bitcoin",
    "ethereum",
    "solana"
]

# API configuration
API_CONFIG = {
    "POST_LIMIT": 100,
    "REQUEST_DELAY": 0.5,
    "MAX_RETRIES": 3
}

# Sentiment thresholds
SENTIMENT_THRESHOLDS = {
    "STRONG_BEARISH": -0.7,
    "BEARISH": -0.3,
    "NEUTRAL": 0.2,
    "BULLISH": 0.5,
    "STRONG_BULLISH": 0.8
}