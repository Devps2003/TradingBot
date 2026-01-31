"""
Central configuration for the Indian Market Trading Research Agent.
All settings and constants are defined here.
"""

import os
from pathlib import Path
from typing import Dict, List

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
CACHE_DIR = DATA_DIR / "cache"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ============================================================================
# API CONFIGURATION
# ============================================================================

# Load from environment variables (recommended) or api_keys.py
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# FREE LLM Options (recommended order):
# 1. "groq" - FREE, very fast, uses Llama 3.1 70B (get key at console.groq.com)
# 2. "ollama" - FREE, runs locally (need to install ollama)
# 3. "gemini" - FREE tier available (get key at ai.google.dev)
# 4. "openai" / "anthropic" - PAID options

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # Get FREE key at console.groq.com
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Get FREE key at ai.google.dev

# LLM Provider Selection (default: groq for FREE high-quality AI)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# Groq settings (FREE, recommended)
# Available models: llama-3.1-8b-instant, llama3-70b-8192, mixtral-8x7b-32768
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # Fast and reliable

# Ollama settings (FREE, local)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Gemini settings (FREE tier)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# ============================================================================
# DATA FETCHING SETTINGS
# ============================================================================

# Cache settings
CACHE_DURATION_MINUTES = 15
PRICE_CACHE_DURATION = 5  # More frequent for live prices
NEWS_CACHE_DURATION = 30  # Less frequent for news

# Historical data
HISTORICAL_DATA_DAYS = 365
INTRADAY_DATA_DAYS = 5

# Rate limiting (requests per minute)
NSE_RATE_LIMIT = 10
YAHOO_RATE_LIMIT = 30
SCRAPING_RATE_LIMIT = 5

# Request settings
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 2

# User agent for scraping
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# ============================================================================
# ANALYSIS WEIGHTS
# ============================================================================

# Default weights for signal generation (must sum to 1.0)
ANALYSIS_WEIGHTS: Dict[str, float] = {
    "technical": 0.35,
    "pattern": 0.15,
    "fundamental": 0.20,
    "sentiment": 0.15,
    "volume": 0.10,
    "market_context": 0.05,
}

# Weight adjustments for different market regimes
REGIME_WEIGHT_ADJUSTMENTS = {
    "trending": {
        "technical": 0.40,
        "pattern": 0.15,
        "fundamental": 0.15,
        "sentiment": 0.15,
        "volume": 0.10,
        "market_context": 0.05,
    },
    "ranging": {
        "technical": 0.30,
        "pattern": 0.20,
        "fundamental": 0.20,
        "sentiment": 0.15,
        "volume": 0.10,
        "market_context": 0.05,
    },
    "volatile": {
        "technical": 0.25,
        "pattern": 0.10,
        "fundamental": 0.25,
        "sentiment": 0.20,
        "volume": 0.10,
        "market_context": 0.10,
    },
}

# ============================================================================
# SIGNAL THRESHOLDS
# ============================================================================

SIGNAL_THRESHOLDS = {
    "strong_buy": 80,
    "buy": 65,
    "hold_upper": 65,
    "hold_lower": 35,
    "sell": 35,
    "strong_sell": 20,
}

# Minimum confidence to generate a signal
MIN_CONFIDENCE_FOR_SIGNAL = 50

# ============================================================================
# RISK MANAGEMENT
# ============================================================================

# Position sizing
DEFAULT_RISK_PER_TRADE = 0.02  # 2% of capital per trade
MAX_PORTFOLIO_RISK = 0.10  # 10% total portfolio at risk
MAX_SINGLE_POSITION = 0.15  # 15% of capital in single stock
MIN_POSITION_SIZE = 0.02  # 2% minimum position

# Stop loss settings
DEFAULT_STOP_LOSS_PERCENT = 0.05  # 5% default stop loss
ATR_STOP_MULTIPLIER = 2.0  # Stop loss = ATR * multiplier
MAX_STOP_LOSS_PERCENT = 0.10  # Never risk more than 10%

# Target settings
MIN_RISK_REWARD_RATIO = 1.5
DEFAULT_TARGET_MULTIPLIER = 2.0  # Target = risk * multiplier

# ============================================================================
# TECHNICAL INDICATOR SETTINGS
# ============================================================================

INDICATOR_SETTINGS = {
    # Moving Averages
    "sma_periods": [20, 50, 100, 200],
    "ema_periods": [9, 21, 50, 200],
    
    # Momentum
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "stochastic": {"k_period": 14, "d_period": 3, "smooth": 3},
    "cci_period": 20,
    "williams_period": 14,
    "roc_period": 12,
    "mfi_period": 14,
    
    # Trend
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "adx_period": 14,
    "adx_strong_trend": 25,
    "supertrend": {"period": 10, "multiplier": 3},
    
    # Volatility
    "bollinger": {"period": 20, "std": 2},
    "atr_period": 14,
    "keltner": {"period": 20, "multiplier": 2},
    "donchian_period": 20,
    
    # Volume
    "volume_sma_period": 20,
    "obv_signal_period": 21,
    "cmf_period": 20,
    "vwap_period": 14,
}

# ============================================================================
# PATTERN RECOGNITION SETTINGS
# ============================================================================

PATTERN_SETTINGS = {
    # Candlestick pattern detection
    "min_body_percent": 0.3,  # Minimum body as % of range
    "doji_body_percent": 0.1,  # Max body for doji
    
    # Chart pattern detection
    "min_pattern_bars": 10,  # Minimum bars for chart patterns
    "max_pattern_bars": 100,  # Maximum lookback
    "breakout_confirmation_percent": 0.02,  # 2% above level
    
    # Pattern confidence thresholds
    "min_pattern_confidence": 60,
    "high_confidence_threshold": 80,
}

# Historical pattern accuracy (backtested)
PATTERN_HISTORICAL_ACCURACY = {
    "bull_flag": 0.68,
    "bear_flag": 0.65,
    "double_bottom": 0.72,
    "double_top": 0.70,
    "head_and_shoulders": 0.75,
    "inverse_head_and_shoulders": 0.73,
    "ascending_triangle": 0.70,
    "descending_triangle": 0.68,
    "cup_and_handle": 0.71,
    "bullish_engulfing": 0.63,
    "bearish_engulfing": 0.62,
    "morning_star": 0.66,
    "evening_star": 0.64,
    "hammer": 0.60,
    "shooting_star": 0.58,
}

# ============================================================================
# SENTIMENT ANALYSIS SETTINGS
# ============================================================================

SENTIMENT_SETTINGS = {
    "model": "finbert",  # or "vader" for rule-based
    "news_lookback_days": 7,
    "social_lookback_days": 3,
    
    # Thresholds
    "positive_threshold": 0.3,
    "negative_threshold": -0.3,
    
    # Source weights
    "source_weights": {
        "economic_times": 1.0,
        "moneycontrol": 0.9,
        "business_standard": 0.9,
        "livemint": 0.85,
        "reuters": 1.0,
        "social_media": 0.5,
    },
}

# ============================================================================
# STOCK UNIVERSES
# ============================================================================

NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "HCLTECH", "AXISBANK", "ASIANPAINT", "MARUTI",
    "SUNPHARMA", "TITAN", "BAJFINANCE", "WIPRO", "ULTRACEMCO",
    "NESTLEIND", "TATAMOTORS", "ONGC", "NTPC", "M&M",
    "POWERGRID", "JSWSTEEL", "TATASTEEL", "ADANIENT", "ADANIPORTS",
    "COALINDIA", "BAJAJFINSV", "TECHM", "HDFCLIFE", "GRASIM",
    "DRREDDY", "SBILIFE", "DIVISLAB", "BRITANNIA", "CIPLA",
    "EICHERMOT", "HINDALCO", "APOLLOHOSP", "INDUSINDBK", "TATACONSUM",
    "BPCL", "HEROMOTOCO", "UPL", "SHREECEM", "BAJAJ-AUTO",
]

NIFTY_NEXT_50 = [
    "ADANIGREEN", "AMBUJACEM", "AUROPHARMA", "BANKBARODA", "BERGEPAINT",
    "BIOCON", "BOSCHLTD", "CHOLAFIN", "COLPAL", "DABUR",
    "DLF", "GAIL", "GODREJCP", "HAVELLS", "ICICIGI",
    "ICICIPRULI", "IDEA", "IDFCFIRSTB", "IGL", "INDIGO",
    "INDUSTOWER", "JUBLFOOD", "LICI", "LUPIN", "MARICO",
    "MCDOWELL-N", "MUTHOOTFIN", "NAUKRI", "NHPC", "NMDC",
    "PEL", "PETRONET", "PFC", "PIIND", "PNB",
    "RECLTD", "SBICARD", "SIEMENS", "SRF", "TATAPOWER",
    "TORNTPHARM", "TRENT", "VEDL", "VBL", "ZOMATO",
]

# Popular mid/small cap stocks
POPULAR_STOCKS = [
    # Multibaggers & popular
    "IRCTC", "IRFC", "RVNL", "SUZLON", "NHPC", "SJVN", "PFC", "RECLTD",
    "IDEA", "YESBANK", "ZOMATO", "PAYTM", "NYKAA", "POLICYBZR",
    "DELHIVERY", "CARTRADE", "HAPPSTMNDS", "RATEGAIN", "MEDPLUS",
    # Defense
    "HAL", "BEL", "BHEL", "MAZAGON", "COCHINSHIP", "GRSE",
    # Railways
    "IRCON", "RITES", "RAILTEL", "TITAGARH",
    # Energy & Green
    "TATAPOWER", "ADANIGREEN", "ADANIENSOL", "NHPC", "SJVN", "JPPOWER",
    # Realty
    "LODHA", "OBEROIRLTY", "GODREJPROP", "PRESTIGE", "BRIGADE", "SOBHA",
    # Others
    "ZYDUSLIFE", "MANKIND", "RAINBOW", "LALPATHLAB", "METROPOLIS",
]

# ============================================================================
# ETFs & COMMODITIES (Yahoo Finance symbols)
# ============================================================================

ETFS = {
    # Gold ETFs
    "GOLDBEES": "GOLDBEES.NS",      # Nippon Gold ETF
    "GOLDCASE": "GOLDCASE.NS",      # ICICI Gold ETF
    "GOLDISGR": "GOLD.NS",          # SBI Gold ETF
    
    # Silver ETFs
    "SILVERBEES": "SILVERBEES.NS",  # Nippon Silver ETF
    "SILVERETF": "SILVERETF.NS",    # ICICI Silver ETF
    
    # Index ETFs
    "NIFTYBEES": "NIFTYBEES.NS",    # Nifty 50 ETF
    "BANKBEES": "BANKBEES.NS",      # Bank Nifty ETF
    "ITBEES": "ITBEES.NS",          # IT Index ETF
    "JUNIORBEES": "JUNIORBEES.NS",  # Nifty Next 50 ETF
    "MIDCPNIFTY": "MIDCPNIFTY.NS",  # Midcap ETF
    "NETFIT": "NETFIT.NS",          # Nifty 50 ETF (Motilal)
    
    # Sector ETFs
    "PHARMABEES": "PHARMABEES.NS",
    "PSUBNKBEES": "PSUBNKBEES.NS",
    "INFRABEES": "INFRABEES.NS",
    "CONSUMBEES": "CONSUMBEES.NS",
    
    # International
    "N100": "N100.NS",              # Nasdaq 100 ETF
    "MAFANG": "MAFANG.NS",          # FAANG stocks ETF
    "HNGSNGBEES": "HNGSNGBEES.NS",  # Hang Seng ETF
}

# Commodities (Yahoo Finance)
COMMODITIES = {
    "GOLD": "GC=F",
    "SILVER": "SI=F", 
    "CRUDE_OIL": "CL=F",
    "NATURAL_GAS": "NG=F",
    "COPPER": "HG=F",
}

# Full universe for scanning
FULL_UNIVERSE = NIFTY_50 + NIFTY_NEXT_50 + POPULAR_STOCKS
ETF_LIST = list(ETFS.keys())

# Sector classification
SECTORS = {
    "BANKING": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK", "BANKBARODA", "PNB"],
    "IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM"],
    "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO", "COLPAL", "GODREJCP"],
    "AUTO": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT"],
    "PHARMA": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "AUROPHARMA", "BIOCON", "LUPIN", "TORNTPHARM"],
    "ENERGY": ["RELIANCE", "ONGC", "BPCL", "GAIL", "NTPC", "POWERGRID", "TATAPOWER", "ADANIGREEN"],
    "METALS": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "NMDC", "COALINDIA"],
    "FINANCE": ["BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE", "ICICIGI", "ICICIPRULI", "MUTHOOTFIN"],
    "REALTY": ["DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE"],
    "INFRA": ["LT", "ADANIENT", "ADANIPORTS", "ULTRACEMCO", "SHREECEM", "AMBUJACEM", "GRASIM"],
}

# ============================================================================
# MARKET TIMING
# ============================================================================

# Indian market hours (IST)
MARKET_OPEN_TIME = "09:15"
MARKET_CLOSE_TIME = "15:30"
PRE_MARKET_START = "09:00"
PRE_MARKET_END = "09:08"

# Trading sessions
OPENING_SESSION_END = "09:45"
CLOSING_SESSION_START = "15:00"

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================

# Rich console settings
CONSOLE_WIDTH = 120
TABLE_STYLE = "rounded"
POSITIVE_COLOR = "green"
NEGATIVE_COLOR = "red"
NEUTRAL_COLOR = "yellow"
HEADER_COLOR = "cyan"

# Report settings
GENERATE_PDF_REPORTS = False
SAVE_DAILY_REPORTS = True
REPORT_FORMAT = "txt"  # "txt", "html", or "pdf"

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "logs" / "trading_agent.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5

# ============================================================================
# DATABASE
# ============================================================================

DATABASE_URL = f"sqlite:///{DATA_DIR / 'market_data.db'}"
