# 🚀 Indian Market Trading Agent - Complete Technical Documentation

> **Author:** Dev PS  
> **Version:** 2.0  
> **Stack:** Python, Streamlit, Pandas, AI/ML, Cloud (Supabase)  
> **Purpose:** AI-powered swing trading research system for Indian markets

---

## 📋 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Data Pipeline](#3-data-pipeline)
4. [Technical Analysis Engine](#4-technical-analysis-engine)
5. [AI/ML Components](#5-aiml-components)
6. [Signal Generation System](#6-signal-generation-system)
7. [Risk Management](#7-risk-management)
8. [Portfolio Management](#8-portfolio-management)
9. [Cloud Infrastructure](#9-cloud-infrastructure)
10. [Frontend (UI/UX)](#10-frontend-uiux)
11. [API Design](#11-api-design)
12. [Flowcharts](#12-flowcharts)
13. [Algorithms & Logic](#13-algorithms--logic)
14. [Trading Terms Glossary](#14-trading-terms-glossary)

---

## 1. Executive Summary

### What is this project?

An **AI-powered stock research and trading signal system** designed for swing traders (1-2 week holding periods) in the Indian stock market (NSE/BSE). It combines:

- **Quantitative Analysis:** Technical indicators, pattern recognition, volume analysis
- **AI/LLM Integration:** Natural language insights using Groq/Ollama (zero cost)
- **Real-time Data:** Live prices from Yahoo Finance, news sentiment
- **Portfolio Tracking:** Full P&L tracking with cloud sync
- **Professional UI:** Streamlit-based dashboard accessible from any device

### Key Features

| Feature | Technology | Purpose |
|---------|------------|---------|
| Multi-timeframe Analysis | Pandas, TA-Lib | Trend alignment across daily/weekly |
| Smart Money Detection | Volume-Price Analysis | Identify institutional accumulation |
| Pattern Recognition | Rule-based + Heuristics | Detect chart patterns |
| Relative Strength | Statistical Analysis | Compare stock vs benchmark |
| AI Advisor | Groq LLM (Llama 3.1) | Personalized trading advice |
| Cloud Sync | Supabase (PostgreSQL) | Access portfolio from any device |
| Risk Calculator | Mathematical Models | Position sizing, stop-loss levels |

### Business Value

- **Time Saved:** Automates 2-3 hours of daily research
- **Objectivity:** Removes emotional bias from trading decisions
- **Risk Management:** Enforces 2% risk rule automatically
- **Accessibility:** Works on phone during market hours

---

## 2. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           PRESENTATION LAYER                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Streamlit Web Dashboard                       │   │
│  │   • Dashboard  • Research  • Scanner  • Portfolio  • AI Chat    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          APPLICATION LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │ TradingAgent │  │ ProAnalyzer  │  │ MoneyMaker   │  │ LLMReasoner│  │
│  │   (Main)     │  │  (Analysis)  │  │  (Scanner)   │  │    (AI)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          BUSINESS LOGIC LAYER                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         ANALYZERS                                │   │
│  │  • TechnicalAnalyzer    • VolumeAnalyzer     • PatternRecognizer│   │
│  │  • FundamentalAnalyzer  • SentimentAnalyzer  • CorrelationAnalyzer│ │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                          SIGNALS                                 │   │
│  │  • SignalGenerator      • ConfidenceScorer   • RiskCalculator   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         PORTFOLIO                                │   │
│  │  • SmartPortfolio       • PositionSizer      • TradeTracker     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            DATA LAYER                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                       DATA FETCHERS                              │   │
│  │  • PriceFetcher (Yahoo)  • NewsFetcher       • FII_DII_Fetcher  │   │
│  │  • GlobalFetcher         • BulkDealsFetcher  • InsiderFetcher   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        STORAGE                                   │   │
│  │  • Local JSON Files      • Supabase (Cloud)  • Cache Layer      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
indian-market-agent/
│
├── app.py                      # Streamlit UI (main entry point)
├── run.py                      # CLI interactive mode
├── requirements.txt            # Python dependencies
│
├── config/
│   ├── settings.py             # Configuration (stock lists, API keys)
│   ├── trading_rules.py        # Trading parameters
│   └── api_keys.py             # API credentials
│
├── src/
│   ├── agent.py                # Main orchestrator (TradingAgent class)
│   │
│   ├── data_fetchers/          # Data ingestion layer
│   │   ├── price_fetcher.py    # OHLCV data from Yahoo Finance
│   │   ├── global_fetcher.py   # Global market cues
│   │   ├── news_fetcher.py     # News headlines
│   │   ├── fii_dii_fetcher.py  # Institutional flow data
│   │   ├── bulk_deals_fetcher.py
│   │   └── insider_fetcher.py
│   │
│   ├── analyzers/              # Analysis engines
│   │   ├── pro_analyzer.py     # ⭐ Main professional analyzer
│   │   ├── technical_analyzer.py
│   │   ├── volume_analyzer.py
│   │   ├── pattern_recognizer.py
│   │   ├── sentiment_analyzer.py
│   │   ├── fundamental_analyzer.py
│   │   └── correlation_analyzer.py
│   │
│   ├── signals/                # Signal generation
│   │   ├── money_maker.py      # ⭐ Opportunity scanner
│   │   ├── signal_generator.py
│   │   ├── confidence_scorer.py
│   │   └── risk_calculator.py
│   │
│   ├── ai_layer/               # AI/ML components
│   │   ├── llm_reasoner.py     # ⭐ LLM integration (Groq/Ollama)
│   │   ├── pattern_ml.py       # ML pattern recognition
│   │   └── sentiment_model.py  # NLP sentiment
│   │
│   ├── portfolio/              # Portfolio management
│   │   ├── smart_portfolio.py  # ⭐ Full P&L tracking
│   │   ├── cloud_storage.py    # Supabase sync
│   │   ├── position_sizer.py
│   │   └── trade_tracker.py
│   │
│   └── utils/                  # Utilities
│       ├── helpers.py
│       ├── logger.py
│       └── indian_market_utils.py
│
└── data/                       # Local data storage
    ├── portfolio.json
    ├── trade_history.json
    └── cache/                  # API response cache
```

### Technology Stack

| Layer | Technology | Why? |
|-------|------------|------|
| Frontend | Streamlit | Rapid prototyping, Python-native |
| Backend | Python 3.11 | Data science ecosystem |
| Data Processing | Pandas, NumPy | Industry standard for financial data |
| Technical Analysis | `ta` library | 40+ indicators built-in |
| Visualization | Plotly | Interactive charts |
| AI/LLM | Groq API (Llama 3.1) | Free, fast inference |
| Database | Supabase (PostgreSQL) | Free tier, real-time sync |
| Deployment | Streamlit Cloud | Free hosting |
| Version Control | Git/GitHub | Code management |

---

## 3. Data Pipeline

### Data Flow Diagram

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   EXTERNAL APIs  │     │    FETCHERS      │     │   PROCESSORS     │
│                  │     │                  │     │                  │
│  Yahoo Finance ──┼────▶│  PriceFetcher   ─┼────▶│  DataCleaner     │
│  Google News   ──┼────▶│  NewsFetcher    ─┼────▶│  Normalizer      │
│  NSE/BSE       ──┼────▶│  FII_DII_Fetcher─┼────▶│  CacheManager    │
│                  │     │                  │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                           │
                                                           ▼
                                                  ┌──────────────────┐
                                                  │    ANALYZERS     │
                                                  │                  │
                                                  │  Technical ──────┼──┐
                                                  │  Volume    ──────┼──┤
                                                  │  Patterns  ──────┼──┤
                                                  │  Sentiment ──────┼──┤
                                                  └──────────────────┘  │
                                                                        ▼
                                                  ┌──────────────────────┐
                                                  │   SIGNAL GENERATOR   │
                                                  │                      │
                                                  │  Score Aggregation   │
                                                  │  Signal: BUY/SELL    │
                                                  │  Entry/Stop/Target   │
                                                  └──────────────────────┘
```

### PriceFetcher Implementation

```python
class PriceFetcher:
    """
    Fetches OHLCV data from Yahoo Finance.
    
    Key Methods:
    - get_historical_data(symbol, period, interval)
    - get_live_price(symbol)
    
    Data Returned:
    - date, open, high, low, close, volume
    - Cleaned and indexed for analysis
    """
    
    def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        yahoo_symbol = f"{symbol}.NS"  # NSE suffix
        ticker = yf.Ticker(yahoo_symbol)
        df = ticker.history(period=period)
        
        # Clean and standardize
        df.reset_index(inplace=True)
        df.columns = df.columns.str.lower()
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        
        return df
```

### Caching Strategy

```
Request → Check Cache → If Valid → Return Cached
                     → If Expired → Fetch New → Update Cache → Return
                     
Cache TTL:
- Price Data: 5 minutes (live), 1 hour (historical)
- News: 30 minutes
- Global Cues: 15 minutes
- FII/DII: 6 hours (updated once daily)
```

---

## 4. Technical Analysis Engine

### ProAnalyzer - Core Analysis Module

The `ProAnalyzer` class is the heart of the technical analysis system.

```python
class ProAnalyzer:
    """
    Professional-grade stock analyzer combining multiple techniques.
    
    Output: Dictionary containing all analysis components + final signal
    """
    
    def full_analysis(self, symbol: str) -> Dict[str, Any]:
        # Fetch multi-timeframe data
        df_daily = self.price_fetcher.get_historical_data(symbol, "1y", "1d")
        df_weekly = self._resample_to_weekly(df_daily)
        
        # Run analysis components
        trend = self._analyze_trend(df_daily, df_weekly)
        momentum = self._analyze_momentum(df_daily)
        volume = self._analyze_volume(df_daily)
        patterns = self._detect_patterns(df_daily)
        levels = self._find_key_levels(df_daily)
        strength = self._calculate_relative_strength(symbol, df_daily)
        
        # Calculate composite scores
        scores = self._calculate_scores(trend, momentum, volume, patterns, strength)
        
        # Generate final signal
        signal = self._generate_signal(scores, current_price, levels, df_daily)
        
        return {
            "symbol": symbol,
            "trend": trend,
            "momentum": momentum,
            "volume": volume,
            "patterns": patterns,
            "levels": levels,
            "relative_strength": strength,
            "scores": scores,
            "signal": signal,
        }
```

### 4.1 Trend Analysis

**Purpose:** Determine the primary direction of price movement.

**Indicators Used:**

| Indicator | Formula | Interpretation |
|-----------|---------|----------------|
| EMA 8 | Exponential Moving Average (8 days) | Short-term trend |
| EMA 21 | Exponential Moving Average (21 days) | Medium-term trend |
| EMA 50 | Exponential Moving Average (50 days) | Long-term trend |
| EMA 200 | Exponential Moving Average (200 days) | Primary trend |
| ADX | Average Directional Index | Trend strength (0-100) |

**Logic:**

```python
def _analyze_trend(self, df: pd.DataFrame, df_weekly: pd.DataFrame) -> Dict:
    close = df['close']
    
    # Calculate EMAs
    ema_8 = close.ewm(span=8).mean()
    ema_21 = close.ewm(span=21).mean()
    ema_50 = close.ewm(span=50).mean()
    ema_200 = close.ewm(span=200).mean()
    
    current = close.iloc[-1]
    
    # Trend determination
    if current > ema_21.iloc[-1] > ema_50.iloc[-1]:
        daily_trend = "BULLISH"
    elif current < ema_21.iloc[-1] < ema_50.iloc[-1]:
        daily_trend = "BEARISH"
    else:
        daily_trend = "NEUTRAL"
    
    # ADX for trend strength
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    adx_value = adx.adx().iloc[-1]
    
    trend_strength = "STRONG" if adx_value > 25 else "WEAK" if adx_value < 20 else "MODERATE"
    
    # Multi-timeframe alignment
    aligned = (daily_trend == weekly_trend) and (daily_trend != "NEUTRAL")
    
    return {
        "daily": daily_trend,
        "weekly": weekly_trend,
        "aligned": aligned,
        "strength": trend_strength,
        "adx": adx_value,
        "above_200ema": current > ema_200.iloc[-1],
    }
```

**Key Insight:** Trend alignment across timeframes is crucial. A bullish daily trend with a bullish weekly trend has much higher probability than conflicting trends.

### 4.2 Momentum Analysis

**Purpose:** Measure the speed and strength of price movements.

**Indicators Used:**

| Indicator | Range | Overbought | Oversold |
|-----------|-------|------------|----------|
| RSI (14) | 0-100 | > 70 | < 30 |
| Stochastic %K | 0-100 | > 80 | < 20 |
| MACD | Variable | Above signal line | Below signal line |
| ROC (10, 20) | % change | Positive = momentum | Negative = weakness |

**RSI Divergence Detection:**

```python
def _check_rsi_divergence(self, price: pd.Series, rsi: pd.Series) -> str:
    """
    Divergence = price and indicator moving in opposite directions
    
    Bullish Divergence: Price making lower lows, RSI making higher lows
    → Signals potential reversal UP
    
    Bearish Divergence: Price making higher highs, RSI making lower highs
    → Signals potential reversal DOWN
    """
    price_slope = (price.iloc[-1] - price.iloc[-20]) / 20
    rsi_slope = (rsi.iloc[-1] - rsi.iloc[-20]) / 20
    
    if price_slope < 0 and rsi_slope > 0 and rsi.iloc[-1] < 40:
        return "BULLISH"  # Price down, RSI up = reversal coming
    elif price_slope > 0 and rsi_slope < 0 and rsi.iloc[-1] > 60:
        return "BEARISH"  # Price up, RSI down = weakness
    
    return "NONE"
```

### 4.3 Volume Analysis (Smart Money Detection)

**Purpose:** Identify institutional accumulation or distribution.

**Concept:** "Smart money" (institutional investors) leaves footprints in volume patterns.

```python
def _analyze_volume(self, df: pd.DataFrame) -> Dict:
    close = df['close']
    volume = df['volume']
    
    avg_volume_20 = volume.rolling(20).mean().iloc[-1]
    
    # Count accumulation vs distribution days
    accumulation_days = 0
    distribution_days = 0
    
    for i in range(-10, 0):
        if close.iloc[i] > close.iloc[i-1]:  # Up day
            if volume.iloc[i] > avg_volume_20:  # High volume
                accumulation_days += 1  # Smart money buying
        else:  # Down day
            if volume.iloc[i] > avg_volume_20:  # High volume
                distribution_days += 1  # Smart money selling
    
    if accumulation_days > distribution_days + 2:
        smart_money = "ACCUMULATING"  # Bullish
    elif distribution_days > accumulation_days + 2:
        smart_money = "DISTRIBUTING"  # Bearish
    else:
        smart_money = "NEUTRAL"
    
    # On-Balance Volume (OBV)
    obv = ta.volume.OnBalanceVolumeIndicator(close, volume)
    obv_trend = "UP" if obv.iloc[-1] > obv.iloc[-20] else "DOWN"
    
    return {
        "smart_money": smart_money,
        "obv_trend": obv_trend,
        "accumulation_days": accumulation_days,
        "distribution_days": distribution_days,
    }
```

**On-Balance Volume (OBV) Formula:**
```
If Close > Previous Close: OBV = Previous OBV + Volume
If Close < Previous Close: OBV = Previous OBV - Volume
If Close = Previous Close: OBV = Previous OBV
```

### 4.4 Pattern Recognition

**Candlestick Patterns Detected:**

| Pattern | Type | Description | Strength |
|---------|------|-------------|----------|
| Bullish Engulfing | Reversal | Today's body completely covers yesterday's bearish body | 75 |
| Bearish Engulfing | Reversal | Today's body completely covers yesterday's bullish body | 75 |
| Hammer | Reversal | Long lower wick, small body at top | 70 |
| Shooting Star | Reversal | Long upper wick, small body at bottom | 70 |
| Doji | Indecision | Open ≈ Close (very small body) | 50 |

**Chart Patterns Detected:**

| Pattern | Logic | Implication |
|---------|-------|-------------|
| Higher Highs & Higher Lows | Each swing high > previous, each swing low > previous | Uptrend continuation |
| 20-Day Breakout | Close > 20-day high | Momentum breakout |
| 20-Day Breakdown | Close < 20-day low | Breakdown |
| Consolidation (Squeeze) | ATR < 70% of ATR 20 days ago | Volatility expansion coming |

```python
def _detect_patterns(self, df: pd.DataFrame) -> Dict:
    patterns_found = []
    
    close = df['close']
    open_price = df['open']
    high = df['high']
    low = df['low']
    
    # Bullish Engulfing
    if (close.iloc[-1] > open_price.iloc[-1] and  # Today bullish
        close.iloc[-2] < open_price.iloc[-2] and  # Yesterday bearish
        close.iloc[-1] > open_price.iloc[-2] and  # Today close > yesterday open
        open_price.iloc[-1] < close.iloc[-2]):    # Today open < yesterday close
        patterns_found.append({
            "name": "Bullish Engulfing",
            "type": "BULLISH",
            "strength": 75
        })
    
    # 20-Day Breakout
    high_20 = high.rolling(20).max().iloc[-2]
    if close.iloc[-1] > high_20:
        patterns_found.append({
            "name": "20-Day Breakout",
            "type": "BULLISH",
            "strength": 85
        })
    
    return {
        "patterns": patterns_found,
        "dominant": "BULLISH" if bullish > bearish else "BEARISH" if bearish > bullish else "NEUTRAL"
    }
```

### 4.5 Key Levels Detection

**Levels Calculated:**

1. **Support/Resistance:** Recent swing highs/lows
2. **Pivot Points:** Classic pivot point formula
3. **52-Week Range:** Distance from yearly high/low

```python
def _find_key_levels(self, df: pd.DataFrame) -> Dict:
    # Pivot Points (Traditional)
    prev_high = df['high'].iloc[-2]
    prev_low = df['low'].iloc[-2]
    prev_close = df['close'].iloc[-2]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    
    resistance_1 = 2 * pivot - prev_low
    resistance_2 = pivot + (prev_high - prev_low)
    support_1 = 2 * pivot - prev_high
    support_2 = pivot - (prev_high - prev_low)
    
    # 52-week position
    high_52w = df['high'].max()
    low_52w = df['low'].min()
    current = df['close'].iloc[-1]
    
    position_pct = ((current - low_52w) / (high_52w - low_52w)) * 100
    
    return {
        "pivot": pivot,
        "resistance": {"r1": resistance_1, "r2": resistance_2},
        "support": {"s1": support_1, "s2": support_2},
        "52week": {"high": high_52w, "low": low_52w, "position_pct": position_pct}
    }
```

### 4.6 Relative Strength

**Purpose:** Compare stock performance against benchmark (Nifty 50).

**Why it matters:** Stocks that outperform the index during uptrends tend to continue outperforming.

```python
def _calculate_relative_strength(self, symbol: str, df: pd.DataFrame) -> Dict:
    # Stock returns
    stock_return_1m = ((close.iloc[-1] / close.iloc[-22]) - 1) * 100
    stock_return_3m = ((close.iloc[-1] / close.iloc[-66]) - 1) * 100
    
    # Nifty returns
    nifty_return_1m = ...
    nifty_return_3m = ...
    
    # Alpha = Stock return - Index return
    alpha_1m = stock_return_1m - nifty_return_1m
    alpha_3m = stock_return_3m - nifty_return_3m
    
    # RS Score (0-100)
    rs_score = 50 + (alpha_1m * 2) + (alpha_3m * 1)
    
    if rs_score > 65:
        vs_nifty = "OUTPERFORMING"
    elif rs_score < 35:
        vs_nifty = "UNDERPERFORMING"
    else:
        vs_nifty = "IN-LINE"
    
    return {
        "rs_score": rs_score,
        "vs_nifty": vs_nifty,
        "alpha_1m": alpha_1m,
        "alpha_3m": alpha_3m
    }
```

---

## 5. AI/ML Components

### 5.1 LLM Reasoner

**Purpose:** Provide natural language insights and personalized recommendations.

**Architecture:**

```
User Query → Context Building → LLM API Call → Response Parsing → Display
                  │
                  ├── Portfolio Context
                  ├── Stock Analysis Data
                  └── Market Conditions
```

**Supported Providers:**

| Provider | Model | Cost | Speed |
|----------|-------|------|-------|
| Groq | Llama 3.1 8B | FREE | Fast |
| Ollama | Local Llama | FREE | Local |
| Gemini | gemini-pro | FREE tier | Medium |
| OpenAI | GPT-4 | Paid | Premium |

**Implementation:**

```python
class LLMReasoner:
    def __init__(self, provider: str = "groq"):
        if provider == "groq":
            self.client = groq.Groq(api_key=GROQ_API_KEY)
            self.model = "llama-3.1-8b-instant"
    
    def get_personalized_signal(self, symbol: str, stock_data: Dict, portfolio: Dict) -> str:
        """
        Generate personalized recommendation based on:
        1. Current stock analysis
        2. User's portfolio (existing holdings, P&L, win rate)
        3. Available cash
        """
        
        prompt = f"""
        STOCK: {symbol}
        Analysis: {json.dumps(stock_data)}
        
        USER'S PORTFOLIO:
        - Holdings: {portfolio['holdings']}
        - Win Rate: {portfolio['win_rate']}
        - Cash: {portfolio['cash']}
        
        Provide personalized BUY/SELL/HOLD recommendation with:
        1. Exact entry price
        2. Stop loss level
        3. Target prices
        4. Position size recommendation
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        
        return response.choices[0].message.content
```

### 5.2 Sentiment Analysis

**Purpose:** Analyze news sentiment for market context.

**Approach:** VADER (Valence Aware Dictionary and sEntiment Reasoner)

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze_headlines(self, headlines: List[str]) -> Dict:
        scores = []
        
        for headline in headlines:
            score = self.analyzer.polarity_scores(headline)
            scores.append(score['compound'])
        
        avg_sentiment = sum(scores) / len(scores) if scores else 0
        
        if avg_sentiment > 0.1:
            sentiment = "BULLISH"
        elif avg_sentiment < -0.1:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"
        
        return {
            "sentiment": sentiment,
            "score": avg_sentiment,
            "headlines_analyzed": len(headlines)
        }
```

---

## 6. Signal Generation System

### Scoring System

Each analysis component generates a score (0-100). Final score is weighted average.

```python
def _calculate_scores(self, trend, momentum, volume, patterns, strength) -> Dict:
    
    # Weights for swing trading (trend is most important)
    weights = {
        "trend": 0.30,
        "momentum": 0.25,
        "volume": 0.20,
        "patterns": 0.15,
        "relative_strength": 0.10,
    }
    
    overall = (
        trend_score * 0.30 +
        momentum_score * 0.25 +
        volume_score * 0.20 +
        pattern_score * 0.15 +
        rs_score * 0.10
    )
    
    # Bonus for alignment
    if trend["aligned"] and momentum["condition"] == "BULLISH":
        overall += 5
    
    if volume["smart_money"] == "ACCUMULATING":
        overall += 5
    
    return {"overall": min(100, overall)}
```

### Signal Thresholds

| Score | Signal | Action |
|-------|--------|--------|
| ≥ 75 | STRONG_BUY | Enter immediately |
| 60-74 | BUY | Enter at support |
| 41-59 | HOLD | Wait for clarity |
| 26-40 | SELL | Exit at resistance |
| ≤ 25 | STRONG_SELL | Exit immediately |

### Entry, Stop Loss, Target Calculation

```python
def _generate_signal(self, scores, current_price, levels, df) -> Dict:
    # ATR for volatility-based stops
    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
    atr_value = atr.average_true_range().iloc[-1]
    
    if action == "BUY":
        entry = current_price
        
        # Stop Loss: Max of (2 ATR below price, just below support)
        stop_loss = max(
            current_price - (2 * atr_value),
            levels["support"]["nearest"] * 0.99
        )
        
        # Targets based on ATR multiples
        target_1 = current_price + (2 * atr_value)  # 1:1 R:R
        target_2 = current_price + (3 * atr_value)  # 1:1.5 R:R
        target_3 = levels["resistance"]["nearest"]
    
    # Risk-Reward Ratio
    risk = current_price - stop_loss
    reward = target_1 - current_price
    risk_reward = reward / risk
    
    return {
        "signal": "BUY" if scores["overall"] >= 60 else "HOLD",
        "entry_price": entry,
        "stop_loss": stop_loss,
        "targets": {"target_1": target_1, "target_2": target_2, "target_3": target_3},
        "risk_reward_ratio": risk_reward,
        "confidence": scores["overall"]
    }
```

---

## 7. Risk Management

### Position Sizing (2% Rule)

**Principle:** Never risk more than 2% of capital on a single trade.

```python
def calculate_position_size(capital: float, entry: float, stop_loss: float) -> Dict:
    risk_per_share = abs(entry - stop_loss)
    max_risk = capital * 0.02  # 2% of capital
    
    shares = int(max_risk / risk_per_share)
    position_value = shares * entry
    
    return {
        "recommended_shares": shares,
        "position_value": position_value,
        "max_loss": max_risk,  # Maximum you can lose
        "position_pct": (position_value / capital) * 100
    }
```

**Example:**
- Capital: ₹1,00,000
- Entry: ₹100
- Stop: ₹95
- Risk per share: ₹5
- Max risk (2%): ₹2,000
- Position: 2000/5 = 400 shares @ ₹40,000

### Stop Loss Types

| Type | Description | When to Use |
|------|-------------|-------------|
| ATR-based | 2x ATR below entry | Trending markets |
| Support-based | Just below key support | Range-bound |
| Percentage | Fixed % (e.g., 7%) | Simplicity |
| Trailing | Moves up as price rises | Momentum trades |

---

## 8. Portfolio Management

### SmartPortfolio Class

```python
class SmartPortfolio:
    """
    Complete portfolio tracking with:
    - Live P&L calculation
    - Trade history
    - Performance statistics
    - Cloud synchronization
    """
    
    def buy(self, symbol, quantity, price, stop_loss, target):
        """Record a buy transaction."""
        # Update holdings (average up/down if existing)
        # Deduct from cash
        # Record in trade history
        # Sync to cloud
    
    def sell(self, symbol, quantity, price):
        """Record a sell transaction."""
        # Calculate P&L
        # Update holdings
        # Add to cash
        # Determine WIN/LOSS
        # Update statistics
    
    def get_live_portfolio(self):
        """Get real-time P&L for all holdings."""
        for holding in holdings:
            current_price = fetch_live_price(holding.symbol)
            pnl = (current_price - holding.avg_price) * holding.quantity
            pnl_pct = (current_price / holding.avg_price - 1) * 100
            
            # Check alerts
            if current_price <= holding.stop_loss:
                alerts.append("STOP LOSS HIT")
            if current_price >= holding.target:
                alerts.append("TARGET HIT")
    
    def get_performance_stats(self):
        """Calculate trading statistics."""
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        
        return {
            "total_trades": len(trades),
            "win_rate": len(wins) / len(trades) * 100,
            "avg_win": sum(w.pnl for w in wins) / len(wins),
            "avg_loss": sum(l.pnl for l in losses) / len(losses),
            "profit_factor": total_profit / abs(total_loss),
            "avg_holding_days": sum(t.holding_days for t in trades) / len(trades)
        }
```

### Cloud Sync (Supabase)

```python
class CloudStorage:
    """
    Hybrid storage: Cloud + Local fallback.
    Data syncs across devices automatically.
    """
    
    def __init__(self):
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    def save_portfolio(self, data):
        # Save to cloud
        self.supabase.table("portfolio").upsert({
            "user_id": "default",
            "data": data,
            "updated_at": datetime.now()
        }).execute()
        
        # Also save locally as backup
        self._save_local("portfolio.json", data)
    
    def load_portfolio(self):
        # Try cloud first
        result = self.supabase.table("portfolio").select("*").single().execute()
        if result.data:
            return result.data["data"]
        
        # Fallback to local
        return self._load_local("portfolio.json")
```

---

## 9. Cloud Infrastructure

### Supabase Database Schema

```sql
-- Portfolio holdings
CREATE TABLE portfolio (
    id SERIAL PRIMARY KEY,
    user_id TEXT UNIQUE,
    data JSONB NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Trade history
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    user_id TEXT,
    trade_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Daily snapshots (for equity curve)
CREATE TABLE snapshots (
    id SERIAL PRIMARY KEY,
    user_id TEXT,
    date TEXT NOT NULL,
    snapshot_data JSONB NOT NULL,
    UNIQUE(user_id, date)
);
```

### Deployment Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   GitHub Repo   │────▶│ Streamlit Cloud │────▶│   User Browser  │
│   (Source)      │     │   (Hosting)     │     │   (Phone/PC)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │    Supabase     │
                        │   (Database)    │
                        └─────────────────┘
```

---

## 10. Frontend (UI/UX)

### Streamlit Component Structure

```python
# app.py structure

# 1. Page Config
st.set_page_config(layout="wide", page_title="Trading Agent")

# 2. Custom CSS (Dark theme, animations)
st.markdown("<style>...</style>", unsafe_allow_html=True)

# 3. Authentication
if not check_password():
    st.stop()

# 4. Sidebar Navigation
page = st.sidebar.radio("Navigate", ["Dashboard", "Research", "Scanner", ...])

# 5. Page Routing
if page == "Dashboard":
    render_dashboard()
elif page == "Research":
    render_research()
# ...
```

### Key UI Components

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| Glass Cards | Modern container look | Custom CSS with backdrop-filter |
| Signal Cards | BUY/SELL visual | Gradient backgrounds |
| Progress Bars | Score visualization | CSS div with dynamic width |
| Candlestick Chart | Price visualization | Plotly.go.Candlestick |
| Metrics | Key numbers | st.metric() |
| Chat Interface | AI conversation | Session state + bubbles |

---

## 11. API Design

### Internal APIs (Function Signatures)

```python
# ProAnalyzer
def full_analysis(symbol: str) -> Dict[str, Any]
    """Returns complete analysis with all components."""

# MoneyMaker (Scanner)
def scan_for_opportunities(
    universe: List[str],
    min_score: int = 65,
    min_rr: float = 1.5,
    max_results: int = 10
) -> Dict[str, Any]
    """Returns list of high-probability opportunities."""

def find_breakouts(universe: List[str]) -> List[Dict]
    """Finds stocks breaking out of consolidation."""

def find_reversals(universe: List[str]) -> List[Dict]
    """Finds oversold stocks with bullish divergence."""

def find_momentum_leaders(universe: List[str]) -> List[Dict]
    """Finds stocks outperforming the index."""

# SmartPortfolio
def buy(symbol, quantity, price, stop_loss, target) -> Dict
def sell(symbol, quantity, price) -> Dict
def get_live_portfolio() -> Dict
def get_context_for_ai() -> Dict
```

### External APIs Used

| API | Purpose | Rate Limit | Cost |
|-----|---------|------------|------|
| Yahoo Finance (yfinance) | Price data | ~2000/hour | Free |
| Groq | LLM inference | 30 req/min | Free |
| Supabase | Database | Generous | Free tier |

---

## 12. Flowcharts

### 12.1 Complete System Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER OPENS APP                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │  Password Protected?  │
                        └───────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
             [Enter Password]               [Access Granted]
                    │                               │
                    ▼                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          MAIN DASHBOARD                                  │
│                                                                         │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│   │Dashboard │  │ Research │  │ Scanner  │  │Portfolio │  │AI Advisor│ │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘ │
└────────┼─────────────┼─────────────┼─────────────┼─────────────┼───────┘
         │             │             │             │             │
         ▼             ▼             ▼             ▼             ▼
    [Market      [Symbol        [Scan        [View/Add     [Ask AI]
     Overview]    Analysis]     Market]       Trades]
```

### 12.2 Research Flow

```
┌──────────────────┐
│  User enters     │
│  stock symbol    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌──────────────────┐
│  Fetch Price     │────▶│  Check Cache     │
│  Data (Yahoo)    │     │  (5 min TTL)     │
└────────┬─────────┘     └──────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│            ProAnalyzer.full_analysis()       │
│                                              │
│  ┌─────────────┐  ┌─────────────┐            │
│  │   Trend     │  │  Momentum   │            │
│  │  Analysis   │  │  Analysis   │            │
│  └──────┬──────┘  └──────┬──────┘            │
│         │                │                    │
│  ┌──────┴──────┐  ┌──────┴──────┐            │
│  │   Volume    │  │  Patterns   │            │
│  │  Analysis   │  │ Recognition │            │
│  └──────┬──────┘  └──────┬──────┘            │
│         │                │                    │
│  ┌──────┴──────┐  ┌──────┴──────┐            │
│  │  Relative   │  │    Key      │            │
│  │  Strength   │  │   Levels    │            │
│  └──────┬──────┘  └──────┬──────┘            │
│         │                │                    │
│         └───────┬────────┘                    │
│                 ▼                             │
│         ┌─────────────┐                       │
│         │   Scoring   │                       │
│         │   Engine    │                       │
│         └──────┬──────┘                       │
│                │                              │
│                ▼                              │
│         ┌─────────────┐                       │
│         │   Signal    │                       │
│         │ Generation  │                       │
│         └──────┬──────┘                       │
└────────────────┼─────────────────────────────┘
                 │
                 ▼
         ┌─────────────┐
         │  Display    │
         │  Results    │
         └─────────────┘
```

### 12.3 Signal Generation Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    SIGNAL GENERATION PIPELINE                     │
└──────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────┐
                    │   Input: Analysis   │
                    │   Components        │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Trend Score     │  │ Momentum Score  │  │ Volume Score    │
│ Weight: 30%     │  │ Weight: 25%     │  │ Weight: 20%     │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         └─────────────────┬──┴────────────────────┘
                           ▼
                    ┌─────────────────┐
                    │ Pattern Score   │
                    │ Weight: 15%     │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │ RS Score        │
                    │ Weight: 10%     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  OVERALL SCORE  │◄── Bonus for alignment
                    │    (0-100)      │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         [Score ≥ 75]  [Score 60-74]  [Score < 60]
              │              │              │
              ▼              ▼              ▼
         STRONG_BUY        BUY          HOLD/SELL
              │              │              │
              └──────────────┼──────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Calculate:      │
                    │ • Entry Price   │
                    │ • Stop Loss     │
                    │ • Targets       │
                    │ • Position Size │
                    └─────────────────┘
```

### 12.4 Portfolio Tracking Flow

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  BUY Order  │─────▶│  Record in  │─────▶│  Sync to    │
│             │      │  Portfolio  │      │  Cloud      │
└─────────────┘      └──────┬──────┘      └─────────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │  Holdings   │
                     │  Updated    │
                     └──────┬──────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │  Live P&L   │    │   Alerts    │    │  Signals    │
  │ Calculation │    │  (SL/Target)│    │ for Holdings│
  └─────────────┘    └─────────────┘    └─────────────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │  Dashboard  │
                     │   Display   │
                     └─────────────┘
```

---

## 13. Algorithms & Logic

### 13.1 EMA Calculation

```
EMA(today) = (Price(today) × k) + (EMA(yesterday) × (1 − k))

Where: k = 2 / (N + 1)
N = number of periods

For EMA 21: k = 2 / (21 + 1) = 0.0909
```

### 13.2 RSI Calculation

```
RS = Average Gain / Average Loss (over N periods)
RSI = 100 - (100 / (1 + RS))

N = typically 14 days

Interpretation:
- RSI > 70: Overbought (potential reversal down)
- RSI < 30: Oversold (potential reversal up)
```

### 13.3 MACD Calculation

```
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(9) of MACD Line
Histogram = MACD Line - Signal Line

Buy Signal: MACD crosses above Signal Line
Sell Signal: MACD crosses below Signal Line
```

### 13.4 ATR (Average True Range)

```
True Range = MAX(
    High - Low,
    |High - Previous Close|,
    |Low - Previous Close|
)

ATR = Moving Average of True Range (typically 14 periods)

Usage: Volatility-based stop losses
Stop = Entry - (2 × ATR)
```

### 13.5 Risk-Reward Ratio

```
Risk = Entry Price - Stop Loss
Reward = Target Price - Entry Price
R:R Ratio = Reward / Risk

Example:
Entry: ₹100, Stop: ₹95, Target: ₹115
Risk = ₹5, Reward = ₹15
R:R = 15/5 = 3:1 (Good trade!)

Rule: Only take trades with R:R > 1.5
```

### 13.6 Position Sizing (Kelly Criterion Simplified)

```
Position Size = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)
                ─────────────────────────────────────────────
                              Avg Win

Simplified 2% Rule:
Max Position = (Capital × 0.02) / Risk per Share
```

### 13.7 Win Rate Calculation

```
Win Rate = (Number of Winning Trades / Total Trades) × 100

Profit Factor = Total Profit / Total Loss

A Profit Factor > 1.5 indicates a good trading system.
```

---

## 14. Trading Terms Glossary

### Price Action Terms

| Term | Definition |
|------|------------|
| **OHLC** | Open, High, Low, Close - the four prices for each trading period |
| **Candlestick** | Visual representation of OHLC data |
| **Support** | Price level where buying pressure exceeds selling |
| **Resistance** | Price level where selling pressure exceeds buying |
| **Breakout** | Price moving above resistance with volume |
| **Breakdown** | Price moving below support with volume |
| **Pivot Points** | Calculated support/resistance levels |
| **52-Week High/Low** | Highest/lowest price in past year |
| **Gap Up/Down** | Price opens significantly above/below previous close |

### Trend Terms

| Term | Definition |
|------|------------|
| **Uptrend** | Series of higher highs and higher lows |
| **Downtrend** | Series of lower highs and lower lows |
| **Sideways/Range** | Price moving between support and resistance |
| **EMA (Exponential Moving Average)** | Weighted average giving more importance to recent prices |
| **Golden Cross** | Short-term MA crosses above long-term MA (bullish) |
| **Death Cross** | Short-term MA crosses below long-term MA (bearish) |
| **ADX (Average Directional Index)** | Measures trend strength (not direction) |

### Momentum Terms

| Term | Definition |
|------|------------|
| **RSI (Relative Strength Index)** | Momentum oscillator (0-100) |
| **Overbought** | RSI > 70, price may be due for correction |
| **Oversold** | RSI < 30, price may be due for bounce |
| **MACD** | Moving Average Convergence Divergence |
| **Divergence** | Price and indicator moving in opposite directions |
| **Stochastic** | Compares closing price to price range |
| **ROC (Rate of Change)** | Percentage price change over N periods |

### Volume Terms

| Term | Definition |
|------|------------|
| **Volume** | Number of shares traded |
| **OBV (On-Balance Volume)** | Cumulative volume indicator |
| **Volume Spike** | Unusually high volume (> 2x average) |
| **Accumulation** | Smart money buying (up days with high volume) |
| **Distribution** | Smart money selling (down days with high volume) |
| **Confirmation** | Volume should confirm price moves |

### Pattern Terms

| Term | Definition |
|------|------------|
| **Engulfing** | Candle body completely covers previous candle |
| **Hammer** | Small body with long lower wick (bullish reversal) |
| **Shooting Star** | Small body with long upper wick (bearish reversal) |
| **Doji** | Open ≈ Close, indicates indecision |
| **Consolidation** | Tight price range, low volatility |
| **Squeeze** | Volatility contraction before expansion |

### Risk Management Terms

| Term | Definition |
|------|------------|
| **Stop Loss** | Order to sell if price falls to a certain level |
| **Take Profit** | Order to sell if price rises to target level |
| **Risk-Reward Ratio** | Potential profit vs potential loss |
| **Position Sizing** | How many shares to buy based on risk |
| **2% Rule** | Never risk more than 2% of capital on one trade |
| **Trailing Stop** | Stop that moves up as price rises |
| **ATR Stop** | Stop based on volatility (usually 2× ATR) |

### Trading Terms

| Term | Definition |
|------|------------|
| **Swing Trading** | Holding positions for days to weeks |
| **Day Trading** | Buying and selling within same day |
| **Entry** | Price at which you buy |
| **Exit** | Price at which you sell |
| **Win Rate** | Percentage of profitable trades |
| **Profit Factor** | Gross profit / Gross loss |
| **Drawdown** | Peak-to-trough decline in portfolio |
| **Alpha** | Returns above benchmark (Nifty) |
| **Beta** | Volatility relative to market |

### Market Terms

| Term | Definition |
|------|------------|
| **Bull Market** | Rising prices, optimistic sentiment |
| **Bear Market** | Falling prices, pessimistic sentiment |
| **FII** | Foreign Institutional Investors |
| **DII** | Domestic Institutional Investors |
| **Bulk Deal** | Trade > 0.5% of company shares |
| **Insider Trading** | Trading by company insiders |
| **Market Cap** | Company value = Share Price × Total Shares |
| **Nifty 50** | Index of 50 largest Indian companies |
| **Sensex** | Index of 30 largest Indian companies |

### Technical Indicators Summary

| Indicator | Type | Best For |
|-----------|------|----------|
| EMA | Trend | Direction |
| RSI | Momentum | Overbought/Oversold |
| MACD | Trend + Momentum | Trend changes |
| ADX | Trend | Trend strength |
| Stochastic | Momentum | Reversals |
| OBV | Volume | Accumulation/Distribution |
| ATR | Volatility | Stop loss sizing |
| Bollinger Bands | Volatility | Squeeze breakouts |

---

## Summary

This project demonstrates expertise in:

### Software Engineering
- Clean architecture with separation of concerns
- Modular design with reusable components
- API design and integration
- Error handling and logging
- Cloud deployment and DevOps

### Data Engineering
- ETL pipelines for financial data
- Caching strategies
- Real-time data processing
- Database design (PostgreSQL/Supabase)

### Machine Learning / AI
- Feature engineering for financial data
- Scoring and ranking algorithms
- LLM integration for NLP
- Sentiment analysis

### Domain Knowledge
- Technical analysis indicators
- Risk management principles
- Portfolio management
- Trading psychology

### Frontend Development
- Responsive web design
- Interactive visualizations
- User authentication
- Session management

---

**Built with ❤️ for profitable trading**
