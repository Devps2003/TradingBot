# 🇮🇳 Indian Market Trading Research Agent

An advanced, AI-powered market research and trading signal system specifically designed for the Indian stock market (NSE/BSE). This system functions like a top-tier institutional research desk combined with quantitative analysis capabilities.

## 🎯 Target User

- **Trading Style**: Swing Trader
- **Holding Period**: 1-2 weeks
- **Markets**: NSE/BSE (Indian Stock Market)

## ✨ Features

### 📊 Data Fetching
- **Price Data**: Real-time and historical OHLCV from Yahoo Finance and NSE
- **Fundamental Data**: Financial ratios, quarterly results, shareholding patterns
- **News & Sentiment**: Multi-source news aggregation with NLP sentiment analysis
- **FII/DII Activity**: Institutional investor flows
- **Bulk/Block Deals**: Large transaction tracking
- **Insider Trading**: SAST filings and promoter transactions
- **Global Markets**: US markets, commodities, currencies, VIX

### 📈 Analysis Modules
- **Technical Analysis**: 50+ indicators including RSI, MACD, Bollinger Bands, Supertrend
- **Pattern Recognition**: Candlestick patterns, chart patterns, breakout detection
- **Fundamental Analysis**: Valuation, quality, growth, and financial health scoring
- **Sentiment Analysis**: FinBERT/VADER-based news sentiment
- **Volume Analysis**: Accumulation/distribution, unusual volume, delivery percentage
- **Market Context**: Regime detection, sector rotation, market breadth

### 🤖 AI-Powered Features
- LLM reasoning for trade explanations (OpenAI/Anthropic/Ollama)
- ML-based pattern prediction
- Intelligent signal generation with confidence scoring
- Risk-adjusted position sizing

### 💼 Portfolio Management
- Track holdings with P&L
- Sector allocation analysis
- Portfolio risk monitoring
- Trade history and performance analytics

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project
cd indian-market-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration (Optional)

Set up API keys for enhanced features:

```bash
# Environment variables (recommended)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Or edit config/api_keys.py
```

### 3. Run the Agent

```bash
# Interactive mode
python run.py

# CLI commands
python run.py morning          # Morning briefing
python run.py portfolio        # Portfolio analysis
python run.py research RELIANCE # Deep research on stock
python run.py scan             # Scan for opportunities
python run.py eod              # End of day summary
python run.py quote INFY       # Quick quote
```

## 📋 CLI Commands

| Command | Description |
|---------|-------------|
| `morning` | Generate morning market briefing |
| `portfolio` | Analyze current portfolio |
| `research SYMBOL` | Deep research on a stock |
| `scan` | Scan market for opportunities |
| `eod` | End of day summary |
| `quote SYMBOL` | Quick price quote |
| `add SYMBOL QTY PRICE` | Add position to portfolio |
| `remove SYMBOL QTY PRICE` | Close position |
| `performance` | Trading performance analysis |

## 📁 Project Structure

```
indian-market-agent/
├── config/
│   ├── settings.py           # All configuration
│   ├── api_keys.py           # API keys (gitignored)
│   └── trading_rules.py      # Custom trading rules
├── data/
│   ├── portfolio.json        # User's portfolio
│   ├── watchlist.json        # Watchlist
│   └── trade_history.json    # Trade log
├── src/
│   ├── agent.py              # Main orchestrator
│   ├── main.py               # CLI interface
│   ├── data_fetchers/        # Data fetching modules
│   ├── analyzers/            # Analysis modules
│   ├── signals/              # Signal generation
│   ├── ai_layer/             # AI/ML components
│   ├── portfolio/            # Portfolio management
│   └── utils/                # Utilities
├── reports/                  # Generated reports
├── run.py                    # Runner script
└── requirements.txt          # Dependencies
```

## 📊 Signal System

### Signal Types
- **STRONG_BUY** (Score ≥ 80): High conviction buy
- **BUY** (Score ≥ 65): Buy opportunity
- **HOLD** (Score 35-65): Neutral
- **SELL** (Score < 35): Sell signal
- **STRONG_SELL** (Score < 20): Strong sell

### Signal Weights
- Technical Analysis: 35%
- Pattern Recognition: 15%
- Fundamental Analysis: 20%
- Sentiment Analysis: 15%
- Volume Analysis: 10%
- Market Context: 5%

## ⚙️ Configuration

### Trading Rules (`config/trading_rules.py`)

Customize your trading style:
- Entry/exit rules
- Position sizing method
- Risk management parameters
- Sector preferences
- Blacklist/watchlist

### Settings (`config/settings.py`)

Configure:
- Data caching duration
- Technical indicator parameters
- Signal thresholds
- Risk limits
- API rate limits

## 🔧 Requirements

- Python 3.11+
- Internet connection for data fetching
- Optional: OpenAI/Anthropic API key for AI features

### Key Dependencies
- `yfinance`: Price data
- `pandas-ta`: Technical indicators
- `beautifulsoup4`: Web scraping
- `transformers`: Sentiment analysis (optional)
- `rich`: Beautiful CLI output
- `typer`: CLI framework

## ⚠️ Disclaimer

**This is a DECISION SUPPORT system, not an automated trading bot.**

- The system provides analysis and suggestions only
- Final trading decisions are your responsibility
- Past performance does not guarantee future results
- Always do your own research
- Manage your risk appropriately

## 🔮 Future Enhancements

- [ ] Web UI with Streamlit
- [ ] Backtesting engine
- [ ] Options chain analysis
- [ ] Telegram/Discord alerts
- [ ] Multi-timeframe analysis
- [ ] Portfolio optimization
- [ ] Paper trading mode

## 📞 Support

For issues or suggestions, please open an issue on GitHub.

---

**May your trades be profitable! 📈**
