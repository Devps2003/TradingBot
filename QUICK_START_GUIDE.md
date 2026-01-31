# 🚀 Quick Start Guide: Daily Trading Workflow

## Important Note
This is a **RESEARCH & ANALYSIS** tool - it helps you make better decisions.  
**You execute trades manually on Zerodha Kite.**

---

## 📋 ONE-TIME SETUP (Do This First)

### Step 1: Open Terminal
```bash
cd /Users/devps/Desktop/TradingBot/indian-market-agent
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

Or install minimal packages:
```bash
pip install yfinance pandas pandas-ta requests beautifulsoup4 rich typer vaderSentiment pytz
```

### Step 4: (Optional) Add API Keys for AI Features
Edit `config/api_keys.py`:
```python
OPENAI_API_KEY = "your-openai-key"  # For AI explanations
```

### Step 5: Configure Your Capital
Edit `data/portfolio.json`:
```json
{
  "holdings": [],
  "cash": 100000,
  "total_capital": 500000
}
```
Change `total_capital` to your actual trading capital.

---

## 📅 DAILY WORKFLOW

### 🌅 MORNING (8:30 - 9:15 AM) - Before Market Opens

#### 1. Activate Environment & Run Morning Briefing
```bash
cd /Users/devps/Desktop/TradingBot/indian-market-agent
source venv/bin/activate
python run.py morning
```

**What you'll see:**
- Global market cues (US markets, SGX Nifty)
- FII/DII activity from yesterday
- Market sentiment
- Your portfolio status
- Any alerts for your holdings

#### 2. Check for Opportunities
```bash
python run.py scan
```

**What you'll see:**
- Top 10 BUY signals from Nifty 50
- Entry price, stop loss, targets
- Confidence scores

#### 3. Research Specific Stocks
If you see an interesting opportunity:
```bash
python run.py research RELIANCE
python run.py research HDFCBANK
```

**What you'll see:**
- Complete technical analysis
- Pattern detection
- Fundamental scores
- News sentiment
- AI-generated trade idea

---

### 📈 DURING MARKET HOURS (9:15 AM - 3:30 PM)

#### Quick Quote Check
```bash
python run.py quote RELIANCE
```

#### If You Want to Buy a Stock
1. First research it:
```bash
python run.py research TATAMOTORS
```

2. If signal is BUY/STRONG_BUY, note down:
   - Entry price
   - Stop Loss
   - Target 1 & Target 2

3. **Go to Zerodha Kite** and place the order manually

4. After buying, log it in the system:
```bash
python run.py add TATAMOTORS 50 650 --stop 620 --target 720
```
(This means: 50 shares at ₹650, stop at ₹620, target ₹720)

#### If You Want to Sell
```bash
python run.py remove TATAMOTORS 50 700
```
(Sold 50 shares at ₹700)

---

### 🌙 EVENING (After 4:00 PM) - Market Closed

#### Run End-of-Day Analysis
```bash
python run.py eod
```

**What you'll see:**
- How market performed today
- Sector performance
- Your portfolio P&L
- Tomorrow's watchlist

#### Check Your Portfolio
```bash
python run.py portfolio
```

#### Check Your Trading Performance
```bash
python run.py performance
```

---

## 🎯 TYPICAL DAILY SCHEDULE

| Time | Action | Command |
|------|--------|---------|
| 8:30 AM | Morning briefing | `python run.py morning` |
| 8:45 AM | Scan for opportunities | `python run.py scan` |
| 9:00 AM | Research top picks | `python run.py research SYMBOL` |
| 9:15 AM | Market opens - Place orders on Zerodha | - |
| During day | Quick quotes | `python run.py quote SYMBOL` |
| After trade | Log trades | `python run.py add/remove` |
| 4:00 PM | EOD analysis | `python run.py eod` |
| 4:15 PM | Portfolio review | `python run.py portfolio` |

---

## 📝 ADDING STOCKS TO WATCHLIST

```bash
# Interactive mode
python run.py

# Then type:
>>> watchlist add BAJFINANCE
>>> watchlist add ASIANPAINT
```

Or edit directly: `data/watchlist.json`

---

## 🔍 UNDERSTANDING SIGNALS

| Signal | Score | Meaning | Action |
|--------|-------|---------|--------|
| STRONG_BUY | 80-100 | Excellent setup | Buy with full position |
| BUY | 65-79 | Good opportunity | Buy with smaller position |
| HOLD | 35-64 | Neutral | Don't enter new positions |
| SELL | 20-34 | Weakness | Exit or reduce |
| STRONG_SELL | 0-19 | Avoid | Exit immediately |

---

## 💰 POSITION SIZING GUIDE

The system suggests position sizes based on:
- Your total capital
- Risk per trade (2% default)
- Stop loss distance
- Confidence level

**Example:**
- Capital: ₹5,00,000
- Risk per trade: 2% = ₹10,000
- Stock price: ₹1,000
- Stop loss: ₹950 (₹50 risk per share)
- Position size: ₹10,000 ÷ ₹50 = 200 shares max

---

## ⚠️ IMPORTANT RULES

1. **Never skip the morning briefing** - Know global cues before trading
2. **Always use stop losses** - The system calculates them for you
3. **Don't overtrade** - Wait for high-confidence signals (70%+)
4. **Log every trade** - Helps track performance and improve
5. **Follow the system** - Don't let emotions override signals

---

## 🆘 TROUBLESHOOTING

### "Module not found" error
```bash
source venv/bin/activate
pip install <missing-module>
```

### "No data" for a stock
- Check if the symbol is correct (use NSE symbols)
- The stock might be newly listed

### Slow performance
- First run is always slow (downloads data)
- Subsequent runs use cached data

---

## 📱 QUICK REFERENCE COMMANDS

```bash
# Morning routine
python run.py morning
python run.py scan

# Research
python run.py research SYMBOL
python run.py quote SYMBOL

# Portfolio management
python run.py portfolio
python run.py add SYMBOL QTY PRICE
python run.py remove SYMBOL QTY PRICE

# Analysis
python run.py eod
python run.py performance

# Interactive mode (recommended for beginners)
python run.py
```

---

## 🎓 LEARNING PATH

**Week 1:** Just run morning briefing and observe
**Week 2:** Start paper trading (note trades without real money)
**Week 3:** Small positions with high-confidence signals only
**Week 4:** Normal trading with proper position sizing

---

**Remember: This tool gives you an edge, but the market always has risks. Never invest more than you can afford to lose.**

Good luck with your trading journey! 📈
