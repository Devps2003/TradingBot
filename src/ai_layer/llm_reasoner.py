"""
LLM Reasoning Module with FREE AI Options.

Supports multiple LLM providers (prioritizing FREE options):
1. Groq (FREE, recommended) - Uses Llama 3.1 70B
2. Ollama (FREE, local) - Runs models locally
3. Google Gemini (FREE tier available)
4. OpenAI / Anthropic (PAID options)
"""

import os
import json
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import (
    LLM_PROVIDER, 
    GROQ_API_KEY, GROQ_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    OPENAI_API_KEY, ANTHROPIC_API_KEY,
)
from src.utils.logger import LoggerMixin


# System prompt for trading AI
TRADING_SYSTEM_PROMPT = """You are an expert Indian stock market analyst and swing trader with 15+ years of experience.

Your expertise includes:
- NSE/BSE markets, Nifty 50, sector analysis
- Technical analysis (patterns, indicators, support/resistance)
- Fundamental analysis (P/E, ROE, debt ratios)
- FII/DII flows and their market impact
- ETFs including Gold, Silver, Index ETFs
- Swing trading strategies (1-2 week holding periods)

Style:
- Be direct and actionable
- Give specific price levels when relevant
- Mention both bull and bear scenarios
- Think like a professional trader managing real money

Always consider:
- Current market trend and volatility
- Risk-reward ratios
- Position sizing
- Stop loss levels"""


class LLMReasoner(LoggerMixin):
    """
    LLM-powered reasoning for trading decisions.
    Supports FREE options: Groq, Ollama, Gemini.
    """
    
    def __init__(self, provider: Optional[str] = None):
        """Initialize with specified or default provider."""
        self.provider = provider or LLM_PROVIDER
        self.initialized = False
        self._setup_provider()
    
    def _setup_provider(self):
        """Setup the LLM provider."""
        # Try providers in order of preference (free first)
        providers_to_try = [self.provider]
        
        # Add fallbacks
        if self.provider != "groq":
            providers_to_try.append("groq")
        if self.provider != "ollama":
            providers_to_try.append("ollama")
        if self.provider != "gemini":
            providers_to_try.append("gemini")
        
        for prov in providers_to_try:
            if prov == "groq" and GROQ_API_KEY:
                self.provider = "groq"
                self.initialized = True
                return
            elif prov == "gemini" and GEMINI_API_KEY:
                self.provider = "gemini"
                self.initialized = True
                return
            elif prov == "ollama":
                # Check if Ollama is running
                try:
                    resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
                    if resp.status_code == 200:
                        self.provider = "ollama"
                        self.initialized = True
                        return
                except:
                    pass
            elif prov == "openai" and OPENAI_API_KEY:
                self.provider = "openai"
                self.initialized = True
                return
        
        # No LLM available - will use fallback
        self.initialized = False
    
    def _call_groq(self, prompt: str, system: str = None, max_tokens: int = 500) -> str:
        """Call Groq API (FREE, uses Llama 3.1 70B)."""
        # Available Groq models (as of 2024):
        # - llama-3.1-70b-versatile
        # - llama-3.1-8b-instant  
        # - llama3-70b-8192
        # - mixtral-8x7b-32768
        
        models_to_try = [
            "llama-3.1-8b-instant",      # Fast, reliable
            "llama3-70b-8192",           # Larger context
            "mixtral-8x7b-32768",        # Alternative
            GROQ_MODEL,                   # User configured
        ]
        
        for model in models_to_try:
            try:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system or TRADING_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": max_tokens,
                        "temperature": 0.7,
                    },
                    timeout=30,
                )
                
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                elif response.status_code == 400:
                    # Model not available, try next
                    continue
                else:
                    error_msg = response.text[:200] if response.text else "Unknown error"
                    self.logger.warning(f"Groq API error {response.status_code}: {error_msg}")
            except Exception as e:
                self.logger.warning(f"Groq error with {model}: {e}")
                continue
        
        return None
    
    def _call_ollama(self, prompt: str, system: str = None, max_tokens: int = 500) -> str:
        """Call local Ollama API (FREE, runs locally)."""
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": system or TRADING_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
                timeout=60,
            )
            
            if response.status_code == 200:
                return response.json()["message"]["content"]
            return None
        except Exception as e:
            self.logger.warning(f"Ollama error: {e}")
            return None
    
    def _call_gemini(self, prompt: str, system: str = None, max_tokens: int = 500) -> str:
        """Call Google Gemini API (FREE tier available)."""
        try:
            full_prompt = f"{system or TRADING_SYSTEM_PROMPT}\n\n{prompt}"
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}",
                json={
                    "contents": [{"parts": [{"text": full_prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": max_tokens,
                        "temperature": 0.7,
                    },
                },
                timeout=30,
            )
            
            if response.status_code == 200:
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
            return None
        except Exception as e:
            self.logger.warning(f"Gemini error: {e}")
            return None
    
    def _call_openai(self, prompt: str, system: str = None, max_tokens: int = 500) -> str:
        """Call OpenAI API (PAID)."""
        try:
            import openai
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Cheaper option
                messages=[
                    {"role": "system", "content": system or TRADING_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.warning(f"OpenAI error: {e}")
            return None
    
    def chat(self, message: str, context: Optional[Dict] = None) -> str:
        """
        Chat with the AI about stocks/trading.
        
        Args:
            message: User's question or message
            context: Optional context (stock data, portfolio, etc.)
        
        Returns:
            AI response
        """
        # Build context-aware prompt
        if context:
            prompt = f"""
Context:
{json.dumps(context, indent=2, default=str)[:2000]}

User Question: {message}

Please answer based on the context provided. Be specific and actionable."""
        else:
            prompt = message
        
        # Try providers
        response = None
        
        if self.provider == "groq":
            response = self._call_groq(prompt)
        elif self.provider == "ollama":
            response = self._call_ollama(prompt)
        elif self.provider == "gemini":
            response = self._call_gemini(prompt)
        elif self.provider == "openai":
            response = self._call_openai(prompt)
        
        if response:
            return response
        
        # Fallback
        return self._generate_fallback(message, context)
    
    def _generate_fallback(self, message: str, context: Optional[Dict] = None) -> str:
        """Generate fallback response when no LLM available."""
        msg_lower = message.lower()
        
        if context and "signal" in context:
            signal = context.get("signal", "HOLD")
            conf = context.get("confidence", 50)
            symbol = context.get("symbol", "Stock")
            
            if signal == "STRONG_BUY":
                return f"{symbol} shows a strong bullish setup with {conf:.0f}% confidence. Consider entry with proper stop loss."
            elif signal == "BUY":
                return f"{symbol} is showing positive momentum. Wait for pullback or enter with smaller position."
            elif signal == "SELL":
                return f"{symbol} is weak. Consider exiting or avoiding new positions."
            else:
                return f"{symbol} is neutral. No clear signal at the moment."
        
        if "nifty" in msg_lower or "market" in msg_lower:
            return "Check the morning briefing for current market conditions using: python run.py morning"
        
        if "gold" in msg_lower or "silver" in msg_lower or "etf" in msg_lower:
            return "ETFs like GOLDBEES, SILVERBEES, NIFTYBEES are now available. Use: python run.py research GOLDBEES"
        
        return """🤖 AI not configured. To enable FREE AI chat:

1. Get FREE API key from https://console.groq.com
2. Set it: export GROQ_API_KEY="your-key-here"
3. Run the command again

OR install Ollama for local AI: https://ollama.ai"""
    
    def generate_signal_explanation(self, analysis_data: Dict[str, Any]) -> str:
        """Generate explanation for a trading signal."""
        symbol = analysis_data.get("symbol", "Stock")
        signal = analysis_data.get("signal", "HOLD")
        confidence = analysis_data.get("confidence", 50)
        scores = analysis_data.get("scores", {})
        entry = analysis_data.get("entry_price", 0)
        stop = analysis_data.get("stop_loss", 0)
        target = analysis_data.get("target_1", 0)
        
        prompt = f"""
Explain this trading signal in 2-3 sentences:

Stock: {symbol}
Signal: {signal} ({confidence:.0f}% confidence)
Entry: ₹{entry:.2f} | Stop: ₹{stop:.2f} | Target: ₹{target:.2f}

Scores:
- Technical: {scores.get('technical', 50):.0f}/100
- Fundamental: {scores.get('fundamental', 50):.0f}/100  
- Volume: {scores.get('volume', 50):.0f}/100

Give a brief explanation and one actionable recommendation."""
        
        return self.chat(prompt, analysis_data)
    
    def generate_daily_briefing(self, market_data: Dict[str, Any], portfolio: List[Dict]) -> str:
        """Generate morning market briefing."""
        regime = market_data.get("regime", {}).get("regime", "UNKNOWN")
        favorable = market_data.get("favorability", {}).get("favorable_for_longs", True)
        fii_dii = market_data.get("fii_dii", {})
        
        prompt = f"""
Generate a concise morning briefing for today:

Market:
- Regime: {regime}
- Favorable for longs: {favorable}
- FII Sentiment: {fii_dii.get('sentiment', 'NEUTRAL')}

Portfolio: {len(portfolio)} holdings

Provide:
1. One-line market view
2. Key action for today
3. What to watch

Keep it under 100 words."""
        
        return self.chat(prompt, market_data)
    
    def analyze_stock(self, symbol: str, data: Dict[str, Any]) -> str:
        """Deep analysis of a stock with AI using advanced data."""
        # Get advanced analysis
        try:
            from src.analyzers.advanced_analyzer import get_advanced_analysis
            advanced = get_advanced_analysis(symbol)
        except Exception:
            advanced = data
        
        prompt = f"""
You are analyzing {symbol} for a swing trade (holding period: 1-2 weeks).

═══════════════════════════════════════════════════════════════
MULTI-TIMEFRAME ANALYSIS
═══════════════════════════════════════════════════════════════
Daily Trend: {advanced.get('timeframes', {}).get('daily', {}).get('trend', 'N/A')}
Weekly Trend: {advanced.get('timeframes', {}).get('weekly', {}).get('trend', 'N/A')}
Monthly Trend: {advanced.get('timeframes', {}).get('monthly', {}).get('trend', 'N/A')}
Timeframe Alignment: {advanced.get('timeframes', {}).get('alignment', {}).get('status', 'N/A')}

═══════════════════════════════════════════════════════════════
KEY PRICE LEVELS
═══════════════════════════════════════════════════════════════
Current Price: ₹{advanced.get('current_price', 0)}
Resistance 1: ₹{advanced.get('key_levels', {}).get('resistance_1', 'N/A')}
Resistance 2: ₹{advanced.get('key_levels', {}).get('resistance_2', 'N/A')}
Support 1: ₹{advanced.get('key_levels', {}).get('support_1', 'N/A')}
Support 2: ₹{advanced.get('key_levels', {}).get('support_2', 'N/A')}
52W High: ₹{advanced.get('key_levels', {}).get('high_52w', 'N/A')}
52W Low: ₹{advanced.get('key_levels', {}).get('low_52w', 'N/A')}
Position in 52W Range: {advanced.get('key_levels', {}).get('position_in_range', 'N/A')}%

═══════════════════════════════════════════════════════════════
TREND & MOMENTUM
═══════════════════════════════════════════════════════════════
Trend Direction: {advanced.get('trend', {}).get('direction', 'N/A')}
Trend Strength: {advanced.get('trend', {}).get('strength', 'N/A')}%
RSI: {advanced.get('momentum', {}).get('rsi', 'N/A')}
Momentum Score: {advanced.get('momentum', {}).get('momentum_score', 'N/A')}/100
Condition: {advanced.get('momentum', {}).get('condition', 'N/A')}

═══════════════════════════════════════════════════════════════
RELATIVE STRENGTH
═══════════════════════════════════════════════════════════════
RS Score: {advanced.get('relative_strength', {}).get('rs_score', 'N/A')}/100
vs Nifty: {advanced.get('relative_strength', {}).get('vs_market', 'N/A')}
Stock 20D Return: {advanced.get('relative_strength', {}).get('stock_return_20d', 'N/A')}%
Nifty 20D Return: {advanced.get('relative_strength', {}).get('nifty_return_20d', 'N/A')}%

═══════════════════════════════════════════════════════════════
VOLUME & BREAKOUT
═══════════════════════════════════════════════════════════════
Volume vs Avg: {advanced.get('volume', {}).get('ratio_vs_avg', 'N/A')}x
Volume Signal: {advanced.get('volume', {}).get('price_volume_signal', 'N/A')}
Breakout Status: {advanced.get('breakout', {}).get('status', 'N/A')}
Distance to Breakout: {advanced.get('breakout', {}).get('distance_to_breakout', 'N/A')}%

═══════════════════════════════════════════════════════════════
PATTERNS & RISK
═══════════════════════════════════════════════════════════════
Patterns Detected: {', '.join(advanced.get('patterns', {}).get('detected', [])) or 'None'}
Risk Level: {advanced.get('risk', {}).get('risk_level', 'N/A')}
Volatility (20D): {advanced.get('risk', {}).get('volatility_20d', 'N/A')}%
ATR%: {advanced.get('risk', {}).get('atr_percent', 'N/A')}%
Suggested Stop: {advanced.get('risk', {}).get('suggested_stop_pct', 'N/A')}%

═══════════════════════════════════════════════════════════════
OVERALL SCORE: {advanced.get('overall_score', 50)}/100 → {advanced.get('signal_strength', 'HOLD')}
═══════════════════════════════════════════════════════════════

Based on this comprehensive data, provide:

1. **VERDICT**: Clear BUY/SELL/HOLD recommendation with conviction level (high/medium/low)

2. **ENTRY STRATEGY**: 
   - Exact entry price or zone
   - Whether to buy now or wait for pullback
   - Position sizing suggestion (% of portfolio)

3. **EXIT PLAN**:
   - Stop loss level (exact price)
   - Target 1 (first profit booking)
   - Target 2 (full exit)
   - Trailing stop strategy

4. **KEY RISKS**: Top 3 risks that could invalidate this trade

5. **PROBABILITY**: Your assessment of success probability (%) and reasoning

Be direct and specific. Think like a professional trader."""
        
        return self.chat(prompt, advanced)
    
    def predict_price_movement(self, symbol: str, data: Dict[str, Any]) -> str:
        """AI-powered price prediction with advanced analysis."""
        # Get advanced analysis
        try:
            from src.analyzers.advanced_analyzer import get_advanced_analysis
            advanced = get_advanced_analysis(symbol)
        except Exception:
            advanced = data
        
        prompt = f"""
PRICE PREDICTION REQUEST: {symbol}

Current Price: ₹{advanced.get('current_price', 0)}

Technical Summary:
- Trend: {advanced.get('trend', {}).get('direction', 'N/A')} (Strength: {advanced.get('trend', {}).get('strength', 50)}%)
- RSI: {advanced.get('momentum', {}).get('rsi', 50)}
- Timeframes Aligned: {advanced.get('timeframes', {}).get('alignment', {}).get('status', 'N/A')}
- RS vs Nifty: {advanced.get('relative_strength', {}).get('vs_market', 'N/A')}
- Breakout Status: {advanced.get('breakout', {}).get('status', 'N/A')}

Key Levels:
- Resistance: ₹{advanced.get('key_levels', {}).get('resistance_1', 'N/A')}
- Support: ₹{advanced.get('key_levels', {}).get('support_1', 'N/A')}
- 52W High: ₹{advanced.get('key_levels', {}).get('high_52w', 'N/A')}

Recent Performance:
- 1 Week: {advanced.get('statistics', {}).get('returns', {}).get('1w', 'N/A')}%
- 1 Month: {advanced.get('statistics', {}).get('returns', {}).get('1m', 'N/A')}%
- 3 Months: {advanced.get('statistics', {}).get('returns', {}).get('3m', 'N/A')}%

Overall Score: {advanced.get('overall_score', 50)}/100

═══════════════════════════════════════════════════════════════
PREDICTION REQUEST
═══════════════════════════════════════════════════════════════

Provide your price predictions:

📅 **1 WEEK FORECAST**:
- Direction: (Up/Down/Sideways)
- Target Price: ₹___
- Probability: ___%
- Key trigger: ___

📅 **2 WEEK FORECAST**:
- Direction: (Up/Down/Sideways)  
- Target Price: ₹___
- Probability: ___%
- Key trigger: ___

📅 **1 MONTH FORECAST**:
- Direction: (Up/Down/Sideways)
- Target Price: ₹___
- Probability: ___%
- Key trigger: ___

⚠️ **INVALIDATION**: Price below ₹___ would invalidate bullish view

🎯 **BEST CASE SCENARIO**: ₹___ (+__%) if ___
📉 **WORST CASE SCENARIO**: ₹___ (-__%) if ___

Be bold with specific numbers. Acknowledge uncertainty but commit to a view."""
        
        return self.chat(prompt, advanced)
    
    def suggest_portfolio_action(self, portfolio: Dict, market: Dict) -> str:
        """Suggest actions for portfolio based on full context."""
        prompt = f"""
You are my personal trading advisor. Here is my COMPLETE portfolio context:

═══════════════════════════════════════════════════════════════
MY PORTFOLIO
═══════════════════════════════════════════════════════════════
Total Value: ₹{portfolio.get('portfolio_summary', {}).get('total_value', 0):,.0f}
Total Invested: ₹{portfolio.get('portfolio_summary', {}).get('invested', 0):,.0f}
Current P&L: ₹{portfolio.get('portfolio_summary', {}).get('pnl', 0):,.0f} ({portfolio.get('portfolio_summary', {}).get('pnl_pct', 0):+.2f}%)
Cash Available: ₹{portfolio.get('portfolio_summary', {}).get('cash_available', 0):,.0f}

═══════════════════════════════════════════════════════════════
MY HOLDINGS
═══════════════════════════════════════════════════════════════
{json.dumps(portfolio.get('holdings', []), indent=2, default=str)}

═══════════════════════════════════════════════════════════════
MY TRADING HISTORY
═══════════════════════════════════════════════════════════════
Total Trades: {portfolio.get('performance', {}).get('total_trades', 0)}
Win Rate: {portfolio.get('performance', {}).get('win_rate', 0):.1f}%
Net P&L: ₹{portfolio.get('performance', {}).get('net_pnl', 0):,.0f}
Avg Holding: {portfolio.get('performance', {}).get('avg_holding_days', 0):.0f} days

═══════════════════════════════════════════════════════════════
ALERTS
═══════════════════════════════════════════════════════════════
{', '.join(portfolio.get('alerts', [])) or 'No alerts'}

═══════════════════════════════════════════════════════════════
MARKET CONDITIONS
═══════════════════════════════════════════════════════════════
{json.dumps(market, indent=2, default=str)[:500]}

═══════════════════════════════════════════════════════════════

Based on MY specific portfolio, give me:

1. **IMMEDIATE ACTIONS** (What should I do TODAY):
   - Any positions to EXIT (with exact prices)
   - Any positions to ADD to
   - Any stop losses to adjust

2. **PROFIT BOOKING PLAN**:
   - Which winning positions to book profits
   - How much to sell (partial/full)
   - Trail stop strategy

3. **RISK MANAGEMENT**:
   - Total portfolio risk assessment
   - Position sizing recommendations
   - Diversification feedback

4. **OPPORTUNITY COST**:
   - Am I holding dead money?
   - Better opportunities for my capital?

5. **PERSONALIZED TIP** based on my trading history:
   - If my win rate is low, what am I doing wrong?
   - If I'm holding too long/short, suggest changes

Be SPECIFIC to MY portfolio. Use actual prices and symbols."""
        
        return self.chat(prompt, portfolio)
    
    def get_personalized_signal(self, symbol: str, stock_data: Dict, portfolio_context: Dict) -> str:
        """Get signal personalized to user's portfolio context."""
        
        # Check if user already holds this stock
        holdings = portfolio_context.get("holdings", [])
        current_holding = None
        for h in holdings:
            if h.get("symbol") == symbol:
                current_holding = h
                break
        
        holding_context = ""
        if current_holding:
            holding_context = f"""
⚠️ YOU ALREADY OWN THIS STOCK:
- Quantity: {current_holding.get('quantity', 0)} shares
- Avg Price: ₹{current_holding.get('avg_price', 0):.2f}
- Current P&L: {current_holding.get('pnl_pct', 0):+.2f}%
- Holding Days: {current_holding.get('holding_days', 0)}

Consider: Should you ADD, HOLD, or EXIT?
"""
        else:
            cash = portfolio_context.get("portfolio_summary", {}).get("cash_available", 0)
            holding_context = f"""
📝 YOU DON'T OWN THIS STOCK
Cash Available: ₹{cash:,.0f}
Should you BUY? If yes, how much?
"""
        
        prompt = f"""
PERSONALIZED ANALYSIS for {symbol}

{holding_context}

═══════════════════════════════════════════════════════════════
STOCK DATA
═══════════════════════════════════════════════════════════════
{json.dumps(stock_data, indent=2, default=str)[:2500]}

═══════════════════════════════════════════════════════════════
YOUR PORTFOLIO CONTEXT
═══════════════════════════════════════════════════════════════
Total Value: ₹{portfolio_context.get('portfolio_summary', {}).get('total_value', 0):,.0f}
Current Holdings: {portfolio_context.get('portfolio_summary', {}).get('num_holdings', 0)}
Win Rate: {portfolio_context.get('performance', {}).get('win_rate', 0):.0f}%

═══════════════════════════════════════════════════════════════

Give me a PERSONALIZED recommendation:

1. **ACTION**: BUY / ADD / HOLD / REDUCE / EXIT
2. **QUANTITY**: How many shares (based on my cash and risk)
3. **PRICE**: At what price to enter/exit
4. **STOP LOSS**: Where to place stop (exact price)
5. **TARGET**: Profit target (exact price)
6. **POSITION SIZE**: % of my portfolio
7. **REASONING**: Why this makes sense for MY situation

Think about:
- My current exposure
- My win rate (if low, suggest tighter stops)
- My available cash
- Diversification"""
        
        return self.chat(prompt, {"stock": stock_data, "portfolio": portfolio_context})
    
    def answer_question(self, question: str, context: Dict = None) -> str:
        """Answer any trading-related question."""
        return self.chat(question, context)


class SimpleLLMReasoner:
    """Simple fallback when no LLM is available."""
    
    def generate_signal_explanation(self, data: Dict) -> str:
        signal = data.get("signal", "HOLD")
        symbol = data.get("symbol", "Stock")
        conf = data.get("confidence", 50)
        
        explanations = {
            "STRONG_BUY": f"✅ {symbol} shows strong bullish momentum with {conf:.0f}% confidence. Consider entry.",
            "BUY": f"📈 {symbol} is showing positive signals. Watch for entry opportunity.",
            "HOLD": f"⏸️ {symbol} is neutral. Hold existing positions, avoid new entries.",
            "SELL": f"📉 {symbol} shows weakness. Consider reducing exposure.",
            "STRONG_SELL": f"🔴 {symbol} has strong bearish signals. Exit recommended.",
        }
        return explanations.get(signal, f"{symbol}: {signal}")
    
    def generate_daily_briefing(self, market: Dict, portfolio: List) -> str:
        favorable = market.get("favorability", {}).get("favorable_for_longs", True)
        if favorable:
            return "Market conditions are favorable. Look for buying opportunities in strong stocks."
        return "Market is uncertain. Be cautious with new positions. Focus on quality stocks."
    
    def chat(self, message: str, context: Dict = None) -> str:
        return """🤖 AI not configured. Get FREE AI by:

1. Sign up at https://console.groq.com (free)
2. Get API key
3. Run: export GROQ_API_KEY="your-key"
4. Try again!"""
    
    def analyze_stock(self, symbol: str, data: Dict) -> str:
        return self.generate_signal_explanation(data)
    
    def predict_price_movement(self, symbol: str, data: Dict) -> str:
        return "Enable AI for price predictions. See: python run.py --help"
    
    def answer_question(self, question: str, context: Dict = None) -> str:
        return self.chat(question, context)


def get_reasoner() -> LLMReasoner:
    """Get the best available reasoner."""
    reasoner = LLMReasoner()
    if reasoner.initialized:
        return reasoner
    return SimpleLLMReasoner()
