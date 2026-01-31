"""
💰 PRO ANALYZER - Serious Money-Making Analysis

This module combines multiple professional trading techniques:
1. Multi-timeframe trend alignment
2. Smart money / volume analysis
3. Chart pattern recognition
4. Key level detection
5. Momentum scoring
6. Risk-reward optimization

The goal: Find HIGH PROBABILITY trades with excellent R:R
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import ta
import ta.momentum
import ta.trend
import ta.volatility
import ta.volume

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_fetchers.price_fetcher import PriceFetcher
from src.utils.logger import LoggerMixin


class ProAnalyzer(LoggerMixin):
    """
    Professional-grade stock analyzer for serious traders.
    Combines multiple analysis techniques for high-probability setups.
    """
    
    def __init__(self):
        self.price_fetcher = PriceFetcher()
    
    def full_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Run complete professional analysis on a stock.
        Returns everything needed to make a trading decision.
        """
        self.logger.info(f"Running PRO analysis on {symbol}...")
        
        # Fetch data for multiple timeframes
        df_daily = self.price_fetcher.get_historical_data(symbol, period="1y", interval="1d")
        df_weekly = self._resample_to_weekly(df_daily)
        
        if df_daily.empty or len(df_daily) < 50:
            return {"error": f"Insufficient data for {symbol}"}
        
        # Get current price info
        current_price = float(df_daily['close'].iloc[-1])
        prev_close = float(df_daily['close'].iloc[-2])
        change_pct = ((current_price - prev_close) / prev_close) * 100
        
        # Run all analyses
        trend = self._analyze_trend(df_daily, df_weekly)
        momentum = self._analyze_momentum(df_daily)
        volume = self._analyze_volume(df_daily)
        patterns = self._detect_patterns(df_daily)
        levels = self._find_key_levels(df_daily)
        strength = self._calculate_relative_strength(symbol, df_daily)
        
        # Calculate scores
        scores = self._calculate_scores(trend, momentum, volume, patterns, strength)
        
        # Generate signal
        signal = self._generate_signal(scores, current_price, levels, df_daily)
        
        # Risk management
        risk = self._calculate_risk_reward(current_price, levels, signal, df_daily)
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "change_pct": round(change_pct, 2),
            "timestamp": datetime.now().isoformat(),
            
            # Analysis components
            "trend": trend,
            "momentum": momentum,
            "volume": volume,
            "patterns": patterns,
            "levels": levels,
            "relative_strength": strength,
            
            # Scores
            "scores": scores,
            "overall_score": scores["overall"],
            
            # Signal & Risk
            "signal": signal,
            "risk": risk,
            
            # Quick summary
            "summary": self._generate_summary(symbol, signal, scores, risk),
        }
    
    def _resample_to_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert daily data to weekly."""
        if df.empty:
            return df
        
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy.set_index('date', inplace=True)
        
        weekly = df_copy.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        weekly.reset_index(inplace=True)
        return weekly
    
    # =========================================================================
    # TREND ANALYSIS
    # =========================================================================
    
    def _analyze_trend(self, df: pd.DataFrame, df_weekly: pd.DataFrame) -> Dict[str, Any]:
        """
        Multi-timeframe trend analysis.
        Checks if daily, weekly trends align.
        """
        close = df['close']
        
        # EMAs
        ema_8 = close.ewm(span=8).mean()
        ema_21 = close.ewm(span=21).mean()
        ema_50 = close.ewm(span=50).mean()
        ema_200 = close.ewm(span=200).mean() if len(close) >= 200 else close.ewm(span=100).mean()
        
        current = close.iloc[-1]
        
        # Daily trend
        daily_trend = "BULLISH" if current > ema_21.iloc[-1] > ema_50.iloc[-1] else \
                      "BEARISH" if current < ema_21.iloc[-1] < ema_50.iloc[-1] else "NEUTRAL"
        
        # Weekly trend
        if not df_weekly.empty and len(df_weekly) >= 10:
            w_close = df_weekly['close']
            w_ema_8 = w_close.ewm(span=8).mean()
            w_ema_21 = w_close.ewm(span=21).mean()
            weekly_trend = "BULLISH" if w_close.iloc[-1] > w_ema_8.iloc[-1] > w_ema_21.iloc[-1] else \
                           "BEARISH" if w_close.iloc[-1] < w_ema_8.iloc[-1] < w_ema_21.iloc[-1] else "NEUTRAL"
        else:
            weekly_trend = daily_trend
        
        # EMA alignment score (0-100)
        ema_stack_bullish = int(current > ema_8.iloc[-1] > ema_21.iloc[-1] > ema_50.iloc[-1])
        ema_stack_bearish = int(current < ema_8.iloc[-1] < ema_21.iloc[-1] < ema_50.iloc[-1])
        
        # Trend strength (ADX)
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        adx_value = adx.adx().iloc[-1] if not adx.adx().empty else 20
        
        trend_strength = "STRONG" if adx_value > 25 else "WEAK" if adx_value < 20 else "MODERATE"
        
        # Above/below 200 EMA
        above_200ema = current > ema_200.iloc[-1]
        
        # Trend alignment
        aligned = daily_trend == weekly_trend and daily_trend != "NEUTRAL"
        
        return {
            "daily": daily_trend,
            "weekly": weekly_trend,
            "aligned": aligned,
            "strength": trend_strength,
            "adx": round(adx_value, 1),
            "above_200ema": above_200ema,
            "ema_values": {
                "ema_8": round(ema_8.iloc[-1], 2),
                "ema_21": round(ema_21.iloc[-1], 2),
                "ema_50": round(ema_50.iloc[-1], 2),
                "ema_200": round(ema_200.iloc[-1], 2),
            },
            "score": min(100, (50 if aligned else 25) + adx_value + (10 if above_200ema else 0)),
        }
    
    # =========================================================================
    # MOMENTUM ANALYSIS
    # =========================================================================
    
    def _analyze_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Momentum indicators analysis.
        RSI, MACD, Stochastic, ROC
        """
        close = df['close']
        high = df['high']
        low = df['low']
        
        # RSI
        rsi = ta.momentum.RSIIndicator(close, window=14)
        rsi_value = rsi.rsi().iloc[-1]
        
        # RSI divergence check
        rsi_divergence = self._check_rsi_divergence(close, rsi.rsi())
        
        # MACD
        macd = ta.trend.MACD(close)
        macd_line = macd.macd().iloc[-1]
        macd_signal = macd.macd_signal().iloc[-1]
        macd_hist = macd.macd_diff().iloc[-1]
        macd_prev_hist = macd.macd_diff().iloc[-2] if len(macd.macd_diff()) > 1 else 0
        macd_increasing = macd_hist > macd_prev_hist
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(high, low, close)
        stoch_k = stoch.stoch().iloc[-1]
        stoch_d = stoch.stoch_signal().iloc[-1]
        
        # Rate of Change (momentum)
        roc_10 = ((close.iloc[-1] / close.iloc[-10]) - 1) * 100 if len(close) > 10 else 0
        roc_20 = ((close.iloc[-1] / close.iloc[-20]) - 1) * 100 if len(close) > 20 else 0
        
        # Momentum condition
        if rsi_value > 70:
            condition = "OVERBOUGHT"
        elif rsi_value < 30:
            condition = "OVERSOLD"
        elif rsi_value > 50 and macd_hist > 0:
            condition = "BULLISH"
        elif rsi_value < 50 and macd_hist < 0:
            condition = "BEARISH"
        else:
            condition = "NEUTRAL"
        
        # Score
        score = 50
        if condition == "BULLISH":
            score = 70
        elif condition == "OVERSOLD" and macd_increasing:
            score = 80  # Potential reversal
        elif condition == "BEARISH":
            score = 30
        elif condition == "OVERBOUGHT":
            score = 40
        
        if rsi_divergence == "BULLISH":
            score += 15
        elif rsi_divergence == "BEARISH":
            score -= 15
        
        return {
            "rsi": round(rsi_value, 1),
            "rsi_divergence": rsi_divergence,
            "macd": {
                "line": round(macd_line, 2),
                "signal": round(macd_signal, 2),
                "histogram": round(macd_hist, 2),
                "increasing": macd_increasing,
                "bullish": macd_line > macd_signal,
            },
            "stochastic": {
                "k": round(stoch_k, 1),
                "d": round(stoch_d, 1),
                "oversold": stoch_k < 20,
                "overbought": stoch_k > 80,
            },
            "roc": {
                "10d": round(roc_10, 2),
                "20d": round(roc_20, 2),
            },
            "condition": condition,
            "score": min(100, max(0, score)),
        }
    
    def _check_rsi_divergence(self, price: pd.Series, rsi: pd.Series) -> str:
        """Check for RSI divergence (leading indicator)."""
        if len(price) < 20:
            return "NONE"
        
        # Get last 20 days
        price_recent = price.iloc[-20:]
        rsi_recent = rsi.iloc[-20:]
        
        # Find local lows
        price_low_idx = price_recent.idxmin()
        
        # Check if price making lower low but RSI making higher low (bullish divergence)
        # Simplified check
        price_slope = (price_recent.iloc[-1] - price_recent.iloc[0]) / len(price_recent)
        rsi_slope = (rsi_recent.iloc[-1] - rsi_recent.iloc[0]) / len(rsi_recent)
        
        if price_slope < 0 and rsi_slope > 0 and rsi_recent.iloc[-1] < 40:
            return "BULLISH"
        elif price_slope > 0 and rsi_slope < 0 and rsi_recent.iloc[-1] > 60:
            return "BEARISH"
        
        return "NONE"
    
    # =========================================================================
    # VOLUME ANALYSIS (Smart Money)
    # =========================================================================
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Volume analysis to detect smart money activity.
        High volume on up days = accumulation
        High volume on down days = distribution
        """
        close = df['close']
        volume = df['volume']
        
        # Average volume
        avg_volume_20 = volume.rolling(20).mean().iloc[-1]
        current_volume = volume.iloc[-1]
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
        
        # Volume trend
        vol_sma_5 = volume.rolling(5).mean()
        vol_sma_20 = volume.rolling(20).mean()
        volume_increasing = vol_sma_5.iloc[-1] > vol_sma_20.iloc[-1]
        
        # On-Balance Volume (OBV)
        obv = ta.volume.OnBalanceVolumeIndicator(close, volume)
        obv_values = obv.on_balance_volume()
        obv_trend = "UP" if obv_values.iloc[-1] > obv_values.iloc[-20] else "DOWN"
        
        # Volume-Price Analysis
        # Check last 10 days: up days with high volume = accumulation
        accumulation_days = 0
        distribution_days = 0
        
        for i in range(-10, 0):
            if close.iloc[i] > close.iloc[i-1]:  # Up day
                if volume.iloc[i] > avg_volume_20:
                    accumulation_days += 1
            else:  # Down day
                if volume.iloc[i] > avg_volume_20:
                    distribution_days += 1
        
        if accumulation_days > distribution_days + 2:
            smart_money = "ACCUMULATING"
        elif distribution_days > accumulation_days + 2:
            smart_money = "DISTRIBUTING"
        else:
            smart_money = "NEUTRAL"
        
        # Volume spike detection
        volume_spike = volume_ratio > 2
        
        # Score
        score = 50
        if smart_money == "ACCUMULATING":
            score = 75
        elif smart_money == "DISTRIBUTING":
            score = 25
        
        if obv_trend == "UP":
            score += 10
        else:
            score -= 10
        
        return {
            "current": int(current_volume),
            "avg_20d": int(avg_volume_20),
            "ratio": round(volume_ratio, 2),
            "spike": volume_spike,
            "increasing": volume_increasing,
            "obv_trend": obv_trend,
            "smart_money": smart_money,
            "accumulation_days": accumulation_days,
            "distribution_days": distribution_days,
            "score": min(100, max(0, score)),
        }
    
    # =========================================================================
    # PATTERN DETECTION
    # =========================================================================
    
    def _detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect chart patterns and candlestick patterns.
        """
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']
        
        patterns_found = []
        
        # ===== Candlestick Patterns (Last 3 candles) =====
        
        # Bullish Engulfing
        if (close.iloc[-1] > open_price.iloc[-1] and  # Today bullish
            close.iloc[-2] < open_price.iloc[-2] and  # Yesterday bearish
            close.iloc[-1] > open_price.iloc[-2] and  # Today close > yesterday open
            open_price.iloc[-1] < close.iloc[-2]):    # Today open < yesterday close
            patterns_found.append({"name": "Bullish Engulfing", "type": "BULLISH", "strength": 75})
        
        # Bearish Engulfing
        if (close.iloc[-1] < open_price.iloc[-1] and
            close.iloc[-2] > open_price.iloc[-2] and
            close.iloc[-1] < open_price.iloc[-2] and
            open_price.iloc[-1] > close.iloc[-2]):
            patterns_found.append({"name": "Bearish Engulfing", "type": "BEARISH", "strength": 75})
        
        # Hammer (bullish at support)
        body = abs(close.iloc[-1] - open_price.iloc[-1])
        lower_wick = min(close.iloc[-1], open_price.iloc[-1]) - low.iloc[-1]
        upper_wick = high.iloc[-1] - max(close.iloc[-1], open_price.iloc[-1])
        
        if lower_wick > body * 2 and upper_wick < body * 0.5:
            patterns_found.append({"name": "Hammer", "type": "BULLISH", "strength": 70})
        
        # Shooting Star (bearish at resistance)
        if upper_wick > body * 2 and lower_wick < body * 0.5:
            patterns_found.append({"name": "Shooting Star", "type": "BEARISH", "strength": 70})
        
        # Doji (indecision)
        if body < (high.iloc[-1] - low.iloc[-1]) * 0.1:
            patterns_found.append({"name": "Doji", "type": "NEUTRAL", "strength": 50})
        
        # ===== Chart Patterns (Multi-day) =====
        
        # Higher Highs and Higher Lows (Uptrend)
        if len(df) >= 10:
            recent_highs = high.iloc[-10:]
            recent_lows = low.iloc[-10:]
            
            hh = all(recent_highs.iloc[i] >= recent_highs.iloc[i-2] for i in range(2, len(recent_highs), 2))
            hl = all(recent_lows.iloc[i] >= recent_lows.iloc[i-2] for i in range(2, len(recent_lows), 2))
            
            if hh and hl:
                patterns_found.append({"name": "Higher Highs & Lows", "type": "BULLISH", "strength": 80})
        
        # Breakout detection
        high_20 = high.rolling(20).max().iloc[-2]  # Previous 20-day high
        if close.iloc[-1] > high_20:
            patterns_found.append({"name": "20-Day Breakout", "type": "BULLISH", "strength": 85})
        
        low_20 = low.rolling(20).min().iloc[-2]
        if close.iloc[-1] < low_20:
            patterns_found.append({"name": "20-Day Breakdown", "type": "BEARISH", "strength": 85})
        
        # Consolidation (low volatility, potential breakout coming)
        atr = ta.volatility.AverageTrueRange(high, low, close, window=14)
        atr_value = atr.average_true_range().iloc[-1]
        atr_20 = atr.average_true_range().iloc[-20] if len(atr.average_true_range()) > 20 else atr_value
        
        if atr_value < atr_20 * 0.7:
            patterns_found.append({"name": "Consolidation (Squeeze)", "type": "NEUTRAL", "strength": 60})
        
        # Score based on patterns
        bullish_count = sum(1 for p in patterns_found if p["type"] == "BULLISH")
        bearish_count = sum(1 for p in patterns_found if p["type"] == "BEARISH")
        
        score = 50 + (bullish_count - bearish_count) * 15
        
        return {
            "patterns": patterns_found,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "dominant": "BULLISH" if bullish_count > bearish_count else "BEARISH" if bearish_count > bullish_count else "NEUTRAL",
            "score": min(100, max(0, score)),
        }
    
    # =========================================================================
    # KEY LEVELS
    # =========================================================================
    
    def _find_key_levels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Find important support and resistance levels.
        """
        close = df['close']
        high = df['high']
        low = df['low']
        current = close.iloc[-1]
        
        # 52-week high/low
        high_52w = high.max()
        low_52w = low.min()
        
        # Recent swing points (last 60 days)
        recent_high = high.iloc[-60:].max() if len(high) >= 60 else high.max()
        recent_low = low.iloc[-60:].min() if len(low) >= 60 else low.min()
        
        # Pivot points (traditional)
        prev_high = high.iloc[-2]
        prev_low = low.iloc[-2]
        prev_close = close.iloc[-2]
        
        pivot = (prev_high + prev_low + prev_close) / 3
        r1 = 2 * pivot - prev_low
        r2 = pivot + (prev_high - prev_low)
        s1 = 2 * pivot - prev_high
        s2 = pivot - (prev_high - prev_low)
        
        # Find nearest support/resistance
        supports = sorted([s for s in [s1, s2, recent_low, low_52w] if s < current], reverse=True)
        resistances = sorted([r for r in [r1, r2, recent_high, high_52w] if r > current])
        
        nearest_support = supports[0] if supports else current * 0.95
        nearest_resistance = resistances[0] if resistances else current * 1.05
        
        # Distance to levels (as %)
        distance_to_support = ((current - nearest_support) / current) * 100
        distance_to_resistance = ((nearest_resistance - current) / current) * 100
        
        # Position in range
        range_52w = high_52w - low_52w
        position_in_range = ((current - low_52w) / range_52w) * 100 if range_52w > 0 else 50
        
        return {
            "current_price": round(current, 2),
            "support": {
                "nearest": round(nearest_support, 2),
                "distance_pct": round(distance_to_support, 2),
                "all": [round(s, 2) for s in supports[:3]],
            },
            "resistance": {
                "nearest": round(nearest_resistance, 2),
                "distance_pct": round(distance_to_resistance, 2),
                "all": [round(r, 2) for r in resistances[:3]],
            },
            "52week": {
                "high": round(high_52w, 2),
                "low": round(low_52w, 2),
                "position_pct": round(position_in_range, 1),
            },
            "pivot_points": {
                "pivot": round(pivot, 2),
                "r1": round(r1, 2),
                "r2": round(r2, 2),
                "s1": round(s1, 2),
                "s2": round(s2, 2),
            },
        }
    
    # =========================================================================
    # RELATIVE STRENGTH
    # =========================================================================
    
    def _calculate_relative_strength(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate relative strength vs Nifty.
        Outperformers tend to continue outperforming.
        """
        close = df['close']
        
        # Get Nifty data
        nifty_df = self.price_fetcher.get_historical_data("^NSEI", period="1y", interval="1d")
        
        if nifty_df.empty:
            return {"rs_score": 50, "vs_nifty": "N/A"}
        
        # Calculate returns
        stock_return_1m = ((close.iloc[-1] / close.iloc[-22]) - 1) * 100 if len(close) > 22 else 0
        stock_return_3m = ((close.iloc[-1] / close.iloc[-66]) - 1) * 100 if len(close) > 66 else 0
        
        nifty_close = nifty_df['close']
        nifty_return_1m = ((nifty_close.iloc[-1] / nifty_close.iloc[-22]) - 1) * 100 if len(nifty_close) > 22 else 0
        nifty_return_3m = ((nifty_close.iloc[-1] / nifty_close.iloc[-66]) - 1) * 100 if len(nifty_close) > 66 else 0
        
        # RS calculation
        rs_1m = stock_return_1m - nifty_return_1m
        rs_3m = stock_return_3m - nifty_return_3m
        
        # RS score (0-100)
        rs_score = 50 + (rs_1m * 2) + (rs_3m * 1)
        rs_score = min(100, max(0, rs_score))
        
        if rs_score > 65:
            vs_nifty = "OUTPERFORMING"
        elif rs_score < 35:
            vs_nifty = "UNDERPERFORMING"
        else:
            vs_nifty = "IN-LINE"
        
        return {
            "rs_score": round(rs_score, 1),
            "vs_nifty": vs_nifty,
            "stock_return_1m": round(stock_return_1m, 2),
            "stock_return_3m": round(stock_return_3m, 2),
            "nifty_return_1m": round(nifty_return_1m, 2),
            "nifty_return_3m": round(nifty_return_3m, 2),
            "alpha_1m": round(rs_1m, 2),
            "alpha_3m": round(rs_3m, 2),
        }
    
    # =========================================================================
    # SCORING SYSTEM
    # =========================================================================
    
    def _calculate_scores(self, trend: Dict, momentum: Dict, volume: Dict, 
                          patterns: Dict, strength: Dict) -> Dict[str, Any]:
        """
        Calculate weighted scores for final signal.
        """
        # Individual scores
        trend_score = trend.get("score", 50)
        momentum_score = momentum.get("score", 50)
        volume_score = volume.get("score", 50)
        pattern_score = patterns.get("score", 50)
        rs_score = strength.get("rs_score", 50)
        
        # Weights (trend is most important for swing trading)
        weights = {
            "trend": 0.30,
            "momentum": 0.25,
            "volume": 0.20,
            "patterns": 0.15,
            "relative_strength": 0.10,
        }
        
        overall = (
            trend_score * weights["trend"] +
            momentum_score * weights["momentum"] +
            volume_score * weights["volume"] +
            pattern_score * weights["patterns"] +
            rs_score * weights["relative_strength"]
        )
        
        # Bonus for alignment
        if trend["aligned"] and momentum["condition"] in ["BULLISH", "OVERSOLD"]:
            overall += 5
        if volume["smart_money"] == "ACCUMULATING" and trend["daily"] == "BULLISH":
            overall += 5
        
        return {
            "trend": round(trend_score, 1),
            "momentum": round(momentum_score, 1),
            "volume": round(volume_score, 1),
            "patterns": round(pattern_score, 1),
            "relative_strength": round(rs_score, 1),
            "overall": round(min(100, overall), 1),
        }
    
    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================
    
    def _generate_signal(self, scores: Dict, current_price: float, 
                         levels: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate final trading signal with entry, stop, targets.
        """
        overall = scores["overall"]
        
        # Signal based on score
        if overall >= 75:
            signal = "STRONG_BUY"
            action = "BUY"
            confidence = min(95, overall + 10)
        elif overall >= 60:
            signal = "BUY"
            action = "BUY"
            confidence = overall
        elif overall <= 25:
            signal = "STRONG_SELL"
            action = "SELL"
            confidence = min(95, 100 - overall + 10)
        elif overall <= 40:
            signal = "SELL"
            action = "SELL"
            confidence = 100 - overall
        else:
            signal = "HOLD"
            action = "WAIT"
            confidence = 50
        
        # Entry, Stop Loss, Targets
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        atr_value = atr.average_true_range().iloc[-1]
        
        if action == "BUY":
            entry = current_price  # Market order or limit at current
            stop_loss = max(
                current_price - (2 * atr_value),  # 2 ATR stop
                levels["support"]["nearest"] * 0.99  # Just below support
            )
            target_1 = current_price + (2 * atr_value)  # 1:1 R:R
            target_2 = current_price + (3 * atr_value)  # 1:1.5 R:R
            target_3 = levels["resistance"]["nearest"]  # Resistance
        elif action == "SELL":
            entry = current_price
            stop_loss = min(
                current_price + (2 * atr_value),
                levels["resistance"]["nearest"] * 1.01
            )
            target_1 = current_price - (2 * atr_value)
            target_2 = current_price - (3 * atr_value)
            target_3 = levels["support"]["nearest"]
        else:
            entry = current_price
            stop_loss = levels["support"]["nearest"]
            target_1 = levels["resistance"]["nearest"]
            target_2 = target_1 * 1.02
            target_3 = target_1 * 1.05
        
        # Risk calculation
        risk_pct = abs((current_price - stop_loss) / current_price) * 100
        reward_pct = abs((target_1 - current_price) / current_price) * 100
        risk_reward = reward_pct / risk_pct if risk_pct > 0 else 0
        
        return {
            "signal": signal,
            "action": action,
            "confidence": round(confidence, 1),
            "entry_price": round(entry, 2),
            "stop_loss": round(stop_loss, 2),
            "targets": {
                "target_1": round(target_1, 2),
                "target_2": round(target_2, 2),
                "target_3": round(target_3, 2),
            },
            "risk_pct": round(risk_pct, 2),
            "reward_pct": round(reward_pct, 2),
            "risk_reward_ratio": round(risk_reward, 2),
        }
    
    # =========================================================================
    # RISK-REWARD OPTIMIZATION
    # =========================================================================
    
    def _calculate_risk_reward(self, current_price: float, levels: Dict, 
                                signal: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate position sizing and risk management.
        """
        stop_loss = signal["stop_loss"]
        target = signal["targets"]["target_1"]
        
        risk_per_share = abs(current_price - stop_loss)
        reward_per_share = abs(target - current_price)
        
        # Position sizing (risk 2% of capital)
        capital = 100000  # Default capital
        risk_amount = capital * 0.02  # 2% risk
        
        shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        position_value = shares * current_price
        position_pct = (position_value / capital) * 100
        
        # Max loss / Max gain
        max_loss = shares * risk_per_share
        max_gain_t1 = shares * reward_per_share
        
        return {
            "recommended_shares": shares,
            "position_value": round(position_value, 2),
            "position_pct": round(position_pct, 2),
            "risk_per_share": round(risk_per_share, 2),
            "max_loss": round(max_loss, 2),
            "max_gain_t1": round(max_gain_t1, 2),
            "capital_at_risk_pct": 2.0,
        }
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    def _generate_summary(self, symbol: str, signal: Dict, scores: Dict, risk: Dict) -> str:
        """
        Generate human-readable summary.
        """
        sig = signal["signal"]
        conf = signal["confidence"]
        rr = signal["risk_reward_ratio"]
        
        if "BUY" in sig:
            action_text = f"🟢 {sig} with {conf:.0f}% confidence"
        elif "SELL" in sig:
            action_text = f"🔴 {sig} with {conf:.0f}% confidence"
        else:
            action_text = f"🟡 {sig} - Wait for better setup"
        
        summary = f"""
{symbol}: {action_text}

📊 Scores: Trend {scores['trend']:.0f} | Momentum {scores['momentum']:.0f} | Volume {scores['volume']:.0f}

💰 Trade Setup:
• Entry: ₹{signal['entry_price']:,.2f}
• Stop: ₹{signal['stop_loss']:,.2f} (-{signal['risk_pct']:.1f}%)
• Target: ₹{signal['targets']['target_1']:,.2f} (+{signal['reward_pct']:.1f}%)
• R:R Ratio: 1:{rr:.1f}

📈 Position: {risk['recommended_shares']} shares (₹{risk['position_value']:,.0f})
• Max Loss: ₹{risk['max_loss']:,.0f} (2% of capital)
• Max Gain: ₹{risk['max_gain_t1']:,.0f}
"""
        return summary.strip()


# Convenience function
def get_pro_analysis(symbol: str) -> Dict[str, Any]:
    """Get professional analysis for a symbol."""
    analyzer = ProAnalyzer()
    return analyzer.full_analysis(symbol)
