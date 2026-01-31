"""
Advanced Analysis Module for Professional-Grade Predictions.

Features:
- Multi-timeframe analysis (Daily, Weekly, Monthly)
- Key support/resistance levels
- Relative strength vs market and sector
- Trend strength scoring
- Breakout/Breakdown detection
- Volume-price divergence
- Institutional activity signals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_fetchers.price_fetcher import PriceFetcher
from src.utils.logger import LoggerMixin


class AdvancedAnalyzer(LoggerMixin):
    """
    Professional-grade stock analysis with multi-timeframe and relative strength.
    """
    
    def __init__(self):
        self.price_fetcher = PriceFetcher()
    
    def get_comprehensive_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Get complete professional analysis for a stock.
        
        Returns comprehensive data for AI-powered predictions.
        """
        # Fetch data for different timeframes
        df_daily = self.price_fetcher.get_historical_data(symbol, period="1y", interval="1d")
        df_weekly = self._resample_to_weekly(df_daily)
        df_monthly = self._resample_to_monthly(df_daily)
        
        if df_daily.empty:
            return {"error": f"No data for {symbol}"}
        
        # Current price info
        current = df_daily.iloc[-1]
        current_price = current["close"]
        
        # Multi-timeframe analysis
        daily_analysis = self._analyze_timeframe(df_daily, "daily")
        weekly_analysis = self._analyze_timeframe(df_weekly, "weekly")
        monthly_analysis = self._analyze_timeframe(df_monthly, "monthly")
        
        # Key levels
        key_levels = self._find_key_levels(df_daily)
        
        # Trend analysis
        trend = self._analyze_trend(df_daily)
        
        # Momentum
        momentum = self._calculate_momentum(df_daily)
        
        # Volume analysis
        volume = self._analyze_volume(df_daily)
        
        # Relative strength
        rs_market = self._relative_strength_vs_market(df_daily)
        
        # Pattern detection
        patterns = self._detect_patterns(df_daily)
        
        # Breakout analysis
        breakout = self._analyze_breakout(df_daily, key_levels)
        
        # Risk metrics
        risk = self._calculate_risk_metrics(df_daily)
        
        # Price statistics
        stats = self._price_statistics(df_daily)
        
        # Combine timeframe signals
        timeframe_alignment = self._check_timeframe_alignment(
            daily_analysis, weekly_analysis, monthly_analysis
        )
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            daily_analysis, weekly_analysis, trend, momentum, volume, rs_market, breakout
        )
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "current_price": round(current_price, 2),
            
            # Multi-timeframe
            "timeframes": {
                "daily": daily_analysis,
                "weekly": weekly_analysis,
                "monthly": monthly_analysis,
                "alignment": timeframe_alignment,
            },
            
            # Key levels
            "key_levels": key_levels,
            
            # Trend
            "trend": trend,
            
            # Momentum
            "momentum": momentum,
            
            # Volume
            "volume": volume,
            
            # Relative strength
            "relative_strength": rs_market,
            
            # Patterns
            "patterns": patterns,
            
            # Breakout
            "breakout": breakout,
            
            # Risk
            "risk": risk,
            
            # Statistics
            "statistics": stats,
            
            # Overall
            "overall_score": overall_score,
            "signal_strength": self._get_signal_strength(overall_score),
        }
    
    def _resample_to_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample daily data to weekly."""
        if df.empty or "date" not in df.columns:
            return pd.DataFrame()
        
        df = df.copy()
        df.set_index("date", inplace=True)
        
        weekly = df.resample("W").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        
        weekly = weekly.reset_index()
        return weekly
    
    def _resample_to_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample daily data to monthly."""
        if df.empty or "date" not in df.columns:
            return pd.DataFrame()
        
        df = df.copy()
        df.set_index("date", inplace=True)
        
        monthly = df.resample("M").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        
        monthly = monthly.reset_index()
        return monthly
    
    def _analyze_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Analyze a single timeframe."""
        if df.empty or len(df) < 20:
            return {"trend": "UNKNOWN", "strength": 0}
        
        close = df["close"]
        
        # Moving averages
        sma_20 = close.rolling(20).mean().iloc[-1] if len(df) >= 20 else close.iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1] if len(df) >= 50 else sma_20
        
        current = close.iloc[-1]
        
        # Trend
        if current > sma_20 > sma_50:
            trend = "BULLISH"
            strength = min(100, 50 + (current - sma_20) / sma_20 * 500)
        elif current < sma_20 < sma_50:
            trend = "BEARISH"
            strength = max(0, 50 - (sma_20 - current) / sma_20 * 500)
        elif current > sma_20:
            trend = "MILDLY_BULLISH"
            strength = 60
        elif current < sma_20:
            trend = "MILDLY_BEARISH"
            strength = 40
        else:
            trend = "NEUTRAL"
            strength = 50
        
        # RSI
        rsi = self._calculate_rsi(close)
        
        # MACD
        macd, signal, hist = self._calculate_macd(close)
        macd_signal = "BULLISH" if hist > 0 else "BEARISH"
        
        return {
            "trend": trend,
            "strength": round(strength, 1),
            "price_vs_sma20": round((current / sma_20 - 1) * 100, 2),
            "price_vs_sma50": round((current / sma_50 - 1) * 100, 2),
            "rsi": round(rsi, 1),
            "macd_signal": macd_signal,
            "macd_histogram": round(hist, 2),
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        rs = gain / loss.replace(0, 0.0001)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[float, float, float]:
        """Calculate MACD."""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        hist = macd - signal
        
        return macd.iloc[-1], signal.iloc[-1], hist.iloc[-1]
    
    def _find_key_levels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Find key support and resistance levels."""
        if df.empty or len(df) < 20:
            return {}
        
        high = df["high"]
        low = df["low"]
        close = df["close"]
        current = close.iloc[-1]
        
        # Recent highs and lows
        high_52w = high.tail(252).max() if len(df) >= 252 else high.max()
        low_52w = low.tail(252).min() if len(df) >= 252 else low.min()
        
        high_20d = high.tail(20).max()
        low_20d = low.tail(20).min()
        
        # Pivot points
        pivot = (high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 3
        r1 = 2 * pivot - low.iloc[-1]
        s1 = 2 * pivot - high.iloc[-1]
        r2 = pivot + (high.iloc[-1] - low.iloc[-1])
        s2 = pivot - (high.iloc[-1] - low.iloc[-1])
        
        # Distance from levels
        dist_from_high = (high_52w - current) / current * 100
        dist_from_low = (current - low_52w) / low_52w * 100
        
        return {
            "current": round(current, 2),
            "resistance_1": round(r1, 2),
            "resistance_2": round(r2, 2),
            "support_1": round(s1, 2),
            "support_2": round(s2, 2),
            "pivot": round(pivot, 2),
            "high_52w": round(high_52w, 2),
            "low_52w": round(low_52w, 2),
            "high_20d": round(high_20d, 2),
            "low_20d": round(low_20d, 2),
            "dist_from_52w_high": round(dist_from_high, 2),
            "dist_from_52w_low": round(dist_from_low, 2),
            "position_in_range": round((current - low_52w) / (high_52w - low_52w) * 100, 1),
        }
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive trend analysis."""
        if df.empty or len(df) < 50:
            return {"direction": "UNKNOWN", "strength": 0}
        
        close = df["close"]
        
        # Multiple MAs
        sma_10 = close.rolling(10).mean()
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean() if len(df) >= 200 else sma_50
        
        current = close.iloc[-1]
        
        # MA alignment score
        ma_score = 0
        if current > sma_10.iloc[-1]: ma_score += 1
        if current > sma_20.iloc[-1]: ma_score += 1
        if current > sma_50.iloc[-1]: ma_score += 1
        if sma_10.iloc[-1] > sma_20.iloc[-1]: ma_score += 1
        if sma_20.iloc[-1] > sma_50.iloc[-1]: ma_score += 1
        if len(df) >= 200 and sma_50.iloc[-1] > sma_200.iloc[-1]: ma_score += 1
        
        # Trend direction
        if ma_score >= 5:
            direction = "STRONG_UPTREND"
        elif ma_score >= 4:
            direction = "UPTREND"
        elif ma_score >= 3:
            direction = "NEUTRAL"
        elif ma_score >= 2:
            direction = "DOWNTREND"
        else:
            direction = "STRONG_DOWNTREND"
        
        # Price changes
        change_5d = (current - close.iloc[-5]) / close.iloc[-5] * 100 if len(df) >= 5 else 0
        change_20d = (current - close.iloc[-20]) / close.iloc[-20] * 100 if len(df) >= 20 else 0
        change_60d = (current - close.iloc[-60]) / close.iloc[-60] * 100 if len(df) >= 60 else 0
        
        # ADX (simplified)
        tr = pd.DataFrame({
            "h-l": df["high"] - df["low"],
            "h-pc": abs(df["high"] - close.shift(1)),
            "l-pc": abs(df["low"] - close.shift(1)),
        }).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        adx = min(100, atr / current * 1000)  # Simplified ADX proxy
        
        return {
            "direction": direction,
            "ma_alignment_score": ma_score,
            "strength": round(ma_score / 6 * 100, 1),
            "change_5d": round(change_5d, 2),
            "change_20d": round(change_20d, 2),
            "change_60d": round(change_60d, 2),
            "adx_proxy": round(adx, 1),
            "above_200dma": len(df) >= 200 and current > sma_200.iloc[-1],
        }
    
    def _calculate_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate momentum indicators."""
        if df.empty or len(df) < 20:
            return {}
        
        close = df["close"]
        current = close.iloc[-1]
        
        # ROC (Rate of Change)
        roc_10 = (current - close.iloc[-10]) / close.iloc[-10] * 100 if len(df) >= 10 else 0
        roc_20 = (current - close.iloc[-20]) / close.iloc[-20] * 100 if len(df) >= 20 else 0
        
        # RSI
        rsi = self._calculate_rsi(close)
        
        # Stochastic
        low_14 = df["low"].tail(14).min()
        high_14 = df["high"].tail(14).max()
        stoch_k = (current - low_14) / (high_14 - low_14) * 100 if high_14 != low_14 else 50
        
        # Williams %R
        williams_r = -((high_14 - current) / (high_14 - low_14) * 100) if high_14 != low_14 else -50
        
        # Momentum score
        mom_score = 50
        if rsi > 50: mom_score += 10
        if rsi > 60: mom_score += 10
        if stoch_k > 50: mom_score += 10
        if roc_20 > 0: mom_score += 10
        if roc_20 > 5: mom_score += 10
        
        # Overbought/oversold
        if rsi > 70 and stoch_k > 80:
            condition = "OVERBOUGHT"
        elif rsi < 30 and stoch_k < 20:
            condition = "OVERSOLD"
        else:
            condition = "NEUTRAL"
        
        return {
            "rsi": round(rsi, 1),
            "stochastic_k": round(stoch_k, 1),
            "williams_r": round(williams_r, 1),
            "roc_10d": round(roc_10, 2),
            "roc_20d": round(roc_20, 2),
            "momentum_score": min(100, mom_score),
            "condition": condition,
        }
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns."""
        if df.empty or len(df) < 20 or "volume" not in df.columns:
            return {}
        
        volume = df["volume"]
        close = df["close"]
        
        current_vol = volume.iloc[-1]
        avg_vol_20 = volume.tail(20).mean()
        avg_vol_50 = volume.tail(50).mean() if len(df) >= 50 else avg_vol_20
        
        # Volume ratio
        vol_ratio = current_vol / avg_vol_20 if avg_vol_20 > 0 else 1
        
        # Price-volume relationship
        price_up = close.iloc[-1] > close.iloc[-2]
        vol_up = current_vol > avg_vol_20
        
        if price_up and vol_up:
            pv_signal = "BULLISH_CONFIRMATION"
        elif not price_up and vol_up:
            pv_signal = "BEARISH_CONFIRMATION"
        elif price_up and not vol_up:
            pv_signal = "WEAK_RALLY"
        else:
            pv_signal = "WEAK_DECLINE"
        
        # Volume trend
        vol_sma_5 = volume.tail(5).mean()
        vol_sma_20 = volume.tail(20).mean()
        vol_trend = "INCREASING" if vol_sma_5 > vol_sma_20 else "DECREASING"
        
        return {
            "current": int(current_vol),
            "avg_20d": int(avg_vol_20),
            "avg_50d": int(avg_vol_50),
            "ratio_vs_avg": round(vol_ratio, 2),
            "trend": vol_trend,
            "price_volume_signal": pv_signal,
            "is_high_volume": vol_ratio > 1.5,
        }
    
    def _relative_strength_vs_market(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate relative strength vs Nifty."""
        if df.empty or len(df) < 20:
            return {}
        
        # Get Nifty data
        nifty = self.price_fetcher.get_index_data("NIFTY", period="1y")
        
        if nifty.empty:
            return {"rs_score": 50, "vs_market": "UNKNOWN"}
        
        stock_return_20d = (df["close"].iloc[-1] - df["close"].iloc[-20]) / df["close"].iloc[-20] * 100
        nifty_return_20d = (nifty["close"].iloc[-1] - nifty["close"].iloc[-20]) / nifty["close"].iloc[-20] * 100
        
        stock_return_60d = 0
        nifty_return_60d = 0
        if len(df) >= 60 and len(nifty) >= 60:
            stock_return_60d = (df["close"].iloc[-1] - df["close"].iloc[-60]) / df["close"].iloc[-60] * 100
            nifty_return_60d = (nifty["close"].iloc[-1] - nifty["close"].iloc[-60]) / nifty["close"].iloc[-60] * 100
        
        # Relative strength
        rs_20d = stock_return_20d - nifty_return_20d
        rs_60d = stock_return_60d - nifty_return_60d
        
        # RS rating (0-100)
        rs_score = 50 + rs_20d * 2 + rs_60d
        rs_score = max(0, min(100, rs_score))
        
        if rs_score > 70:
            vs_market = "OUTPERFORMING"
        elif rs_score > 55:
            vs_market = "SLIGHTLY_BETTER"
        elif rs_score > 45:
            vs_market = "IN_LINE"
        elif rs_score > 30:
            vs_market = "SLIGHTLY_WORSE"
        else:
            vs_market = "UNDERPERFORMING"
        
        return {
            "rs_score": round(rs_score, 1),
            "vs_market": vs_market,
            "stock_return_20d": round(stock_return_20d, 2),
            "nifty_return_20d": round(nifty_return_20d, 2),
            "relative_return_20d": round(rs_20d, 2),
            "stock_return_60d": round(stock_return_60d, 2),
            "nifty_return_60d": round(nifty_return_60d, 2),
            "relative_return_60d": round(rs_60d, 2),
        }
    
    def _detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect chart patterns."""
        if df.empty or len(df) < 20:
            return {}
        
        patterns = []
        close = df["close"]
        high = df["high"]
        low = df["low"]
        
        current = close.iloc[-1]
        
        # Higher highs, higher lows
        highs = high.tail(20)
        lows = low.tail(20)
        
        hh = highs.iloc[-1] > highs.iloc[-5] > highs.iloc[-10]
        hl = lows.iloc[-1] > lows.iloc[-5] > lows.iloc[-10]
        
        if hh and hl:
            patterns.append("HIGHER_HIGHS_HIGHER_LOWS")
        
        # Lower highs, lower lows
        lh = highs.iloc[-1] < highs.iloc[-5] < highs.iloc[-10]
        ll = lows.iloc[-1] < lows.iloc[-5] < lows.iloc[-10]
        
        if lh and ll:
            patterns.append("LOWER_HIGHS_LOWER_LOWS")
        
        # Consolidation
        range_20d = (highs.max() - lows.min()) / current * 100
        if range_20d < 5:
            patterns.append("TIGHT_CONSOLIDATION")
        elif range_20d < 10:
            patterns.append("CONSOLIDATION")
        
        # Near 52-week high
        high_52w = high.tail(252).max() if len(df) >= 252 else high.max()
        if current >= high_52w * 0.95:
            patterns.append("NEAR_52W_HIGH")
        
        # Near 52-week low
        low_52w = low.tail(252).min() if len(df) >= 252 else low.min()
        if current <= low_52w * 1.05:
            patterns.append("NEAR_52W_LOW")
        
        # Volume climax
        vol = df["volume"]
        if vol.iloc[-1] > vol.tail(20).mean() * 2:
            patterns.append("VOLUME_CLIMAX")
        
        return {
            "detected": patterns,
            "count": len(patterns),
            "bullish_count": sum(1 for p in patterns if p in ["HIGHER_HIGHS_HIGHER_LOWS", "NEAR_52W_HIGH", "TIGHT_CONSOLIDATION"]),
            "bearish_count": sum(1 for p in patterns if p in ["LOWER_HIGHS_LOWER_LOWS", "NEAR_52W_LOW"]),
        }
    
    def _analyze_breakout(self, df: pd.DataFrame, levels: Dict) -> Dict[str, Any]:
        """Analyze breakout/breakdown potential."""
        if df.empty or not levels:
            return {}
        
        current = df["close"].iloc[-1]
        high_20d = levels.get("high_20d", current)
        low_20d = levels.get("low_20d", current)
        
        # Distance to breakout
        dist_to_breakout = (high_20d - current) / current * 100
        dist_to_breakdown = (current - low_20d) / current * 100
        
        # Recent breakout check
        prev_close = df["close"].iloc[-2]
        is_breakout = current > high_20d and prev_close <= high_20d
        is_breakdown = current < low_20d and prev_close >= low_20d
        
        # Volume confirmation
        vol_ratio = df["volume"].iloc[-1] / df["volume"].tail(20).mean()
        vol_confirmed = vol_ratio > 1.5
        
        status = "NEUTRAL"
        if is_breakout and vol_confirmed:
            status = "CONFIRMED_BREAKOUT"
        elif is_breakout:
            status = "BREAKOUT_NO_VOLUME"
        elif is_breakdown and vol_confirmed:
            status = "CONFIRMED_BREAKDOWN"
        elif is_breakdown:
            status = "BREAKDOWN_NO_VOLUME"
        elif dist_to_breakout < 2:
            status = "NEAR_BREAKOUT"
        elif dist_to_breakdown < 2:
            status = "NEAR_BREAKDOWN"
        
        return {
            "status": status,
            "distance_to_breakout": round(dist_to_breakout, 2),
            "distance_to_breakdown": round(dist_to_breakdown, 2),
            "volume_confirmed": vol_confirmed,
            "breakout_level": round(high_20d, 2),
            "breakdown_level": round(low_20d, 2),
        }
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics."""
        if df.empty or len(df) < 20:
            return {}
        
        close = df["close"]
        returns = close.pct_change().dropna()
        
        # Volatility
        volatility_20d = returns.tail(20).std() * np.sqrt(252) * 100
        
        # Max drawdown
        rolling_max = close.cummax()
        drawdown = (close - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # Beta (simplified - vs returns)
        beta = returns.tail(60).std() / returns.std() if len(df) >= 60 else 1.0
        
        # ATR
        tr = pd.DataFrame({
            "h-l": df["high"] - df["low"],
            "h-pc": abs(df["high"] - close.shift(1)),
            "l-pc": abs(df["low"] - close.shift(1)),
        }).max(axis=1)
        atr = tr.tail(14).mean()
        atr_pct = atr / close.iloc[-1] * 100
        
        # Risk level
        if volatility_20d > 40 or atr_pct > 3:
            risk_level = "HIGH"
        elif volatility_20d > 25 or atr_pct > 2:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "volatility_20d": round(volatility_20d, 2),
            "max_drawdown": round(max_drawdown, 2),
            "atr": round(atr, 2),
            "atr_percent": round(atr_pct, 2),
            "beta_proxy": round(beta, 2),
            "risk_level": risk_level,
            "suggested_stop_pct": round(atr_pct * 2, 2),
        }
    
    def _price_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate price statistics."""
        if df.empty:
            return {}
        
        close = df["close"]
        current = close.iloc[-1]
        
        # Historical percentiles
        percentile_current = (close < current).sum() / len(close) * 100
        
        # Returns
        returns = {
            "1d": round((current / close.iloc[-2] - 1) * 100, 2) if len(df) >= 2 else 0,
            "1w": round((current / close.iloc[-5] - 1) * 100, 2) if len(df) >= 5 else 0,
            "1m": round((current / close.iloc[-22] - 1) * 100, 2) if len(df) >= 22 else 0,
            "3m": round((current / close.iloc[-66] - 1) * 100, 2) if len(df) >= 66 else 0,
            "6m": round((current / close.iloc[-132] - 1) * 100, 2) if len(df) >= 132 else 0,
            "1y": round((current / close.iloc[0] - 1) * 100, 2),
        }
        
        return {
            "percentile_1y": round(percentile_current, 1),
            "returns": returns,
            "avg_price_1m": round(close.tail(22).mean(), 2),
            "avg_price_3m": round(close.tail(66).mean(), 2) if len(df) >= 66 else None,
        }
    
    def _check_timeframe_alignment(self, daily: Dict, weekly: Dict, monthly: Dict) -> Dict[str, Any]:
        """Check if all timeframes align."""
        trends = [
            daily.get("trend", "NEUTRAL"),
            weekly.get("trend", "NEUTRAL"),
            monthly.get("trend", "NEUTRAL"),
        ]
        
        bullish = sum(1 for t in trends if "BULLISH" in t)
        bearish = sum(1 for t in trends if "BEARISH" in t)
        
        if bullish >= 2:
            alignment = "BULLISH_ALIGNED"
            strength = bullish / 3 * 100
        elif bearish >= 2:
            alignment = "BEARISH_ALIGNED"
            strength = bearish / 3 * 100
        else:
            alignment = "MIXED"
            strength = 50
        
        return {
            "status": alignment,
            "strength": round(strength, 1),
            "daily": daily.get("trend"),
            "weekly": weekly.get("trend"),
            "monthly": monthly.get("trend"),
        }
    
    def _calculate_overall_score(
        self,
        daily: Dict,
        weekly: Dict,
        trend: Dict,
        momentum: Dict,
        volume: Dict,
        rs: Dict,
        breakout: Dict,
    ) -> float:
        """Calculate overall bullish/bearish score (0-100)."""
        score = 50  # Start neutral
        
        # Trend contribution (30%)
        trend_strength = trend.get("strength", 50)
        score += (trend_strength - 50) * 0.3
        
        # Timeframe alignment (20%)
        daily_strength = daily.get("strength", 50)
        weekly_strength = weekly.get("strength", 50)
        tf_avg = (daily_strength + weekly_strength) / 2
        score += (tf_avg - 50) * 0.2
        
        # Momentum (20%)
        mom_score = momentum.get("momentum_score", 50)
        score += (mom_score - 50) * 0.2
        
        # Relative strength (15%)
        rs_score = rs.get("rs_score", 50)
        score += (rs_score - 50) * 0.15
        
        # Volume (10%)
        if volume.get("price_volume_signal") == "BULLISH_CONFIRMATION":
            score += 5
        elif volume.get("price_volume_signal") == "BEARISH_CONFIRMATION":
            score -= 5
        
        # Breakout bonus (5%)
        if breakout.get("status") == "CONFIRMED_BREAKOUT":
            score += 10
        elif breakout.get("status") == "CONFIRMED_BREAKDOWN":
            score -= 10
        elif breakout.get("status") == "NEAR_BREAKOUT":
            score += 3
        
        return max(0, min(100, round(score, 1)))
    
    def _get_signal_strength(self, score: float) -> str:
        """Convert score to signal strength."""
        if score >= 75:
            return "STRONG_BUY"
        elif score >= 60:
            return "BUY"
        elif score >= 45:
            return "HOLD"
        elif score >= 30:
            return "SELL"
        else:
            return "STRONG_SELL"


def get_advanced_analysis(symbol: str) -> Dict[str, Any]:
    """Get advanced analysis for a symbol."""
    analyzer = AdvancedAnalyzer()
    return analyzer.get_comprehensive_analysis(symbol)
