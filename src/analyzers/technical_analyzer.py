"""
Technical Analysis Module for Indian Stock Market.

Comprehensive technical analysis including:
- Trend indicators (SMA, EMA, MACD, ADX, Supertrend)
- Momentum indicators (RSI, Stochastic, CCI, MFI)
- Volatility indicators (Bollinger Bands, ATR)
- Volume indicators (OBV, VWAP, CMF)
- Support/Resistance levels
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

try:
    import pandas_ta as ta
except ImportError:
    ta = None

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import INDICATOR_SETTINGS
from src.utils.logger import LoggerMixin


class TechnicalAnalyzer(LoggerMixin):
    """
    Performs comprehensive technical analysis on stock data.
    """
    
    def __init__(self):
        """Initialize the technical analyzer."""
        self.settings = INDICATOR_SETTINGS
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a dataframe.
        
        Args:
            df: OHLCV dataframe with columns: open, high, low, close, volume
        
        Returns:
            DataFrame with all indicators added
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Ensure required columns exist
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                self.logger.error(f"Missing required column: {col}")
                return df
        
        # Calculate indicators using pandas_ta if available
        if ta is not None:
            df = self._calculate_with_pandas_ta(df)
        else:
            df = self._calculate_manually(df)
        
        return df
    
    def _calculate_with_pandas_ta(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicators using pandas_ta library.
        
        Args:
            df: OHLCV dataframe
        
        Returns:
            DataFrame with indicators
        """
        # Trend Indicators
        for period in self.settings["sma_periods"]:
            df[f"sma_{period}"] = ta.sma(df["close"], length=period)
        
        for period in self.settings["ema_periods"]:
            df[f"ema_{period}"] = ta.ema(df["close"], length=period)
        
        # MACD
        macd_settings = self.settings["macd"]
        macd = ta.macd(
            df["close"],
            fast=macd_settings["fast"],
            slow=macd_settings["slow"],
            signal=macd_settings["signal"]
        )
        if macd is not None:
            df = pd.concat([df, macd], axis=1)
        
        # ADX
        adx = ta.adx(df["high"], df["low"], df["close"], length=self.settings["adx_period"])
        if adx is not None:
            df = pd.concat([df, adx], axis=1)
        
        # Supertrend
        st = ta.supertrend(
            df["high"], df["low"], df["close"],
            length=self.settings["supertrend"]["period"],
            multiplier=self.settings["supertrend"]["multiplier"]
        )
        if st is not None:
            df = pd.concat([df, st], axis=1)
        
        # Momentum Indicators
        df["rsi"] = ta.rsi(df["close"], length=self.settings["rsi_period"])
        
        stoch = ta.stoch(
            df["high"], df["low"], df["close"],
            k=self.settings["stochastic"]["k_period"],
            d=self.settings["stochastic"]["d_period"],
            smooth_k=self.settings["stochastic"]["smooth"]
        )
        if stoch is not None:
            df = pd.concat([df, stoch], axis=1)
        
        df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=self.settings["cci_period"])
        df["williams_r"] = ta.willr(df["high"], df["low"], df["close"], length=self.settings["williams_period"])
        df["roc"] = ta.roc(df["close"], length=self.settings["roc_period"])
        df["mfi"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=self.settings["mfi_period"])
        
        # Volatility Indicators
        bb = ta.bbands(
            df["close"],
            length=self.settings["bollinger"]["period"],
            std=self.settings["bollinger"]["std"]
        )
        if bb is not None:
            df = pd.concat([df, bb], axis=1)
        
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=self.settings["atr_period"])
        
        kc = ta.kc(
            df["high"], df["low"], df["close"],
            length=self.settings["keltner"]["period"],
            scalar=self.settings["keltner"]["multiplier"]
        )
        if kc is not None:
            df = pd.concat([df, kc], axis=1)
        
        dc = ta.donchian(df["high"], df["low"], lower_length=self.settings["donchian_period"])
        if dc is not None:
            df = pd.concat([df, dc], axis=1)
        
        # Volume Indicators
        df["obv"] = ta.obv(df["close"], df["volume"])
        df["cmf"] = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=self.settings["cmf_period"])
        df["ad"] = ta.ad(df["high"], df["low"], df["close"], df["volume"])
        
        # Volume SMA
        df["volume_sma"] = ta.sma(df["volume"], length=self.settings["volume_sma_period"])
        df["volume_ratio"] = df["volume"] / df["volume_sma"]
        
        # VWAP (for intraday)
        try:
            df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
        except Exception:
            pass
        
        return df
    
    def _calculate_manually(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicators manually without pandas_ta.
        
        Args:
            df: OHLCV dataframe
        
        Returns:
            DataFrame with indicators
        """
        # Simple Moving Averages
        for period in self.settings["sma_periods"]:
            df[f"sma_{period}"] = df["close"].rolling(window=period).mean()
        
        # Exponential Moving Averages
        for period in self.settings["ema_periods"]:
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
        
        # MACD
        fast = self.settings["macd"]["fast"]
        slow = self.settings["macd"]["slow"]
        signal = self.settings["macd"]["signal"]
        df["macd"] = df["close"].ewm(span=fast).mean() - df["close"].ewm(span=slow).mean()
        df["macd_signal"] = df["macd"].ewm(span=signal).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.settings["rsi_period"]).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.settings["rsi_period"]).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        period = self.settings["bollinger"]["period"]
        std_dev = self.settings["bollinger"]["std"]
        df["bb_middle"] = df["close"].rolling(window=period).mean()
        df["bb_std"] = df["close"].rolling(window=period).std()
        df["bb_upper"] = df["bb_middle"] + (std_dev * df["bb_std"])
        df["bb_lower"] = df["bb_middle"] - (std_dev * df["bb_std"])
        
        # ATR
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1))
            )
        )
        df["atr"] = df["tr"].rolling(window=self.settings["atr_period"]).mean()
        
        # Volume SMA
        df["volume_sma"] = df["volume"].rolling(window=self.settings["volume_sma_period"]).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]
        
        # OBV
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        
        return df
    
    def get_indicator_signals(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Get buy/sell signals from indicators.
        
        Args:
            df: DataFrame with calculated indicators
        
        Returns:
            Dictionary with indicator signals
        """
        if df.empty:
            return {}
        
        signals = {}
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
        
        # Moving Average Signals
        if "ema_9" in df.columns and "ema_21" in df.columns:
            if latest["ema_9"] > latest["ema_21"] and prev["ema_9"] <= prev["ema_21"]:
                signals["ema_crossover"] = "BUY"
            elif latest["ema_9"] < latest["ema_21"] and prev["ema_9"] >= prev["ema_21"]:
                signals["ema_crossover"] = "SELL"
            elif latest["ema_9"] > latest["ema_21"]:
                signals["ema_crossover"] = "BULLISH"
            else:
                signals["ema_crossover"] = "BEARISH"
        
        # Price vs Moving Averages
        if "sma_200" in df.columns:
            signals["vs_200dma"] = "ABOVE" if latest["close"] > latest["sma_200"] else "BELOW"
        if "sma_50" in df.columns:
            signals["vs_50dma"] = "ABOVE" if latest["close"] > latest["sma_50"] else "BELOW"
        
        # MACD Signal
        if "macd" in df.columns and "macd_signal" in df.columns:
            if latest["macd"] > latest["macd_signal"] and prev["macd"] <= prev["macd_signal"]:
                signals["macd"] = "BUY"
            elif latest["macd"] < latest["macd_signal"] and prev["macd"] >= prev["macd_signal"]:
                signals["macd"] = "SELL"
            elif latest["macd"] > latest["macd_signal"]:
                signals["macd"] = "BULLISH"
            else:
                signals["macd"] = "BEARISH"
        
        # RSI Signal
        if "rsi" in df.columns:
            rsi = latest["rsi"]
            if rsi < self.settings["rsi_oversold"]:
                signals["rsi"] = "OVERSOLD"
            elif rsi > self.settings["rsi_overbought"]:
                signals["rsi"] = "OVERBOUGHT"
            elif rsi < 45:
                signals["rsi"] = "BEARISH"
            elif rsi > 55:
                signals["rsi"] = "BULLISH"
            else:
                signals["rsi"] = "NEUTRAL"
        
        # Stochastic Signal
        stoch_k = None
        for col in df.columns:
            if "stoch" in col.lower() and "k" in col.lower():
                stoch_k = latest[col]
                break
        
        if stoch_k is not None:
            if stoch_k < 20:
                signals["stochastic"] = "OVERSOLD"
            elif stoch_k > 80:
                signals["stochastic"] = "OVERBOUGHT"
            else:
                signals["stochastic"] = "NEUTRAL"
        
        # ADX Signal
        adx_col = None
        for col in df.columns:
            if col.upper().startswith("ADX"):
                adx_col = col
                break
        
        if adx_col and adx_col in df.columns:
            adx = latest[adx_col]
            if adx > self.settings["adx_strong_trend"]:
                signals["adx"] = "STRONG_TREND"
            elif adx > 20:
                signals["adx"] = "TRENDING"
            else:
                signals["adx"] = "NO_TREND"
        
        # Bollinger Bands Signal
        if "bb_upper" in df.columns and "bb_lower" in df.columns:
            if latest["close"] > latest["bb_upper"]:
                signals["bollinger"] = "OVERBOUGHT"
            elif latest["close"] < latest["bb_lower"]:
                signals["bollinger"] = "OVERSOLD"
            else:
                signals["bollinger"] = "NEUTRAL"
        
        # Volume Signal
        if "volume_ratio" in df.columns:
            vol_ratio = latest["volume_ratio"]
            if vol_ratio > 2:
                signals["volume"] = "HIGH"
            elif vol_ratio > 1.5:
                signals["volume"] = "ABOVE_AVERAGE"
            elif vol_ratio < 0.5:
                signals["volume"] = "LOW"
            else:
                signals["volume"] = "NORMAL"
        
        return signals
    
    def identify_support_resistance(
        self,
        df: pd.DataFrame,
        lookback: int = 100,
        num_levels: int = 3,
    ) -> Dict[str, List[float]]:
        """
        Identify support and resistance levels.
        
        Args:
            df: OHLCV dataframe
            lookback: Number of bars to analyze
            num_levels: Number of levels to return
        
        Returns:
            Dictionary with support and resistance levels
        """
        if df.empty or len(df) < lookback:
            lookback = len(df)
        
        recent = df.tail(lookback)
        current_price = df["close"].iloc[-1]
        
        # Find pivot highs and lows
        pivots_high = []
        pivots_low = []
        
        for i in range(2, len(recent) - 2):
            # Pivot high
            if (recent["high"].iloc[i] > recent["high"].iloc[i-1] and
                recent["high"].iloc[i] > recent["high"].iloc[i-2] and
                recent["high"].iloc[i] > recent["high"].iloc[i+1] and
                recent["high"].iloc[i] > recent["high"].iloc[i+2]):
                pivots_high.append(recent["high"].iloc[i])
            
            # Pivot low
            if (recent["low"].iloc[i] < recent["low"].iloc[i-1] and
                recent["low"].iloc[i] < recent["low"].iloc[i-2] and
                recent["low"].iloc[i] < recent["low"].iloc[i+1] and
                recent["low"].iloc[i] < recent["low"].iloc[i+2]):
                pivots_low.append(recent["low"].iloc[i])
        
        # Cluster nearby levels
        supports = self._cluster_levels([p for p in pivots_low if p < current_price])
        resistances = self._cluster_levels([p for p in pivots_high if p > current_price])
        
        # Add recent high/low
        recent_high = recent["high"].max()
        recent_low = recent["low"].min()
        
        if recent_high > current_price and recent_high not in resistances:
            resistances.append(recent_high)
        if recent_low < current_price and recent_low not in supports:
            supports.append(recent_low)
        
        # Sort and limit
        supports = sorted(supports, reverse=True)[:num_levels]
        resistances = sorted(resistances)[:num_levels]
        
        return {
            "support": [round(s, 2) for s in supports],
            "resistance": [round(r, 2) for r in resistances],
            "current_price": round(current_price, 2),
        }
    
    def _cluster_levels(self, levels: List[float], threshold: float = 0.02) -> List[float]:
        """
        Cluster nearby price levels.
        
        Args:
            levels: List of price levels
            threshold: Clustering threshold (2%)
        
        Returns:
            List of clustered levels
        """
        if not levels:
            return []
        
        sorted_levels = sorted(levels)
        clusters = [[sorted_levels[0]]]
        
        for level in sorted_levels[1:]:
            if (level - clusters[-1][0]) / clusters[-1][0] < threshold:
                clusters[-1].append(level)
            else:
                clusters.append([level])
        
        # Return average of each cluster
        return [sum(c) / len(c) for c in clusters]
    
    def get_trend_strength(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze trend strength.
        
        Args:
            df: DataFrame with indicators
        
        Returns:
            Dictionary with trend analysis
        """
        if df.empty:
            return {"trend": "UNKNOWN", "strength": 0}
        
        latest = df.iloc[-1]
        
        # Count bullish vs bearish signals
        bullish_count = 0
        bearish_count = 0
        
        # Price vs Moving Averages
        mas = ["sma_20", "sma_50", "sma_100", "sma_200"]
        for ma in mas:
            if ma in df.columns:
                if latest["close"] > latest[ma]:
                    bullish_count += 1
                else:
                    bearish_count += 1
        
        # Moving Average alignment
        if all(f"sma_{p}" in df.columns for p in [20, 50, 200]):
            if latest["sma_20"] > latest["sma_50"] > latest["sma_200"]:
                bullish_count += 2
            elif latest["sma_20"] < latest["sma_50"] < latest["sma_200"]:
                bearish_count += 2
        
        # MACD
        if "macd" in df.columns:
            if latest["macd"] > 0:
                bullish_count += 1
            else:
                bearish_count += 1
        
        # RSI
        if "rsi" in df.columns:
            if latest["rsi"] > 50:
                bullish_count += 1
            else:
                bearish_count += 1
        
        # ADX for trend strength
        adx_value = 0
        for col in df.columns:
            if col.upper().startswith("ADX"):
                adx_value = latest[col]
                break
        
        total = bullish_count + bearish_count
        if total == 0:
            return {"trend": "NEUTRAL", "strength": 50}
        
        bullish_pct = bullish_count / total
        
        if bullish_pct > 0.7:
            trend = "BULLISH"
        elif bullish_pct > 0.55:
            trend = "SLIGHTLY_BULLISH"
        elif bullish_pct < 0.3:
            trend = "BEARISH"
        elif bullish_pct < 0.45:
            trend = "SLIGHTLY_BEARISH"
        else:
            trend = "NEUTRAL"
        
        # Strength (0-100)
        strength = abs(bullish_pct - 0.5) * 200
        if adx_value > 25:
            strength = min(100, strength * 1.2)
        
        return {
            "trend": trend,
            "strength": round(strength, 0),
            "bullish_signals": bullish_count,
            "bearish_signals": bearish_count,
            "adx": round(adx_value, 2) if adx_value else None,
        }
    
    def generate_technical_summary(
        self,
        df: pd.DataFrame,
        symbol: str = "",
    ) -> Dict[str, Any]:
        """
        Generate comprehensive technical analysis summary.
        
        Args:
            df: OHLCV dataframe
            symbol: Stock symbol
        
        Returns:
            Dictionary with technical summary
        """
        # Calculate all indicators
        df = self.calculate_all_indicators(df)
        
        if df.empty:
            return {"error": "No data available"}
        
        latest = df.iloc[-1]
        
        # Get individual analyses
        signals = self.get_indicator_signals(df)
        sr_levels = self.identify_support_resistance(df)
        trend = self.get_trend_strength(df)
        
        # Calculate overall technical score
        score = self._calculate_technical_score(signals, trend)
        
        # Determine momentum
        rsi = latest.get("rsi", 50)
        if rsi < 30:
            momentum = "OVERSOLD"
        elif rsi > 70:
            momentum = "OVERBOUGHT"
        else:
            momentum = "NEUTRAL"
        
        # Determine volatility
        atr = latest.get("atr", 0)
        avg_price = latest.get("close", 1)
        atr_pct = (atr / avg_price * 100) if avg_price > 0 else 0
        
        if atr_pct > 3:
            volatility = "HIGH"
        elif atr_pct > 1.5:
            volatility = "NORMAL"
        else:
            volatility = "LOW"
        
        # Volume trend
        vol_ratio = latest.get("volume_ratio", 1)
        obv_change = 0
        if "obv" in df.columns and len(df) > 5:
            obv_change = (df["obv"].iloc[-1] - df["obv"].iloc[-5]) / abs(df["obv"].iloc[-5]) * 100 if df["obv"].iloc[-5] != 0 else 0
        
        if vol_ratio > 1.5 and obv_change > 0:
            volume_trend = "ACCUMULATION"
        elif vol_ratio > 1.5 and obv_change < 0:
            volume_trend = "DISTRIBUTION"
        else:
            volume_trend = "NEUTRAL"
        
        # Suggested stop loss and targets
        atr_value = latest.get("atr", 0)
        current_price = latest.get("close", 0)
        
        stop_loss = round(current_price - (2 * atr_value), 2) if atr_value else None
        target_1 = round(current_price + (2 * atr_value), 2) if atr_value else None
        target_2 = round(current_price + (4 * atr_value), 2) if atr_value else None
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "current_price": round(current_price, 2),
            "trend": trend["trend"],
            "trend_strength": trend["strength"],
            "momentum": momentum,
            "volatility": volatility,
            "volume_trend": volume_trend,
            "technical_score": score,
            "key_levels": {
                "support": sr_levels["support"],
                "resistance": sr_levels["resistance"],
                "stop_loss_suggested": stop_loss,
                "target_1": target_1,
                "target_2": target_2,
            },
            "indicator_signals": signals,
            "key_values": {
                "rsi": round(rsi, 2),
                "macd": round(latest.get("macd", 0), 2),
                "atr": round(atr, 2),
                "atr_percent": round(atr_pct, 2),
                "volume_ratio": round(vol_ratio, 2),
            },
            "moving_averages": {
                "sma_20": round(latest.get("sma_20", 0), 2),
                "sma_50": round(latest.get("sma_50", 0), 2),
                "sma_200": round(latest.get("sma_200", 0), 2),
                "ema_9": round(latest.get("ema_9", 0), 2),
                "ema_21": round(latest.get("ema_21", 0), 2),
            },
        }
    
    def _calculate_technical_score(
        self,
        signals: Dict[str, str],
        trend: Dict[str, Any],
    ) -> float:
        """
        Calculate overall technical score (0-100).
        
        Args:
            signals: Indicator signals
            trend: Trend analysis
        
        Returns:
            Technical score
        """
        score = 50  # Start neutral
        
        # Trend contribution (max ±20)
        trend_type = trend.get("trend", "NEUTRAL")
        if trend_type == "BULLISH":
            score += 15
        elif trend_type == "SLIGHTLY_BULLISH":
            score += 8
        elif trend_type == "BEARISH":
            score -= 15
        elif trend_type == "SLIGHTLY_BEARISH":
            score -= 8
        
        # Trend strength contribution
        strength = trend.get("strength", 50)
        score += (strength - 50) / 5  # ±10 points
        
        # Signal contributions
        signal_scores = {
            "BUY": 5, "BULLISH": 3, "ABOVE": 3,
            "SELL": -5, "BEARISH": -3, "BELOW": -3,
            "OVERSOLD": 2, "OVERBOUGHT": -2,
            "STRONG_TREND": 3, "HIGH": 2,
        }
        
        for signal_name, signal_value in signals.items():
            score += signal_scores.get(signal_value, 0)
        
        return max(0, min(100, score))


# Convenience function
def analyze_technical(df: pd.DataFrame, symbol: str = "") -> Dict:
    """Perform technical analysis on a dataframe."""
    analyzer = TechnicalAnalyzer()
    return analyzer.generate_technical_summary(df, symbol)
