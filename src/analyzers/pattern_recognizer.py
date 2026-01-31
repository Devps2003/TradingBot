"""
Chart Pattern Recognition Module.

Detects:
- Candlestick patterns (Doji, Hammer, Engulfing, etc.)
- Chart patterns (Head & Shoulders, Double Top/Bottom, Flags, etc.)
- Breakouts (Range, Trendline, Moving Average)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import PATTERN_SETTINGS, PATTERN_HISTORICAL_ACCURACY
from src.utils.logger import LoggerMixin


class PatternRecognizer(LoggerMixin):
    """
    Recognizes chart patterns in price data.
    """
    
    def __init__(self):
        """Initialize the pattern recognizer."""
        self.settings = PATTERN_SETTINGS
        self.historical_accuracy = PATTERN_HISTORICAL_ACCURACY
    
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect candlestick patterns in the data.
        
        Args:
            df: OHLCV dataframe
        
        Returns:
            List of detected patterns
        """
        if df.empty or len(df) < 3:
            return []
        
        patterns = []
        
        # Check last few candles
        for i in range(-3, 0):
            if abs(i) > len(df):
                continue
            
            idx = i
            
            # Single candle patterns
            single_patterns = self._detect_single_candle_patterns(df, idx)
            patterns.extend(single_patterns)
        
        # Two candle patterns (on last 2 candles)
        if len(df) >= 2:
            two_patterns = self._detect_two_candle_patterns(df)
            patterns.extend(two_patterns)
        
        # Three candle patterns
        if len(df) >= 3:
            three_patterns = self._detect_three_candle_patterns(df)
            patterns.extend(three_patterns)
        
        return patterns
    
    def _detect_single_candle_patterns(
        self,
        df: pd.DataFrame,
        idx: int,
    ) -> List[Dict[str, Any]]:
        """
        Detect single candlestick patterns.
        
        Args:
            df: OHLCV dataframe
            idx: Index to check
        
        Returns:
            List of detected patterns
        """
        patterns = []
        
        row = df.iloc[idx]
        open_p = row["open"]
        high = row["high"]
        low = row["low"]
        close = row["close"]
        
        body = abs(close - open_p)
        range_ = high - low
        
        if range_ == 0:
            return patterns
        
        body_pct = body / range_
        upper_shadow = high - max(open_p, close)
        lower_shadow = min(open_p, close) - low
        
        # Doji
        if body_pct < self.settings["doji_body_percent"]:
            patterns.append({
                "pattern_name": "doji",
                "pattern_type": "NEUTRAL",
                "confidence": 70,
                "description": "Doji - indecision candle",
                "idx": idx,
            })
        
        # Hammer (bullish)
        if (body_pct < 0.3 and 
            lower_shadow > 2 * body and 
            upper_shadow < body * 0.5 and
            close > open_p):
            patterns.append({
                "pattern_name": "hammer",
                "pattern_type": "BULLISH",
                "confidence": 60,
                "description": "Hammer - potential reversal at bottom",
                "historical_accuracy": self.historical_accuracy.get("hammer", 0.6),
                "idx": idx,
            })
        
        # Inverted Hammer
        if (body_pct < 0.3 and 
            upper_shadow > 2 * body and 
            lower_shadow < body * 0.5 and
            close > open_p):
            patterns.append({
                "pattern_name": "inverted_hammer",
                "pattern_type": "BULLISH",
                "confidence": 55,
                "description": "Inverted Hammer - potential bullish reversal",
                "idx": idx,
            })
        
        # Shooting Star (bearish)
        if (body_pct < 0.3 and 
            upper_shadow > 2 * body and 
            lower_shadow < body * 0.5 and
            close < open_p):
            patterns.append({
                "pattern_name": "shooting_star",
                "pattern_type": "BEARISH",
                "confidence": 58,
                "description": "Shooting Star - potential reversal at top",
                "historical_accuracy": self.historical_accuracy.get("shooting_star", 0.58),
                "idx": idx,
            })
        
        # Marubozu (strong trend candle)
        if body_pct > 0.9:
            if close > open_p:
                patterns.append({
                    "pattern_name": "bullish_marubozu",
                    "pattern_type": "BULLISH",
                    "confidence": 65,
                    "description": "Bullish Marubozu - strong buying pressure",
                    "idx": idx,
                })
            else:
                patterns.append({
                    "pattern_name": "bearish_marubozu",
                    "pattern_type": "BEARISH",
                    "confidence": 65,
                    "description": "Bearish Marubozu - strong selling pressure",
                    "idx": idx,
                })
        
        return patterns
    
    def _detect_two_candle_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect two-candle patterns.
        
        Args:
            df: OHLCV dataframe
        
        Returns:
            List of detected patterns
        """
        patterns = []
        
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        
        prev_body = prev["close"] - prev["open"]
        curr_body = curr["close"] - curr["open"]
        
        # Bullish Engulfing
        if (prev_body < 0 and curr_body > 0 and
            curr["open"] < prev["close"] and
            curr["close"] > prev["open"] and
            abs(curr_body) > abs(prev_body)):
            patterns.append({
                "pattern_name": "bullish_engulfing",
                "pattern_type": "BULLISH",
                "confidence": 65,
                "description": "Bullish Engulfing - strong reversal signal",
                "historical_accuracy": self.historical_accuracy.get("bullish_engulfing", 0.63),
            })
        
        # Bearish Engulfing
        if (prev_body > 0 and curr_body < 0 and
            curr["open"] > prev["close"] and
            curr["close"] < prev["open"] and
            abs(curr_body) > abs(prev_body)):
            patterns.append({
                "pattern_name": "bearish_engulfing",
                "pattern_type": "BEARISH",
                "confidence": 64,
                "description": "Bearish Engulfing - potential reversal at top",
                "historical_accuracy": self.historical_accuracy.get("bearish_engulfing", 0.62),
            })
        
        # Piercing Line (bullish)
        if (prev_body < 0 and curr_body > 0 and
            curr["open"] < prev["low"] and
            curr["close"] > (prev["open"] + prev["close"]) / 2 and
            curr["close"] < prev["open"]):
            patterns.append({
                "pattern_name": "piercing_line",
                "pattern_type": "BULLISH",
                "confidence": 60,
                "description": "Piercing Line - bullish reversal",
            })
        
        # Dark Cloud Cover (bearish)
        if (prev_body > 0 and curr_body < 0 and
            curr["open"] > prev["high"] and
            curr["close"] < (prev["open"] + prev["close"]) / 2 and
            curr["close"] > prev["open"]):
            patterns.append({
                "pattern_name": "dark_cloud_cover",
                "pattern_type": "BEARISH",
                "confidence": 60,
                "description": "Dark Cloud Cover - bearish reversal",
            })
        
        # Harami
        if (abs(prev_body) > abs(curr_body) * 2 and
            curr["high"] < max(prev["open"], prev["close"]) and
            curr["low"] > min(prev["open"], prev["close"])):
            pattern_type = "BULLISH" if prev_body < 0 else "BEARISH"
            patterns.append({
                "pattern_name": f"{pattern_type.lower()}_harami",
                "pattern_type": pattern_type,
                "confidence": 55,
                "description": f"{pattern_type.capitalize()} Harami - potential reversal",
            })
        
        return patterns
    
    def _detect_three_candle_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect three-candle patterns.
        
        Args:
            df: OHLCV dataframe
        
        Returns:
            List of detected patterns
        """
        patterns = []
        
        c1 = df.iloc[-3]
        c2 = df.iloc[-2]
        c3 = df.iloc[-1]
        
        body1 = c1["close"] - c1["open"]
        body2 = c2["close"] - c2["open"]
        body3 = c3["close"] - c3["open"]
        
        # Morning Star (bullish)
        if (body1 < 0 and abs(body1) > abs(body2) * 2 and
            body3 > 0 and
            max(c2["open"], c2["close"]) < c1["close"] and
            c3["close"] > (c1["open"] + c1["close"]) / 2):
            patterns.append({
                "pattern_name": "morning_star",
                "pattern_type": "BULLISH",
                "confidence": 68,
                "description": "Morning Star - strong bullish reversal",
                "historical_accuracy": self.historical_accuracy.get("morning_star", 0.66),
            })
        
        # Evening Star (bearish)
        if (body1 > 0 and abs(body1) > abs(body2) * 2 and
            body3 < 0 and
            min(c2["open"], c2["close"]) > c1["close"] and
            c3["close"] < (c1["open"] + c1["close"]) / 2):
            patterns.append({
                "pattern_name": "evening_star",
                "pattern_type": "BEARISH",
                "confidence": 66,
                "description": "Evening Star - strong bearish reversal",
                "historical_accuracy": self.historical_accuracy.get("evening_star", 0.64),
            })
        
        # Three White Soldiers
        if (body1 > 0 and body2 > 0 and body3 > 0 and
            c2["close"] > c1["close"] and c3["close"] > c2["close"] and
            c2["open"] > c1["open"] and c3["open"] > c2["open"]):
            patterns.append({
                "pattern_name": "three_white_soldiers",
                "pattern_type": "BULLISH",
                "confidence": 70,
                "description": "Three White Soldiers - strong bullish continuation",
            })
        
        # Three Black Crows
        if (body1 < 0 and body2 < 0 and body3 < 0 and
            c2["close"] < c1["close"] and c3["close"] < c2["close"] and
            c2["open"] < c1["open"] and c3["open"] < c2["open"]):
            patterns.append({
                "pattern_name": "three_black_crows",
                "pattern_type": "BEARISH",
                "confidence": 70,
                "description": "Three Black Crows - strong bearish signal",
            })
        
        return patterns
    
    def detect_chart_patterns(
        self,
        df: pd.DataFrame,
        min_bars: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Detect larger chart patterns.
        
        Args:
            df: OHLCV dataframe
            min_bars: Minimum bars for pattern detection
        
        Returns:
            List of detected patterns
        """
        if len(df) < min_bars:
            return []
        
        patterns = []
        
        # Double Bottom
        double_bottom = self._detect_double_bottom(df)
        if double_bottom:
            patterns.append(double_bottom)
        
        # Double Top
        double_top = self._detect_double_top(df)
        if double_top:
            patterns.append(double_top)
        
        # Flag patterns
        bull_flag = self._detect_bull_flag(df)
        if bull_flag:
            patterns.append(bull_flag)
        
        bear_flag = self._detect_bear_flag(df)
        if bear_flag:
            patterns.append(bear_flag)
        
        # Triangles
        triangle = self._detect_triangle(df)
        if triangle:
            patterns.append(triangle)
        
        return patterns
    
    def _detect_double_bottom(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Detect double bottom pattern.
        
        Args:
            df: OHLCV dataframe
        
        Returns:
            Pattern dictionary or None
        """
        lookback = min(60, len(df))
        recent = df.tail(lookback)
        
        # Find two lows
        lows = recent["low"].values
        low_indices = []
        
        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                low_indices.append((i, lows[i]))
        
        if len(low_indices) < 2:
            return None
        
        # Check last two lows
        idx1, low1 = low_indices[-2]
        idx2, low2 = low_indices[-1]
        
        # Lows should be similar (within 3%)
        if abs(low1 - low2) / low1 > 0.03:
            return None
        
        # Should have some separation
        if idx2 - idx1 < 5:
            return None
        
        # Current price should be above the neckline
        neckline = recent["high"].iloc[idx1:idx2].max()
        current = df["close"].iloc[-1]
        
        if current > neckline:
            expected_move = (neckline - low1) / low1 * 100
            return {
                "pattern_name": "double_bottom",
                "pattern_type": "BULLISH",
                "confidence": 72,
                "description": "Double Bottom - bullish reversal pattern",
                "entry_price": current,
                "stop_loss": min(low1, low2) * 0.98,
                "target": neckline + (neckline - min(low1, low2)),
                "expected_move": round(expected_move, 2),
                "historical_accuracy": self.historical_accuracy.get("double_bottom", 0.72),
            }
        
        return None
    
    def _detect_double_top(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Detect double top pattern.
        
        Args:
            df: OHLCV dataframe
        
        Returns:
            Pattern dictionary or None
        """
        lookback = min(60, len(df))
        recent = df.tail(lookback)
        
        # Find two highs
        highs = recent["high"].values
        high_indices = []
        
        for i in range(2, len(highs) - 2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                high_indices.append((i, highs[i]))
        
        if len(high_indices) < 2:
            return None
        
        # Check last two highs
        idx1, high1 = high_indices[-2]
        idx2, high2 = high_indices[-1]
        
        # Highs should be similar (within 3%)
        if abs(high1 - high2) / high1 > 0.03:
            return None
        
        # Should have some separation
        if idx2 - idx1 < 5:
            return None
        
        # Current price should be below the neckline
        neckline = recent["low"].iloc[idx1:idx2].min()
        current = df["close"].iloc[-1]
        
        if current < neckline:
            expected_move = (high1 - neckline) / high1 * 100
            return {
                "pattern_name": "double_top",
                "pattern_type": "BEARISH",
                "confidence": 70,
                "description": "Double Top - bearish reversal pattern",
                "entry_price": current,
                "stop_loss": max(high1, high2) * 1.02,
                "target": neckline - (max(high1, high2) - neckline),
                "expected_move": round(-expected_move, 2),
                "historical_accuracy": self.historical_accuracy.get("double_top", 0.70),
            }
        
        return None
    
    def _detect_bull_flag(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Detect bull flag pattern.
        
        Args:
            df: OHLCV dataframe
        
        Returns:
            Pattern dictionary or None
        """
        lookback = min(30, len(df))
        recent = df.tail(lookback)
        
        # Look for strong move up (pole)
        first_half = recent.iloc[:lookback//2]
        second_half = recent.iloc[lookback//2:]
        
        pole_gain = (first_half["close"].iloc[-1] - first_half["close"].iloc[0]) / first_half["close"].iloc[0]
        
        if pole_gain < 0.05:  # At least 5% move
            return None
        
        # Flag should consolidate with lower highs
        flag_highs = second_half["high"].values
        flag_lows = second_half["low"].values
        
        # Check for slight downward drift
        high_slope = (flag_highs[-1] - flag_highs[0]) / len(flag_highs)
        low_slope = (flag_lows[-1] - flag_lows[0]) / len(flag_lows)
        
        if high_slope < 0 and low_slope < 0:  # Downward sloping flag
            current = df["close"].iloc[-1]
            flag_high = second_half["high"].max()
            
            if current > flag_high * 0.98:  # Near breakout
                return {
                    "pattern_name": "bull_flag",
                    "pattern_type": "BULLISH",
                    "confidence": 68,
                    "description": "Bull Flag - continuation pattern",
                    "entry_price": flag_high,
                    "stop_loss": second_half["low"].min() * 0.98,
                    "target": flag_high + (first_half["high"].max() - first_half["low"].min()),
                    "expected_move": round(pole_gain * 100, 2),
                    "historical_accuracy": self.historical_accuracy.get("bull_flag", 0.68),
                }
        
        return None
    
    def _detect_bear_flag(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Detect bear flag pattern.
        
        Args:
            df: OHLCV dataframe
        
        Returns:
            Pattern dictionary or None
        """
        lookback = min(30, len(df))
        recent = df.tail(lookback)
        
        # Look for strong move down (pole)
        first_half = recent.iloc[:lookback//2]
        second_half = recent.iloc[lookback//2:]
        
        pole_loss = (first_half["close"].iloc[-1] - first_half["close"].iloc[0]) / first_half["close"].iloc[0]
        
        if pole_loss > -0.05:  # At least 5% move down
            return None
        
        # Flag should consolidate with higher lows
        flag_highs = second_half["high"].values
        flag_lows = second_half["low"].values
        
        # Check for slight upward drift
        high_slope = (flag_highs[-1] - flag_highs[0]) / len(flag_highs)
        low_slope = (flag_lows[-1] - flag_lows[0]) / len(flag_lows)
        
        if high_slope > 0 and low_slope > 0:  # Upward sloping flag
            current = df["close"].iloc[-1]
            flag_low = second_half["low"].min()
            
            if current < flag_low * 1.02:  # Near breakdown
                return {
                    "pattern_name": "bear_flag",
                    "pattern_type": "BEARISH",
                    "confidence": 65,
                    "description": "Bear Flag - continuation pattern",
                    "entry_price": flag_low,
                    "stop_loss": second_half["high"].max() * 1.02,
                    "target": flag_low - (first_half["high"].max() - first_half["low"].min()),
                    "expected_move": round(pole_loss * 100, 2),
                    "historical_accuracy": self.historical_accuracy.get("bear_flag", 0.65),
                }
        
        return None
    
    def _detect_triangle(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Detect triangle patterns.
        
        Args:
            df: OHLCV dataframe
        
        Returns:
            Pattern dictionary or None
        """
        lookback = min(40, len(df))
        recent = df.tail(lookback)
        
        # Find swing highs and lows
        highs = []
        lows = []
        
        for i in range(2, len(recent) - 2):
            if (recent["high"].iloc[i] > recent["high"].iloc[i-1] and
                recent["high"].iloc[i] > recent["high"].iloc[i-2] and
                recent["high"].iloc[i] > recent["high"].iloc[i+1] and
                recent["high"].iloc[i] > recent["high"].iloc[i+2]):
                highs.append((i, recent["high"].iloc[i]))
            
            if (recent["low"].iloc[i] < recent["low"].iloc[i-1] and
                recent["low"].iloc[i] < recent["low"].iloc[i-2] and
                recent["low"].iloc[i] < recent["low"].iloc[i+1] and
                recent["low"].iloc[i] < recent["low"].iloc[i+2]):
                lows.append((i, recent["low"].iloc[i]))
        
        if len(highs) < 2 or len(lows) < 2:
            return None
        
        # Check for converging lines
        high_slope = (highs[-1][1] - highs[0][1]) / (highs[-1][0] - highs[0][0]) if highs[-1][0] != highs[0][0] else 0
        low_slope = (lows[-1][1] - lows[0][1]) / (lows[-1][0] - lows[0][0]) if lows[-1][0] != lows[0][0] else 0
        
        current = df["close"].iloc[-1]
        
        if high_slope < 0 and low_slope > 0:
            # Symmetrical triangle
            return {
                "pattern_name": "symmetrical_triangle",
                "pattern_type": "CONTINUATION",
                "confidence": 60,
                "description": "Symmetrical Triangle - wait for breakout direction",
                "upper_line": highs[-1][1],
                "lower_line": lows[-1][1],
            }
        elif high_slope < 0 and low_slope >= 0:
            # Ascending triangle (bullish)
            return {
                "pattern_name": "ascending_triangle",
                "pattern_type": "BULLISH",
                "confidence": 70,
                "description": "Ascending Triangle - bullish breakout likely",
                "entry_price": highs[-1][1],
                "stop_loss": lows[-1][1] * 0.98,
                "historical_accuracy": self.historical_accuracy.get("ascending_triangle", 0.70),
            }
        elif high_slope >= 0 and low_slope > 0:
            # Descending triangle (bearish)
            return {
                "pattern_name": "descending_triangle",
                "pattern_type": "BEARISH",
                "confidence": 68,
                "description": "Descending Triangle - bearish breakdown likely",
                "entry_price": lows[-1][1],
                "stop_loss": highs[-1][1] * 1.02,
                "historical_accuracy": self.historical_accuracy.get("descending_triangle", 0.68),
            }
        
        return None
    
    def detect_breakouts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect breakout signals.
        
        Args:
            df: OHLCV dataframe
        
        Returns:
            List of breakout signals
        """
        if len(df) < 20:
            return []
        
        breakouts = []
        current = df["close"].iloc[-1]
        prev = df["close"].iloc[-2]
        
        # Range breakout
        lookback = 20
        recent = df.tail(lookback)
        range_high = recent["high"].max()
        range_low = recent["low"].min()
        
        if current > range_high and prev <= range_high:
            breakouts.append({
                "type": "range_breakout_up",
                "pattern_type": "BULLISH",
                "level": round(range_high, 2),
                "current_price": round(current, 2),
                "confidence": 65,
                "description": f"Breakout above {lookback}-day range high",
            })
        elif current < range_low and prev >= range_low:
            breakouts.append({
                "type": "range_breakout_down",
                "pattern_type": "BEARISH",
                "level": round(range_low, 2),
                "current_price": round(current, 2),
                "confidence": 65,
                "description": f"Breakdown below {lookback}-day range low",
            })
        
        # Moving average breakouts
        if len(df) >= 50:
            sma_50 = df["close"].rolling(50).mean().iloc[-1]
            sma_50_prev = df["close"].rolling(50).mean().iloc[-2]
            
            if current > sma_50 and prev <= sma_50_prev:
                breakouts.append({
                    "type": "ma_breakout_50",
                    "pattern_type": "BULLISH",
                    "level": round(sma_50, 2),
                    "confidence": 60,
                    "description": "Breakout above 50-day moving average",
                })
            elif current < sma_50 and prev >= sma_50_prev:
                breakouts.append({
                    "type": "ma_breakdown_50",
                    "pattern_type": "BEARISH",
                    "level": round(sma_50, 2),
                    "confidence": 60,
                    "description": "Breakdown below 50-day moving average",
                })
        
        # Volume breakout
        avg_volume = df["volume"].rolling(20).mean().iloc[-1]
        if df["volume"].iloc[-1] > avg_volume * 2:
            breakouts.append({
                "type": "volume_breakout",
                "pattern_type": "BULLISH" if current > prev else "BEARISH",
                "volume_ratio": round(df["volume"].iloc[-1] / avg_volume, 2),
                "confidence": 55,
                "description": "Unusual volume detected",
            })
        
        return breakouts
    
    def get_pattern_reliability_score(self, pattern: str) -> float:
        """
        Get historical reliability score for a pattern.
        
        Args:
            pattern: Pattern name
        
        Returns:
            Reliability score (0-1)
        """
        return self.historical_accuracy.get(pattern.lower(), 0.5)
    
    def generate_pattern_summary(
        self,
        df: pd.DataFrame,
        symbol: str = "",
    ) -> Dict[str, Any]:
        """
        Generate comprehensive pattern analysis summary.
        
        Args:
            df: OHLCV dataframe
            symbol: Stock symbol
        
        Returns:
            Pattern summary dictionary
        """
        candlestick = self.detect_candlestick_patterns(df)
        chart = self.detect_chart_patterns(df)
        breakouts = self.detect_breakouts(df)
        
        # Find most significant pattern
        all_patterns = candlestick + chart
        best_pattern = None
        if all_patterns:
            best_pattern = max(all_patterns, key=lambda x: x.get("confidence", 0))
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "candlestick_patterns": candlestick,
            "chart_patterns": chart,
            "breakouts": breakouts,
            "total_patterns": len(all_patterns),
            "best_pattern": best_pattern,
            "overall_bias": self._determine_overall_bias(all_patterns),
            "pattern_score": self._calculate_pattern_score(all_patterns),
        }
    
    def _determine_overall_bias(self, patterns: List[Dict]) -> str:
        """Determine overall bias from patterns."""
        if not patterns:
            return "NEUTRAL"
        
        bullish = sum(1 for p in patterns if p.get("pattern_type") == "BULLISH")
        bearish = sum(1 for p in patterns if p.get("pattern_type") == "BEARISH")
        
        if bullish > bearish + 1:
            return "BULLISH"
        elif bearish > bullish + 1:
            return "BEARISH"
        return "NEUTRAL"
    
    def _calculate_pattern_score(self, patterns: List[Dict]) -> float:
        """Calculate pattern-based score."""
        if not patterns:
            return 50
        
        score = 50
        for p in patterns:
            confidence = p.get("confidence", 50)
            pattern_type = p.get("pattern_type", "NEUTRAL")
            
            adjustment = (confidence - 50) / 10
            if pattern_type == "BEARISH":
                adjustment *= -1
            
            score += adjustment
        
        return max(0, min(100, score))


# Convenience function
def detect_patterns(df: pd.DataFrame, symbol: str = "") -> Dict:
    """Detect patterns in a dataframe."""
    recognizer = PatternRecognizer()
    return recognizer.generate_pattern_summary(df, symbol)
