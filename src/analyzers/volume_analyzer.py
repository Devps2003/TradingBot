"""
Volume Analysis Module.

Analyzes volume patterns including:
- Volume trend vs price trend
- Unusual volume detection
- Volume profile
- Delivery percentage (NSE specific)
- Accumulation/Distribution
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_fetchers.price_fetcher import PriceFetcher
from src.utils.logger import LoggerMixin


class VolumeAnalyzer(LoggerMixin):
    """
    Performs volume analysis on stock data.
    """
    
    def __init__(self):
        """Initialize the volume analyzer."""
        self.price_fetcher = PriceFetcher()
    
    def analyze_volume_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze volume trend relative to price.
        
        Args:
            df: OHLCV dataframe
        
        Returns:
            Volume trend analysis
        """
        if df.empty or len(df) < 20:
            return {"error": "Insufficient data"}
        
        # Calculate volume metrics
        df = df.copy()
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
        df["price_change"] = df["close"].pct_change()
        
        latest = df.iloc[-1]
        
        # Current volume status
        vol_ratio = latest["volume_ratio"]
        if vol_ratio > 2:
            volume_status = "VERY_HIGH"
        elif vol_ratio > 1.5:
            volume_status = "HIGH"
        elif vol_ratio > 0.8:
            volume_status = "NORMAL"
        elif vol_ratio > 0.5:
            volume_status = "LOW"
        else:
            volume_status = "VERY_LOW"
        
        # Volume trend (last 5 days)
        recent_vol = df["volume"].tail(5).mean()
        older_vol = df["volume"].tail(20).head(15).mean()
        vol_trend_change = (recent_vol - older_vol) / older_vol * 100 if older_vol > 0 else 0
        
        if vol_trend_change > 30:
            volume_trend = "INCREASING"
        elif vol_trend_change < -30:
            volume_trend = "DECREASING"
        else:
            volume_trend = "STABLE"
        
        # Price-volume relationship
        recent_data = df.tail(5)
        price_up = (recent_data["close"].iloc[-1] > recent_data["close"].iloc[0])
        vol_up = (recent_data["volume"].mean() > df["volume_sma_20"].iloc[-1])
        
        if price_up and vol_up:
            pv_relationship = "CONFIRMED_UPTREND"
        elif not price_up and vol_up:
            pv_relationship = "DISTRIBUTION"
        elif price_up and not vol_up:
            pv_relationship = "WEAK_UPTREND"
        else:
            pv_relationship = "CONFIRMED_DOWNTREND"
        
        return {
            "current_volume": int(latest["volume"]),
            "volume_sma_20": int(latest["volume_sma_20"]),
            "volume_ratio": round(vol_ratio, 2),
            "volume_status": volume_status,
            "volume_trend": volume_trend,
            "volume_trend_change_pct": round(vol_trend_change, 2),
            "price_volume_relationship": pv_relationship,
        }
    
    def detect_unusual_volume(
        self,
        df: pd.DataFrame,
        threshold: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Detect unusual volume spikes.
        
        Args:
            df: OHLCV dataframe
            threshold: Multiple of average volume
        
        Returns:
            Unusual volume detection result
        """
        if df.empty or len(df) < 20:
            return {"unusual_detected": False}
        
        df = df.copy()
        avg_volume = df["volume"].rolling(20).mean()
        
        latest_vol = df["volume"].iloc[-1]
        latest_avg = avg_volume.iloc[-1]
        
        ratio = latest_vol / latest_avg if latest_avg > 0 else 1
        unusual = ratio >= threshold
        
        # Find recent unusual volume days
        df["vol_ratio"] = df["volume"] / avg_volume
        unusual_days = df[df["vol_ratio"] >= threshold].tail(10)
        
        return {
            "unusual_detected": unusual,
            "current_ratio": round(ratio, 2),
            "threshold": threshold,
            "current_volume": int(latest_vol),
            "average_volume": int(latest_avg),
            "recent_unusual_days": len(unusual_days),
            "interpretation": self._interpret_unusual_volume(unusual, ratio, df),
        }
    
    def _interpret_unusual_volume(
        self,
        unusual: bool,
        ratio: float,
        df: pd.DataFrame,
    ) -> str:
        """Interpret unusual volume."""
        if not unusual:
            return "Normal trading volume"
        
        price_change = df["close"].pct_change().iloc[-1]
        
        if price_change > 0.02:
            return f"Breakout with {ratio:.1f}x volume - strong buying interest"
        elif price_change < -0.02:
            return f"Breakdown with {ratio:.1f}x volume - strong selling pressure"
        else:
            return f"High volume churning ({ratio:.1f}x) - watch for direction"
    
    def calculate_volume_profile(
        self,
        df: pd.DataFrame,
        num_levels: int = 10,
    ) -> Dict[str, Any]:
        """
        Calculate volume profile (price levels with most volume).
        
        Args:
            df: OHLCV dataframe
            num_levels: Number of price levels
        
        Returns:
            Volume profile analysis
        """
        if df.empty or len(df) < 20:
            return {"error": "Insufficient data"}
        
        # Calculate price range
        price_min = df["low"].min()
        price_max = df["high"].max()
        price_range = price_max - price_min
        level_size = price_range / num_levels
        
        # Create price bins
        levels = []
        for i in range(num_levels):
            level_low = price_min + (i * level_size)
            level_high = level_low + level_size
            level_mid = (level_low + level_high) / 2
            
            # Calculate volume at this level
            mask = (df["close"] >= level_low) & (df["close"] < level_high)
            level_volume = df.loc[mask, "volume"].sum()
            
            levels.append({
                "price_low": round(level_low, 2),
                "price_high": round(level_high, 2),
                "price_mid": round(level_mid, 2),
                "volume": int(level_volume),
            })
        
        # Find POC (Point of Control) - highest volume level
        total_volume = sum(l["volume"] for l in levels)
        poc_level = max(levels, key=lambda x: x["volume"])
        
        # Find value area (70% of volume)
        sorted_levels = sorted(levels, key=lambda x: x["volume"], reverse=True)
        value_area_volume = 0
        value_area_levels = []
        
        for level in sorted_levels:
            value_area_levels.append(level)
            value_area_volume += level["volume"]
            if value_area_volume >= total_volume * 0.7:
                break
        
        va_prices = [l["price_mid"] for l in value_area_levels]
        
        current_price = df["close"].iloc[-1]
        
        return {
            "levels": levels,
            "poc": {
                "price": poc_level["price_mid"],
                "volume": poc_level["volume"],
            },
            "value_area": {
                "high": max(va_prices),
                "low": min(va_prices),
            },
            "current_price": round(current_price, 2),
            "price_vs_poc": "ABOVE" if current_price > poc_level["price_mid"] else "BELOW",
            "price_in_value_area": min(va_prices) <= current_price <= max(va_prices),
        }
    
    def analyze_delivery_percentage(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze delivery percentage (NSE specific).
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Delivery analysis
        """
        delivery_data = self.price_fetcher.get_delivery_data(symbol)
        
        if "error" in delivery_data:
            return delivery_data
        
        delivery_pct = delivery_data.get("delivery_percentage")
        
        if delivery_pct is None:
            return {
                "symbol": symbol,
                "delivery_percentage": None,
                "assessment": "UNKNOWN",
            }
        
        # Interpret delivery percentage
        if delivery_pct > 50:
            assessment = "HIGH_CONVICTION"
            interpretation = "High delivery suggests genuine buying/selling"
        elif delivery_pct > 35:
            assessment = "MODERATE"
            interpretation = "Moderate delivery, mix of traders and investors"
        else:
            assessment = "SPECULATIVE"
            interpretation = "Low delivery suggests speculative trading"
        
        return {
            "symbol": symbol,
            "delivery_percentage": delivery_pct,
            "traded_quantity": delivery_data.get("traded_quantity"),
            "delivered_quantity": delivery_data.get("deliverable_quantity"),
            "assessment": assessment,
            "interpretation": interpretation,
        }
    
    def detect_accumulation_distribution(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Detect accumulation or distribution patterns.
        
        Args:
            df: OHLCV dataframe
        
        Returns:
            Accumulation/Distribution analysis
        """
        if df.empty or len(df) < 20:
            return {"error": "Insufficient data"}
        
        df = df.copy()
        
        # Calculate A/D Line
        df["clv"] = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
        df["clv"] = df["clv"].fillna(0)
        df["ad"] = (df["clv"] * df["volume"]).cumsum()
        
        # Calculate OBV
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        
        # Analyze trends
        ad_recent = df["ad"].tail(10).mean()
        ad_older = df["ad"].tail(30).head(20).mean()
        ad_trend = (ad_recent - ad_older) / abs(ad_older) * 100 if ad_older != 0 else 0
        
        obv_recent = df["obv"].tail(10).mean()
        obv_older = df["obv"].tail(30).head(20).mean()
        obv_trend = (obv_recent - obv_older) / abs(obv_older) * 100 if obv_older != 0 else 0
        
        price_recent = df["close"].tail(10).mean()
        price_older = df["close"].tail(30).head(20).mean()
        price_trend = (price_recent - price_older) / price_older * 100
        
        # Determine pattern
        if ad_trend > 10 and obv_trend > 10:
            if price_trend > 0:
                pattern = "ACCUMULATION_CONFIRMED"
            else:
                pattern = "HIDDEN_ACCUMULATION"
        elif ad_trend < -10 and obv_trend < -10:
            if price_trend < 0:
                pattern = "DISTRIBUTION_CONFIRMED"
            else:
                pattern = "HIDDEN_DISTRIBUTION"
        else:
            pattern = "NEUTRAL"
        
        # Check for divergence
        divergence = None
        if price_trend > 5 and (ad_trend < -5 or obv_trend < -5):
            divergence = "BEARISH_DIVERGENCE"
        elif price_trend < -5 and (ad_trend > 5 or obv_trend > 5):
            divergence = "BULLISH_DIVERGENCE"
        
        return {
            "pattern": pattern,
            "ad_trend_pct": round(ad_trend, 2),
            "obv_trend_pct": round(obv_trend, 2),
            "price_trend_pct": round(price_trend, 2),
            "divergence": divergence,
            "interpretation": self._interpret_ad_pattern(pattern, divergence),
        }
    
    def _interpret_ad_pattern(
        self,
        pattern: str,
        divergence: Optional[str],
    ) -> str:
        """Interpret A/D pattern."""
        interpretations = {
            "ACCUMULATION_CONFIRMED": "Smart money accumulating - bullish",
            "HIDDEN_ACCUMULATION": "Accumulation despite falling prices - watch for reversal",
            "DISTRIBUTION_CONFIRMED": "Distribution ongoing - bearish",
            "HIDDEN_DISTRIBUTION": "Distribution despite rising prices - caution",
            "NEUTRAL": "No clear accumulation or distribution",
        }
        
        base = interpretations.get(pattern, "Unknown pattern")
        
        if divergence == "BEARISH_DIVERGENCE":
            base += ". Warning: Bearish divergence detected"
        elif divergence == "BULLISH_DIVERGENCE":
            base += ". Note: Bullish divergence detected"
        
        return base
    
    def generate_volume_summary(
        self,
        df: pd.DataFrame,
        symbol: str = "",
    ) -> Dict[str, Any]:
        """
        Generate comprehensive volume analysis summary.
        
        Args:
            df: OHLCV dataframe
            symbol: Stock symbol
        
        Returns:
            Volume summary dictionary
        """
        trend = self.analyze_volume_trend(df)
        unusual = self.detect_unusual_volume(df)
        profile = self.calculate_volume_profile(df)
        ad = self.detect_accumulation_distribution(df)
        
        # Get delivery if symbol provided
        delivery = None
        if symbol:
            delivery = self.analyze_delivery_percentage(symbol)
        
        # Calculate volume score
        score = 50
        
        # Volume confirmation
        pv_rel = trend.get("price_volume_relationship", "")
        if pv_rel == "CONFIRMED_UPTREND":
            score += 15
        elif pv_rel == "WEAK_UPTREND":
            score += 5
        elif pv_rel == "DISTRIBUTION":
            score -= 10
        elif pv_rel == "CONFIRMED_DOWNTREND":
            score -= 15
        
        # Accumulation/Distribution
        pattern = ad.get("pattern", "")
        if "ACCUMULATION" in pattern:
            score += 15
        elif "DISTRIBUTION" in pattern:
            score -= 15
        
        # Divergence
        if ad.get("divergence") == "BULLISH_DIVERGENCE":
            score += 10
        elif ad.get("divergence") == "BEARISH_DIVERGENCE":
            score -= 10
        
        # Delivery
        if delivery and delivery.get("assessment") == "HIGH_CONVICTION":
            score += 5
        
        score = max(0, min(100, score))
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "volume_score": score,
            "volume_trend": trend,
            "unusual_volume": unusual,
            "volume_profile": profile,
            "accumulation_distribution": ad,
            "delivery": delivery,
            "overall_assessment": self._get_overall_assessment(score, trend, ad),
        }
    
    def _get_overall_assessment(
        self,
        score: float,
        trend: Dict,
        ad: Dict,
    ) -> str:
        """Get overall volume assessment."""
        if score > 70:
            return "STRONG_ACCUMULATION"
        elif score > 60:
            return "ACCUMULATION"
        elif score < 30:
            return "STRONG_DISTRIBUTION"
        elif score < 40:
            return "DISTRIBUTION"
        else:
            return "NEUTRAL"


# Convenience function
def analyze_volume(df: pd.DataFrame, symbol: str = "") -> Dict:
    """Analyze volume for a dataframe."""
    analyzer = VolumeAnalyzer()
    return analyzer.generate_volume_summary(df, symbol)
