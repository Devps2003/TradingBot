"""
Market Context Analysis Module.

Analyzes broader market conditions:
- Market regime (trending, ranging, volatile)
- Market breadth (advance/decline)
- Sector rotation
- Risk levels
- Global cues impact
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_fetchers.price_fetcher import PriceFetcher
from src.data_fetchers.fii_dii_fetcher import FIIDIIFetcher
from src.data_fetchers.global_fetcher import GlobalFetcher
from config.settings import SECTORS, NIFTY_50
from src.utils.logger import LoggerMixin


class MarketContextAnalyzer(LoggerMixin):
    """
    Analyzes broader market context and conditions.
    """
    
    def __init__(self):
        """Initialize the market context analyzer."""
        self.price_fetcher = PriceFetcher()
        self.fii_dii_fetcher = FIIDIIFetcher()
        self.global_fetcher = GlobalFetcher()
    
    def get_market_regime(self) -> Dict[str, Any]:
        """
        Determine current market regime.
        
        Returns:
            Market regime analysis
        """
        # Get Nifty data
        nifty = self.price_fetcher.get_index_data("NIFTY 50")
        
        if nifty.empty or len(nifty) < 50:
            return {"regime": "UNKNOWN", "error": "Insufficient data"}
        
        # Calculate indicators
        nifty["sma_20"] = nifty["close"].rolling(20).mean()
        nifty["sma_50"] = nifty["close"].rolling(50).mean()
        nifty["atr"] = self._calculate_atr(nifty)
        nifty["atr_pct"] = nifty["atr"] / nifty["close"] * 100
        
        latest = nifty.iloc[-1]
        
        # Trend analysis
        above_20 = latest["close"] > latest["sma_20"]
        above_50 = latest["close"] > latest["sma_50"]
        ma_aligned_up = latest["sma_20"] > latest["sma_50"]
        ma_aligned_down = latest["sma_20"] < latest["sma_50"]
        
        # Price movement
        price_5d = (latest["close"] - nifty["close"].iloc[-5]) / nifty["close"].iloc[-5] * 100
        price_20d = (latest["close"] - nifty["close"].iloc[-20]) / nifty["close"].iloc[-20] * 100
        
        # Volatility
        avg_atr_pct = nifty["atr_pct"].tail(20).mean()
        
        # Determine regime
        if avg_atr_pct > 1.5:
            volatility = "HIGH"
        elif avg_atr_pct > 0.8:
            volatility = "NORMAL"
        else:
            volatility = "LOW"
        
        if above_20 and above_50 and ma_aligned_up:
            if abs(price_20d) > 5:
                regime = "STRONG_UPTREND"
            else:
                regime = "UPTREND"
        elif not above_20 and not above_50 and ma_aligned_down:
            if abs(price_20d) > 5:
                regime = "STRONG_DOWNTREND"
            else:
                regime = "DOWNTREND"
        elif abs(price_20d) < 2:
            regime = "RANGING"
        else:
            regime = "TRANSITIONING"
        
        return {
            "regime": regime,
            "volatility": volatility,
            "nifty_current": round(latest["close"], 2),
            "above_20dma": above_20,
            "above_50dma": above_50,
            "ma_aligned": "UP" if ma_aligned_up else "DOWN" if ma_aligned_down else "MIXED",
            "price_change_5d": round(price_5d, 2),
            "price_change_20d": round(price_20d, 2),
            "avg_volatility_pct": round(avg_atr_pct, 2),
            "interpretation": self._interpret_regime(regime, volatility),
        }
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        high = df["high"]
        low = df["low"]
        close = df["close"].shift(1)
        
        tr = pd.concat([
            high - low,
            abs(high - close),
            abs(low - close)
        ], axis=1).max(axis=1)
        
        return tr.rolling(period).mean()
    
    def _interpret_regime(self, regime: str, volatility: str) -> str:
        """Interpret market regime."""
        interpretations = {
            "STRONG_UPTREND": "Strong bullish momentum - favor long positions",
            "UPTREND": "Bullish trend - look for buying opportunities on dips",
            "STRONG_DOWNTREND": "Strong bearish momentum - avoid new longs",
            "DOWNTREND": "Bearish trend - be cautious with longs",
            "RANGING": "Sideways market - focus on range trading",
            "TRANSITIONING": "Market in transition - wait for clarity",
        }
        
        base = interpretations.get(regime, "Unknown regime")
        
        if volatility == "HIGH":
            base += ". High volatility - use smaller positions"
        elif volatility == "LOW":
            base += ". Low volatility - good for trend following"
        
        return base
    
    def analyze_market_breadth(self) -> Dict[str, Any]:
        """
        Analyze market breadth (advance/decline).
        
        Returns:
            Market breadth analysis
        """
        advances = 0
        declines = 0
        unchanged = 0
        
        # Sample Nifty 50 stocks
        for symbol in NIFTY_50[:30]:  # Sample 30 stocks
            try:
                price = self.price_fetcher.get_live_price(symbol)
                change = price.get("change_percent", 0)
                
                if change and change > 0.2:
                    advances += 1
                elif change and change < -0.2:
                    declines += 1
                else:
                    unchanged += 1
            except Exception:
                continue
        
        total = advances + declines + unchanged
        
        if total == 0:
            return {"error": "Could not calculate breadth"}
        
        ad_ratio = advances / max(declines, 1)
        advance_pct = advances / total * 100
        
        if ad_ratio > 2:
            breadth = "VERY_POSITIVE"
        elif ad_ratio > 1.2:
            breadth = "POSITIVE"
        elif ad_ratio > 0.8:
            breadth = "NEUTRAL"
        elif ad_ratio > 0.5:
            breadth = "NEGATIVE"
        else:
            breadth = "VERY_NEGATIVE"
        
        return {
            "advances": advances,
            "declines": declines,
            "unchanged": unchanged,
            "total": total,
            "ad_ratio": round(ad_ratio, 2),
            "advance_pct": round(advance_pct, 1),
            "breadth": breadth,
            "interpretation": self._interpret_breadth(breadth, ad_ratio),
        }
    
    def _interpret_breadth(self, breadth: str, ratio: float) -> str:
        """Interpret market breadth."""
        if breadth == "VERY_POSITIVE":
            return f"Broad-based rally - {ratio:.1f}x more advances than declines"
        elif breadth == "POSITIVE":
            return "Healthy breadth supporting the rally"
        elif breadth == "NEUTRAL":
            return "Mixed market - selective stock picking needed"
        elif breadth == "NEGATIVE":
            return "Weak breadth - rallies may not sustain"
        else:
            return "Very weak breadth - caution warranted"
    
    def get_sector_performance(self) -> Dict[str, Any]:
        """
        Get sector-wise performance.
        
        Returns:
            Sector performance analysis
        """
        sector_perf = {}
        
        for sector, stocks in SECTORS.items():
            if not stocks:
                continue
            
            changes = []
            for symbol in stocks[:5]:  # Top 5 from each sector
                try:
                    price = self.price_fetcher.get_live_price(symbol)
                    change = price.get("change_percent", 0)
                    if change:
                        changes.append(change)
                except Exception:
                    continue
            
            if changes:
                avg_change = sum(changes) / len(changes)
                sector_perf[sector] = round(avg_change, 2)
        
        # Sort by performance
        sorted_sectors = sorted(sector_perf.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "sector_performance": dict(sorted_sectors),
            "top_sector": sorted_sectors[0] if sorted_sectors else None,
            "bottom_sector": sorted_sectors[-1] if sorted_sectors else None,
            "leaders": [s for s, p in sorted_sectors if p > 1],
            "laggards": [s for s, p in sorted_sectors if p < -1],
        }
    
    def analyze_sector_rotation(self) -> Dict[str, Any]:
        """
        Analyze sector rotation patterns.
        
        Returns:
            Sector rotation analysis
        """
        # This would typically compare sector performance over different periods
        # Simplified version using current data
        
        current = self.get_sector_performance()
        
        if "error" in current:
            return current
        
        # Determine market phase based on leading sectors
        leaders = current.get("leaders", [])
        
        # Market cycle mapping
        if any(s in leaders for s in ["BANKING", "FINANCE", "AUTO"]):
            phase = "EARLY_CYCLE"
            recommendation = "Economy recovering - favor cyclicals"
        elif any(s in leaders for s in ["IT", "METALS", "ENERGY"]):
            phase = "MID_CYCLE"
            recommendation = "Growth phase - favor momentum stocks"
        elif any(s in leaders for s in ["PHARMA", "FMCG"]):
            phase = "LATE_CYCLE"
            recommendation = "Defensive sectors leading - be cautious"
        else:
            phase = "UNCERTAIN"
            recommendation = "No clear rotation - stock selection key"
        
        return {
            "current_leaders": leaders,
            "current_laggards": current.get("laggards", []),
            "market_phase": phase,
            "recommendation": recommendation,
            "sector_details": current.get("sector_performance", {}),
        }
    
    def get_market_risk_level(self) -> Dict[str, Any]:
        """
        Assess overall market risk level.
        
        Returns:
            Market risk assessment
        """
        risk_score = 50  # Start neutral
        risk_factors = []
        
        # 1. VIX check
        vix_data = self.global_fetcher.get_vix_data()
        india_vix = vix_data.get("india_vix", {}).get("price", 15)
        
        if india_vix > 25:
            risk_score += 20
            risk_factors.append(f"India VIX elevated at {india_vix:.1f}")
        elif india_vix > 20:
            risk_score += 10
            risk_factors.append(f"India VIX moderately high at {india_vix:.1f}")
        elif india_vix < 12:
            risk_score -= 10
        
        # 2. FII activity
        fii_dii = self.fii_dii_fetcher.get_daily_fii_dii()
        fii_net = fii_dii.get("fii", {}).get("net_value", 0) or 0
        
        if fii_net < -2000:
            risk_score += 15
            risk_factors.append(f"Heavy FII selling: ₹{abs(fii_net):.0f} Cr")
        elif fii_net < -1000:
            risk_score += 10
            risk_factors.append(f"FII outflow: ₹{abs(fii_net):.0f} Cr")
        elif fii_net > 1000:
            risk_score -= 10
        
        # 3. Market regime
        regime = self.get_market_regime()
        if regime.get("regime") in ["STRONG_DOWNTREND", "DOWNTREND"]:
            risk_score += 15
            risk_factors.append("Market in downtrend")
        elif regime.get("volatility") == "HIGH":
            risk_score += 10
            risk_factors.append("High market volatility")
        
        # 4. Global cues
        global_cues = self.global_fetcher.get_market_cues_for_india()
        us_change = global_cues.get("us_close", {}).get("sp500", {}).get("change_pct", 0) or 0
        
        if us_change < -1:
            risk_score += 10
            risk_factors.append(f"Negative US market cue: {us_change:.1f}%")
        
        # Determine risk level
        if risk_score > 70:
            level = "HIGH"
            action = "Reduce exposure, tighten stops"
        elif risk_score > 55:
            level = "ELEVATED"
            action = "Be cautious, smaller position sizes"
        elif risk_score > 40:
            level = "NORMAL"
            action = "Normal trading conditions"
        else:
            level = "LOW"
            action = "Favorable conditions for longs"
        
        return {
            "risk_score": risk_score,
            "risk_level": level,
            "risk_factors": risk_factors,
            "recommended_action": action,
            "india_vix": india_vix,
            "fii_net": fii_net,
        }
    
    def is_market_favorable_for_longs(self) -> Dict[str, Any]:
        """
        Determine if market is favorable for long positions.
        
        Returns:
            Market favorability assessment
        """
        regime = self.get_market_regime()
        risk = self.get_market_risk_level()
        breadth = self.analyze_market_breadth()
        
        favorable_points = 0
        reasons = []
        
        # Regime check
        if regime.get("regime") in ["UPTREND", "STRONG_UPTREND"]:
            favorable_points += 2
            reasons.append("Market in uptrend")
        elif regime.get("regime") in ["DOWNTREND", "STRONG_DOWNTREND"]:
            favorable_points -= 2
            reasons.append("Market in downtrend - headwind")
        
        # MA check
        if regime.get("above_20dma") and regime.get("above_50dma"):
            favorable_points += 1
            reasons.append("Above key moving averages")
        elif not regime.get("above_20dma") and not regime.get("above_50dma"):
            favorable_points -= 1
            reasons.append("Below key moving averages")
        
        # Risk check
        if risk.get("risk_level") == "LOW":
            favorable_points += 1
            reasons.append("Low risk environment")
        elif risk.get("risk_level") == "HIGH":
            favorable_points -= 2
            reasons.append("High risk environment")
        
        # Breadth check
        if breadth.get("breadth") in ["POSITIVE", "VERY_POSITIVE"]:
            favorable_points += 1
            reasons.append("Positive market breadth")
        elif breadth.get("breadth") in ["NEGATIVE", "VERY_NEGATIVE"]:
            favorable_points -= 1
            reasons.append("Weak market breadth")
        
        # Determine favorability
        if favorable_points >= 3:
            favorable = True
            assessment = "HIGHLY_FAVORABLE"
        elif favorable_points >= 1:
            favorable = True
            assessment = "FAVORABLE"
        elif favorable_points >= -1:
            favorable = False
            assessment = "NEUTRAL"
        else:
            favorable = False
            assessment = "UNFAVORABLE"
        
        return {
            "favorable_for_longs": favorable,
            "assessment": assessment,
            "score": favorable_points,
            "reasons": reasons,
            "recommendation": self._get_trading_recommendation(assessment),
        }
    
    def _get_trading_recommendation(self, assessment: str) -> str:
        """Get trading recommendation based on assessment."""
        recommendations = {
            "HIGHLY_FAVORABLE": "Excellent conditions for swing longs. Use full position sizes.",
            "FAVORABLE": "Good conditions for longs. Normal position sizes.",
            "NEUTRAL": "Mixed conditions. Be selective, use smaller sizes.",
            "UNFAVORABLE": "Challenging conditions. Avoid new longs or use minimal sizes.",
        }
        return recommendations.get(assessment, "Exercise caution")
    
    def generate_market_context_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive market context summary.
        
        Returns:
            Market context summary
        """
        regime = self.get_market_regime()
        breadth = self.analyze_market_breadth()
        sectors = self.get_sector_performance()
        rotation = self.analyze_sector_rotation()
        risk = self.get_market_risk_level()
        favorable = self.is_market_favorable_for_longs()
        
        # FII/DII
        fii_dii = self.fii_dii_fetcher.get_market_sentiment_indicator()
        
        # Global
        global_cues = self.global_fetcher.get_market_cues_for_india()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "regime": regime,
            "breadth": breadth,
            "sector_performance": sectors,
            "sector_rotation": rotation,
            "risk_assessment": risk,
            "favorability": favorable,
            "fii_dii": fii_dii,
            "global_cues": {
                "expected_gap": global_cues.get("expected_gap"),
                "us_sentiment": global_cues.get("overall_sentiment"),
                "highlights": global_cues.get("key_highlights", []),
            },
            "summary": self._generate_text_summary(regime, risk, favorable),
        }
    
    def _generate_text_summary(
        self,
        regime: Dict,
        risk: Dict,
        favorable: Dict,
    ) -> str:
        """Generate text summary of market context."""
        parts = []
        
        parts.append(f"Regime: {regime.get('regime', 'UNKNOWN')}")
        parts.append(f"Risk: {risk.get('risk_level', 'NORMAL')}")
        parts.append(f"For longs: {favorable.get('assessment', 'NEUTRAL')}")
        
        return " | ".join(parts)


# Convenience function
def get_market_context() -> Dict:
    """Get current market context."""
    analyzer = MarketContextAnalyzer()
    return analyzer.generate_market_context_summary()
