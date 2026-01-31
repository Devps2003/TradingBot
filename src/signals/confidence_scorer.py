"""
Confidence Scoring Module.

Calculates confidence scores for trading signals based on:
- Multiple indicator confirmation
- Pattern reliability
- Market alignment
- Historical accuracy
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import LoggerMixin


class ConfidenceScorer(LoggerMixin):
    """
    Calculates confidence scores for trading signals.
    """
    
    def __init__(self):
        """Initialize the confidence scorer."""
        pass
    
    def calculate_confidence(
        self,
        analysis_data: Dict[str, Any],
    ) -> float:
        """
        Calculate overall confidence score.
        
        Args:
            analysis_data: Dictionary containing all analysis results
        
        Returns:
            Confidence score (0-100)
        """
        base_confidence = 50
        adjustments = []
        
        # Score consistency bonus
        scores = analysis_data.get("scores", {})
        consistency = self._check_score_consistency(scores)
        adjustments.append(("score_consistency", consistency))
        
        # Technical confirmation
        technical = analysis_data.get("technical", {})
        tech_conf = self._assess_technical_confidence(technical)
        adjustments.append(("technical", tech_conf))
        
        # Pattern confirmation
        patterns = analysis_data.get("patterns", {})
        pattern_conf = self._assess_pattern_confidence(patterns)
        adjustments.append(("pattern", pattern_conf))
        
        # Market alignment
        market = analysis_data.get("market_context", {})
        market_conf = self._assess_market_alignment(market)
        adjustments.append(("market", market_conf))
        
        # Calculate final confidence
        total_adjustment = sum(adj for _, adj in adjustments)
        confidence = base_confidence + total_adjustment
        
        # Cap between 0 and 100
        return max(0, min(100, confidence))
    
    def _check_score_consistency(self, scores: Dict[str, float]) -> float:
        """
        Check if scores are consistent (all pointing same direction).
        
        Args:
            scores: Dictionary of analysis scores
        
        Returns:
            Adjustment value
        """
        if not scores:
            return 0
        
        values = list(scores.values())
        
        # Count bullish vs bearish scores
        bullish = sum(1 for v in values if v > 60)
        bearish = sum(1 for v in values if v < 40)
        neutral = len(values) - bullish - bearish
        
        total = len(values)
        
        # High consistency = high confidence adjustment
        if bullish >= total * 0.8:
            return 15  # Very consistent bullish
        elif bullish >= total * 0.6:
            return 10  # Mostly bullish
        elif bearish >= total * 0.8:
            return 10  # Very consistent bearish (also gives confidence)
        elif bearish >= total * 0.6:
            return 5  # Mostly bearish
        elif neutral >= total * 0.6:
            return -5  # Mostly neutral = less conviction
        else:
            return -10  # Mixed signals = lower confidence
    
    def _assess_technical_confidence(self, technical: Dict) -> float:
        """
        Assess confidence from technical analysis.
        
        Args:
            technical: Technical analysis results
        
        Returns:
            Adjustment value
        """
        adjustment = 0
        
        # Trend strength
        strength = technical.get("trend_strength", 50)
        if strength > 70:
            adjustment += 10
        elif strength > 50:
            adjustment += 5
        elif strength < 30:
            adjustment -= 5
        
        # Volume confirmation
        signals = technical.get("indicator_signals", {})
        if signals.get("volume") == "HIGH":
            adjustment += 5
        elif signals.get("volume") == "LOW":
            adjustment -= 5
        
        # Multiple indicator confirmation
        bullish_signals = sum(
            1 for v in signals.values()
            if v in ["BUY", "BULLISH", "ABOVE"]
        )
        bearish_signals = sum(
            1 for v in signals.values()
            if v in ["SELL", "BEARISH", "BELOW"]
        )
        
        if bullish_signals >= 5:
            adjustment += 5
        elif bearish_signals >= 5:
            adjustment += 3  # Consistent bearish is also confident
        
        return adjustment
    
    def _assess_pattern_confidence(self, patterns: Dict) -> float:
        """
        Assess confidence from pattern recognition.
        
        Args:
            patterns: Pattern analysis results
        
        Returns:
            Adjustment value
        """
        adjustment = 0
        
        best_pattern = patterns.get("best_pattern")
        
        if best_pattern:
            pattern_conf = best_pattern.get("confidence", 50)
            historical_acc = best_pattern.get("historical_accuracy", 0.5)
            
            # High confidence pattern
            if pattern_conf > 70 and historical_acc > 0.65:
                adjustment += 10
            elif pattern_conf > 60:
                adjustment += 5
        
        # Multiple patterns pointing same direction
        bias = patterns.get("overall_bias", "NEUTRAL")
        total = patterns.get("total_patterns", 0)
        
        if total >= 3 and bias != "NEUTRAL":
            adjustment += 5
        
        return adjustment
    
    def _assess_market_alignment(self, market: Dict) -> float:
        """
        Assess confidence from market alignment.
        
        Args:
            market: Market context analysis
        
        Returns:
            Adjustment value
        """
        if market.get("favorable_for_longs"):
            assessment = market.get("assessment", "NEUTRAL")
            if assessment == "HIGHLY_FAVORABLE":
                return 10
            elif assessment == "FAVORABLE":
                return 5
        else:
            assessment = market.get("assessment", "NEUTRAL")
            if assessment == "UNFAVORABLE":
                return -10
        
        return 0
    
    def get_confidence_breakdown(
        self,
        analysis_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Get detailed breakdown of confidence calculation.
        
        Args:
            analysis_data: Analysis results
        
        Returns:
            Breakdown dictionary
        """
        scores = analysis_data.get("scores", {})
        technical = analysis_data.get("technical", {})
        patterns = analysis_data.get("patterns", {})
        market = analysis_data.get("market_context", {})
        
        breakdown = {
            "base": 50,
            "score_consistency": self._check_score_consistency(scores),
            "technical_confirmation": self._assess_technical_confidence(technical),
            "pattern_confirmation": self._assess_pattern_confidence(patterns),
            "market_alignment": self._assess_market_alignment(market),
        }
        
        breakdown["total"] = sum(breakdown.values())
        breakdown["final"] = max(0, min(100, breakdown["total"]))
        
        return breakdown
    
    def adjust_for_market_conditions(
        self,
        base_confidence: float,
        conditions: Dict[str, Any],
    ) -> float:
        """
        Adjust confidence for special market conditions.
        
        Args:
            base_confidence: Base confidence score
            conditions: Market conditions
        
        Returns:
            Adjusted confidence
        """
        confidence = base_confidence
        
        # VIX adjustment
        vix = conditions.get("india_vix", 15)
        if vix > 25:
            confidence *= 0.85  # Reduce confidence in high VIX
        elif vix > 20:
            confidence *= 0.92
        
        # Earnings proximity
        days_to_earnings = conditions.get("days_to_earnings")
        if days_to_earnings is not None and days_to_earnings < 7:
            confidence *= 0.9  # Reduce before earnings
        
        # News event
        if conditions.get("major_news_event"):
            confidence *= 0.85
        
        return max(0, min(100, confidence))
    
    def get_confidence_level(self, confidence: float) -> str:
        """
        Get confidence level label.
        
        Args:
            confidence: Confidence score
        
        Returns:
            Confidence level string
        """
        if confidence >= 80:
            return "HIGH"
        elif confidence >= 65:
            return "MEDIUM_HIGH"
        elif confidence >= 50:
            return "MEDIUM"
        elif confidence >= 35:
            return "LOW"
        else:
            return "VERY_LOW"


# Convenience function
def score_confidence(analysis_data: Dict) -> float:
    """Calculate confidence for analysis data."""
    scorer = ConfidenceScorer()
    return scorer.calculate_confidence(analysis_data)
