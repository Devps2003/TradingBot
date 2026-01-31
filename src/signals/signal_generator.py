"""
Signal Generator Module.

Combines all analysis to generate trading signals:
- Technical analysis
- Pattern recognition
- Fundamental analysis
- Sentiment analysis
- Volume analysis
- Market context
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import ANALYSIS_WEIGHTS, SIGNAL_THRESHOLDS, REGIME_WEIGHT_ADJUSTMENTS
from src.data_fetchers.price_fetcher import PriceFetcher
from src.analyzers.technical_analyzer import TechnicalAnalyzer
from src.analyzers.pattern_recognizer import PatternRecognizer
from src.analyzers.fundamental_analyzer import FundamentalAnalyzer
from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.analyzers.volume_analyzer import VolumeAnalyzer
from src.analyzers.market_context_analyzer import MarketContextAnalyzer
from src.signals.confidence_scorer import ConfidenceScorer
from src.signals.risk_calculator import RiskCalculator
from src.utils.logger import LoggerMixin


class SignalGenerator(LoggerMixin):
    """
    Generates trading signals by combining multiple analyses.
    """
    
    def __init__(self):
        """Initialize the signal generator."""
        self.price_fetcher = PriceFetcher()
        self.technical_analyzer = TechnicalAnalyzer()
        self.pattern_recognizer = PatternRecognizer()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()
        self.market_analyzer = MarketContextAnalyzer()
        self.confidence_scorer = ConfidenceScorer()
        self.risk_calculator = RiskCalculator()
    
    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """
        Generate comprehensive trading signal for a stock.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Trading signal dictionary
        """
        self.logger.info(f"Generating signal for {symbol}")
        
        # Fetch price data
        df = self.price_fetcher.get_historical_data(symbol)
        
        if df.empty:
            return {
                "symbol": symbol,
                "error": "Could not fetch price data",
                "signal": "NO_SIGNAL",
            }
        
        # Get all analyses
        technical = self.technical_analyzer.generate_technical_summary(df, symbol)
        patterns = self.pattern_recognizer.generate_pattern_summary(df, symbol)
        fundamental = self.fundamental_analyzer.generate_fundamental_summary(symbol)
        sentiment = self.sentiment_analyzer.generate_sentiment_summary(symbol)
        volume = self.volume_analyzer.generate_volume_summary(df, symbol)
        
        # Get market context
        market_context = self.market_analyzer.is_market_favorable_for_longs()
        
        # Get scores
        scores = {
            "technical": technical.get("technical_score", 50),
            "pattern": patterns.get("pattern_score", 50),
            "fundamental": fundamental.get("fundamental_score", 50),
            "sentiment": sentiment.get("sentiment_score", 50),
            "volume": volume.get("volume_score", 50),
            "market_context": 70 if market_context.get("favorable_for_longs") else 30,
        }
        
        # Adjust weights based on market regime
        weights = self._get_adjusted_weights(market_context)
        
        # Calculate combined score
        combined_score = sum(
            scores[key] * weights.get(key, 0)
            for key in scores
        )
        
        # Determine signal
        signal = self._determine_signal(combined_score)
        
        # Calculate confidence
        confidence = self.confidence_scorer.calculate_confidence({
            "scores": scores,
            "technical": technical,
            "patterns": patterns,
            "market_context": market_context,
        })
        
        # Calculate risk parameters
        current_price = df["close"].iloc[-1]
        risk_params = self.risk_calculator.calculate_risk_parameters(
            symbol=symbol,
            entry_price=current_price,
            df=df,
            confidence=confidence,
        )
        
        # Generate supporting and risk factors
        supporting = self._get_supporting_factors(scores, technical, patterns, sentiment)
        risks = self._get_risk_factors(scores, technical, fundamental, market_context)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            symbol, signal, confidence, scores, supporting, risks
        )
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "signal": signal,
            "confidence": round(confidence, 0),
            "combined_score": round(combined_score, 1),
            "entry_price": round(current_price, 2),
            "stop_loss": risk_params.get("stop_loss"),
            "target_1": risk_params.get("target_1"),
            "target_2": risk_params.get("target_2"),
            "risk_reward_ratio": risk_params.get("risk_reward"),
            "expected_holding_period": self._estimate_holding_period(signal, confidence),
            "position_size_pct": risk_params.get("position_size_pct"),
            "reasoning": reasoning,
            "supporting_factors": supporting,
            "risk_factors": risks,
            "scores": {
                "technical": round(scores["technical"], 1),
                "pattern": round(scores["pattern"], 1),
                "fundamental": round(scores["fundamental"], 1),
                "sentiment": round(scores["sentiment"], 1),
                "volume": round(scores["volume"], 1),
            },
            "market_alignment": market_context.get("favorable_for_longs", False),
            "technical_summary": {
                "trend": technical.get("trend"),
                "momentum": technical.get("momentum"),
                "key_levels": technical.get("key_levels"),
            },
            "pattern_summary": {
                "best_pattern": patterns.get("best_pattern"),
                "overall_bias": patterns.get("overall_bias"),
            },
        }
    
    def _get_adjusted_weights(self, market_context: Dict) -> Dict[str, float]:
        """
        Adjust weights based on market conditions.
        
        Args:
            market_context: Market context analysis
        
        Returns:
            Adjusted weights
        """
        # Start with default weights
        weights = ANALYSIS_WEIGHTS.copy()
        
        # Adjust based on market favorability
        assessment = market_context.get("assessment", "NEUTRAL")
        
        if assessment in ["HIGHLY_FAVORABLE", "FAVORABLE"]:
            # In favorable markets, technical matters more
            weights["technical"] = 0.40
            weights["fundamental"] = 0.15
        elif assessment == "UNFAVORABLE":
            # In unfavorable markets, fundamentals and sentiment matter more
            weights["technical"] = 0.25
            weights["fundamental"] = 0.25
            weights["sentiment"] = 0.20
        
        return weights
    
    def _determine_signal(self, score: float) -> str:
        """
        Determine signal based on combined score.
        
        Args:
            score: Combined analysis score
        
        Returns:
            Signal string
        """
        if score >= SIGNAL_THRESHOLDS["strong_buy"]:
            return "STRONG_BUY"
        elif score >= SIGNAL_THRESHOLDS["buy"]:
            return "BUY"
        elif score >= SIGNAL_THRESHOLDS["hold_lower"]:
            return "HOLD"
        elif score >= SIGNAL_THRESHOLDS["strong_sell"]:
            return "SELL"
        else:
            return "STRONG_SELL"
    
    def _get_supporting_factors(
        self,
        scores: Dict,
        technical: Dict,
        patterns: Dict,
        sentiment: Dict,
    ) -> List[str]:
        """Get supporting factors for the signal."""
        factors = []
        
        if scores["technical"] > 65:
            trend = technical.get("trend", "")
            factors.append(f"Technical: {trend} trend confirmed")
        
        best_pattern = patterns.get("best_pattern")
        if best_pattern and best_pattern.get("confidence", 0) > 60:
            factors.append(f"Pattern: {best_pattern['pattern_name']} detected")
        
        if scores["fundamental"] > 65:
            factors.append("Strong fundamentals")
        
        if scores["sentiment"] > 60:
            factors.append("Positive market sentiment")
        
        if scores["volume"] > 60:
            factors.append("Volume confirming price action")
        
        return factors[:5]
    
    def _get_risk_factors(
        self,
        scores: Dict,
        technical: Dict,
        fundamental: Dict,
        market_context: Dict,
    ) -> List[str]:
        """Get risk factors for the signal."""
        risks = []
        
        if scores["technical"] < 40:
            risks.append("Weak technical setup")
        
        if technical.get("momentum") == "OVERBOUGHT":
            risks.append("Overbought conditions")
        elif technical.get("momentum") == "OVERSOLD":
            risks.append("Oversold - may bounce but trend weak")
        
        red_flags = fundamental.get("red_flags", [])
        if red_flags:
            risks.append(f"{len(red_flags)} fundamental concerns")
        
        if not market_context.get("favorable_for_longs"):
            risks.append("Market not favorable for longs")
        
        if scores["sentiment"] < 40:
            risks.append("Negative sentiment in news")
        
        return risks[:5]
    
    def _estimate_holding_period(self, signal: str, confidence: float) -> str:
        """Estimate expected holding period."""
        if signal in ["STRONG_BUY", "STRONG_SELL"]:
            if confidence > 80:
                return "5-10 days"
            return "7-14 days"
        elif signal in ["BUY", "SELL"]:
            return "7-14 days"
        else:
            return "N/A - Hold/Monitor"
    
    def _generate_reasoning(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        scores: Dict,
        supporting: List[str],
        risks: List[str],
    ) -> str:
        """Generate human-readable reasoning for the signal."""
        parts = []
        
        # Signal introduction
        if signal == "STRONG_BUY":
            parts.append(f"{symbol} shows strong bullish setup")
        elif signal == "BUY":
            parts.append(f"{symbol} presents a buying opportunity")
        elif signal == "HOLD":
            parts.append(f"{symbol} is neutral - hold current position")
        elif signal == "SELL":
            parts.append(f"{symbol} shows weakness - consider reducing")
        else:
            parts.append(f"{symbol} shows strong bearish signals")
        
        # Confidence
        parts.append(f"with {confidence:.0f}% confidence.")
        
        # Key scores
        high_scores = [k for k, v in scores.items() if v > 65]
        if high_scores:
            parts.append(f"Strength in: {', '.join(high_scores)}.")
        
        # Main supporting factor
        if supporting:
            parts.append(supporting[0] + ".")
        
        # Main risk
        if risks:
            parts.append(f"Key risk: {risks[0]}.")
        
        return " ".join(parts)
    
    def generate_portfolio_signals(
        self,
        portfolio: List[Dict],
    ) -> List[Dict[str, Any]]:
        """
        Generate signals for all portfolio holdings.
        
        Args:
            portfolio: List of portfolio positions
        
        Returns:
            List of signals for each holding
        """
        signals = []
        
        for position in portfolio:
            symbol = position.get("symbol")
            if symbol:
                signal = self.generate_signal(symbol)
                signal["position"] = position
                signal["action_required"] = self._check_action_required(
                    signal, position
                )
                signals.append(signal)
        
        return signals
    
    def _check_action_required(
        self,
        signal: Dict,
        position: Dict,
    ) -> Optional[str]:
        """Check if action is required for a position."""
        current_signal = signal.get("signal", "HOLD")
        entry_price = position.get("avg_price", 0)
        current_price = signal.get("entry_price", 0)
        target = position.get("target")
        stop_loss = position.get("stop_loss")
        
        if current_signal in ["SELL", "STRONG_SELL"]:
            return "CONSIDER_EXIT"
        
        if target and current_price >= target * 0.98:
            return "NEAR_TARGET"
        
        if stop_loss and current_price <= stop_loss * 1.02:
            return "NEAR_STOP_LOSS"
        
        return None
    
    def generate_watchlist_signals(
        self,
        watchlist: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Generate signals for watchlist stocks.
        
        Args:
            watchlist: List of stock symbols
        
        Returns:
            List of signals
        """
        signals = []
        
        for symbol in watchlist:
            signal = self.generate_signal(symbol)
            signals.append(signal)
        
        # Sort by combined score
        signals.sort(
            key=lambda x: x.get("combined_score", 0),
            reverse=True
        )
        
        return signals
    
    def get_top_opportunities(
        self,
        n: int = 10,
        universe: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find top trading opportunities.
        
        Args:
            n: Number of opportunities to return
            universe: Stock universe to scan (default: Nifty 50)
        
        Returns:
            List of top opportunities
        """
        from config.settings import NIFTY_50
        
        if universe is None:
            universe = NIFTY_50[:30]  # Scan top 30 for speed
        
        # Check market context first
        market_context = self.market_analyzer.is_market_favorable_for_longs()
        
        if not market_context.get("favorable_for_longs"):
            self.logger.warning("Market not favorable - opportunities may be limited")
        
        signals = []
        
        for symbol in universe:
            try:
                signal = self.generate_signal(symbol)
                
                # Include BUY signals with reasonable confidence
                # Also include HOLD with very high scores (potential breakout candidates)
                if signal.get("signal") in ["STRONG_BUY", "BUY"]:
                    if signal.get("confidence", 0) >= 50:  # Lowered from 60
                        signals.append(signal)
                elif signal.get("signal") == "HOLD" and signal.get("combined_score", 0) >= 55:
                    # Include as "watchlist" candidates
                    signal["signal"] = "WATCHLIST"
                    signals.append(signal)
            except Exception as e:
                self.logger.debug(f"Error scanning {symbol}: {e}")  # Changed to debug
        
        # Sort by confidence, then combined score
        signals.sort(
            key=lambda x: (
                1 if x.get("signal") in ["STRONG_BUY", "BUY"] else 0,
                x.get("confidence", 0),
                x.get("combined_score", 0)
            ),
            reverse=True
        )
        
        return signals[:n]


# Convenience function
def get_signal(symbol: str) -> Dict:
    """Generate signal for a symbol."""
    generator = SignalGenerator()
    return generator.generate_signal(symbol)
