"""
Position Sizer Module.

Calculates optimal position sizes using:
- Fixed percentage
- Risk-based sizing
- Kelly criterion
- Volatility adjustment
"""

from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.trading_rules import POSITION_SIZING_RULES, get_position_size_multiplier
from config.settings import MAX_SINGLE_POSITION, DEFAULT_RISK_PER_TRADE
from src.utils.logger import LoggerMixin


class PositionSizer(LoggerMixin):
    """
    Calculates optimal position sizes for trades.
    """
    
    def __init__(self, capital: float = 500000):
        """
        Initialize position sizer.
        
        Args:
            capital: Total trading capital
        """
        self.capital = capital
        self.settings = POSITION_SIZING_RULES
    
    def calculate_fixed_size(
        self,
        entry_price: float,
        percentage: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate fixed percentage position size.
        
        Args:
            entry_price: Entry price per share
            percentage: Position percentage (default from settings)
        
        Returns:
            Position sizing result
        """
        pct = percentage or self.settings["fixed_position_percent"]
        position_value = self.capital * (pct / 100)
        shares = int(position_value / entry_price)
        actual_value = shares * entry_price
        
        return {
            "method": "fixed",
            "shares": shares,
            "position_value": round(actual_value, 2),
            "position_pct": round(actual_value / self.capital * 100, 2),
            "entry_price": entry_price,
        }
    
    def calculate_risk_based_size(
        self,
        entry_price: float,
        stop_loss: float,
        risk_percent: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate risk-based position size.
        
        Args:
            entry_price: Entry price per share
            stop_loss: Stop loss price
            risk_percent: Risk percentage of capital
        
        Returns:
            Position sizing result
        """
        risk_pct = risk_percent or self.settings["risk_per_trade_percent"]
        
        # Risk per share
        risk_per_share = entry_price - stop_loss
        
        if risk_per_share <= 0:
            return {
                "error": "Stop loss must be below entry price",
                "shares": 0,
            }
        
        # Maximum risk amount
        max_risk = self.capital * (risk_pct / 100)
        
        # Shares based on risk
        shares = int(max_risk / risk_per_share)
        
        # Position value
        position_value = shares * entry_price
        position_pct = position_value / self.capital * 100
        
        # Check max position constraint
        max_position = self.capital * MAX_SINGLE_POSITION
        if position_value > max_position:
            shares = int(max_position / entry_price)
            position_value = shares * entry_price
            position_pct = MAX_SINGLE_POSITION * 100
        
        actual_risk = shares * risk_per_share
        actual_risk_pct = actual_risk / self.capital * 100
        
        return {
            "method": "risk_based",
            "shares": shares,
            "position_value": round(position_value, 2),
            "position_pct": round(position_pct, 2),
            "risk_amount": round(actual_risk, 2),
            "risk_pct": round(actual_risk_pct, 2),
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "risk_per_share": round(risk_per_share, 2),
        }
    
    def calculate_kelly_size(
        self,
        entry_price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Calculate Kelly criterion position size.
        
        Args:
            entry_price: Entry price per share
            win_rate: Historical win rate (0-1)
            avg_win: Average win percentage
            avg_loss: Average loss percentage
            fraction: Kelly fraction (0.5 = half Kelly)
        
        Returns:
            Position sizing result
        """
        # Kelly formula: f* = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-p
        
        if avg_loss == 0:
            return {"error": "Average loss cannot be zero", "shares": 0}
        
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - p
        
        kelly_pct = (b * p - q) / b
        
        # Apply fraction (half Kelly is safer)
        adjusted_pct = kelly_pct * fraction
        
        # Cap at max position
        adjusted_pct = min(adjusted_pct, MAX_SINGLE_POSITION)
        adjusted_pct = max(adjusted_pct, 0)  # Don't go negative
        
        position_value = self.capital * adjusted_pct
        shares = int(position_value / entry_price)
        
        return {
            "method": "kelly",
            "shares": shares,
            "position_value": round(shares * entry_price, 2),
            "position_pct": round(adjusted_pct * 100, 2),
            "full_kelly_pct": round(kelly_pct * 100, 2),
            "fraction_used": fraction,
            "entry_price": entry_price,
        }
    
    def calculate_volatility_adjusted_size(
        self,
        entry_price: float,
        atr: float,
        base_risk_pct: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate volatility-adjusted position size.
        
        Args:
            entry_price: Entry price per share
            atr: Average True Range
            base_risk_pct: Base risk percentage
        
        Returns:
            Position sizing result
        """
        risk_pct = base_risk_pct or self.settings["risk_per_trade_percent"]
        
        # Volatility as percentage of price
        volatility_pct = atr / entry_price
        
        # Adjust position size inversely to volatility
        # Higher volatility = smaller position
        if volatility_pct > 0.03:
            multiplier = 0.7  # High volatility
        elif volatility_pct > 0.015:
            multiplier = 1.0  # Normal volatility
        else:
            multiplier = 1.2  # Low volatility
        
        # Use 2x ATR as stop
        stop_loss = entry_price - (2 * atr)
        
        # Calculate risk-based size with volatility adjustment
        base_size = self.calculate_risk_based_size(entry_price, stop_loss, risk_pct)
        
        adjusted_shares = int(base_size["shares"] * multiplier)
        position_value = adjusted_shares * entry_price
        
        return {
            "method": "volatility_adjusted",
            "shares": adjusted_shares,
            "position_value": round(position_value, 2),
            "position_pct": round(position_value / self.capital * 100, 2),
            "atr": atr,
            "volatility_pct": round(volatility_pct * 100, 2),
            "volatility_level": "HIGH" if volatility_pct > 0.03 else "NORMAL" if volatility_pct > 0.015 else "LOW",
            "multiplier": multiplier,
            "implied_stop": round(stop_loss, 2),
        }
    
    def calculate_optimal_size(
        self,
        entry_price: float,
        stop_loss: float,
        confidence: float = 60,
        volatility: str = "normal",
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size considering all factors.
        
        Args:
            entry_price: Entry price per share
            stop_loss: Stop loss price
            confidence: Trade confidence (0-100)
            volatility: Volatility level (low, normal, high)
        
        Returns:
            Optimal position sizing
        """
        # Start with risk-based sizing
        base = self.calculate_risk_based_size(entry_price, stop_loss)
        
        if "error" in base:
            return base
        
        # Apply confidence and volatility adjustments
        multiplier = get_position_size_multiplier(confidence, volatility)
        
        adjusted_shares = int(base["shares"] * multiplier)
        position_value = adjusted_shares * entry_price
        position_pct = position_value / self.capital * 100
        
        # Ensure within limits
        min_pct = self.settings["min_position_percent"]
        max_pct = self.settings["max_position_percent"]
        
        if position_pct > max_pct:
            adjusted_shares = int(self.capital * max_pct / 100 / entry_price)
            position_value = adjusted_shares * entry_price
            position_pct = max_pct
        elif position_pct < min_pct and adjusted_shares > 0:
            adjusted_shares = int(self.capital * min_pct / 100 / entry_price)
            position_value = adjusted_shares * entry_price
            position_pct = min_pct
        
        return {
            "method": "optimal",
            "shares": adjusted_shares,
            "position_value": round(position_value, 2),
            "position_pct": round(position_pct, 2),
            "base_shares": base["shares"],
            "confidence_adjustment": multiplier,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "risk_amount": round(adjusted_shares * (entry_price - stop_loss), 2),
        }
    
    def get_sizing_recommendation(
        self,
        entry_price: float,
        stop_loss: float,
        confidence: float = 60,
        atr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Get position sizing recommendation with explanation.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            confidence: Trade confidence
            atr: ATR for volatility assessment
        
        Returns:
            Sizing recommendation
        """
        # Determine volatility level
        if atr:
            vol_pct = atr / entry_price
            if vol_pct > 0.03:
                volatility = "high"
            elif vol_pct > 0.015:
                volatility = "normal"
            else:
                volatility = "low"
        else:
            volatility = "normal"
        
        # Calculate optimal size
        sizing = self.calculate_optimal_size(
            entry_price=entry_price,
            stop_loss=stop_loss,
            confidence=confidence,
            volatility=volatility,
        )
        
        if "error" in sizing:
            return sizing
        
        # Generate recommendation
        risk_pct = sizing["risk_amount"] / self.capital * 100
        
        recommendation = []
        
        if confidence >= 80:
            recommendation.append("High confidence setup - full position size")
        elif confidence >= 65:
            recommendation.append("Good confidence - moderate position size")
        else:
            recommendation.append("Lower confidence - smaller position recommended")
        
        if volatility == "high":
            recommendation.append("High volatility - size reduced for protection")
        elif volatility == "low":
            recommendation.append("Low volatility - slightly larger position OK")
        
        if risk_pct > 2:
            recommendation.append(f"Warning: Risk at {risk_pct:.1f}% is above 2% guideline")
        
        return {
            **sizing,
            "volatility_level": volatility,
            "recommendation": " | ".join(recommendation),
        }


# Convenience function
def get_position_size(
    entry: float,
    stop: float,
    capital: float = 500000,
    confidence: float = 60,
) -> Dict:
    """Get position size for a trade."""
    sizer = PositionSizer(capital)
    return sizer.get_sizing_recommendation(entry, stop, confidence)
