"""
Risk Calculator Module.

Calculates risk management parameters:
- Stop loss levels
- Target levels
- Position sizing
- Portfolio risk
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import (
    DEFAULT_RISK_PER_TRADE,
    MAX_PORTFOLIO_RISK,
    MAX_SINGLE_POSITION,
    DEFAULT_STOP_LOSS_PERCENT,
    ATR_STOP_MULTIPLIER,
    MAX_STOP_LOSS_PERCENT,
    MIN_RISK_REWARD_RATIO,
    DEFAULT_TARGET_MULTIPLIER,
)
from config.trading_rules import POSITION_SIZING_RULES
from src.utils.logger import LoggerMixin


class RiskCalculator(LoggerMixin):
    """
    Calculates risk management parameters for trades.
    """
    
    def __init__(self):
        """Initialize the risk calculator."""
        pass
    
    def calculate_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        df: Optional[pd.DataFrame] = None,
        method: str = "atr",
    ) -> Dict[str, Any]:
        """
        Calculate stop loss level.
        
        Args:
            symbol: Stock symbol
            entry_price: Entry price
            df: OHLCV dataframe (optional)
            method: Calculation method (atr, swing, percent)
        
        Returns:
            Stop loss calculation details
        """
        stops = {}
        
        # ATR-based stop loss
        if df is not None and len(df) >= 14:
            atr = self._calculate_atr(df)
            atr_stop = entry_price - (ATR_STOP_MULTIPLIER * atr)
            stops["atr"] = round(atr_stop, 2)
            stops["atr_value"] = round(atr, 2)
        
        # Swing-based stop loss
        if df is not None and len(df) >= 10:
            recent_low = df["low"].tail(10).min()
            swing_stop = recent_low * 0.99  # Slightly below swing low
            stops["swing"] = round(swing_stop, 2)
        
        # Percentage-based stop loss
        pct_stop = entry_price * (1 - DEFAULT_STOP_LOSS_PERCENT)
        stops["percent"] = round(pct_stop, 2)
        
        # Select best stop based on method
        if method == "atr" and "atr" in stops:
            recommended = stops["atr"]
        elif method == "swing" and "swing" in stops:
            recommended = stops["swing"]
        else:
            recommended = stops.get("percent", entry_price * 0.95)
        
        # Check max stop loss constraint
        max_stop = entry_price * (1 - MAX_STOP_LOSS_PERCENT)
        if recommended < max_stop:
            recommended = max_stop
            stops["note"] = "Stop adjusted to max allowed"
        
        risk_percent = (entry_price - recommended) / entry_price * 100
        
        return {
            "symbol": symbol,
            "entry_price": entry_price,
            "stop_loss": round(recommended, 2),
            "risk_percent": round(risk_percent, 2),
            "risk_amount": round(entry_price - recommended, 2),
            "method": method,
            "all_stops": stops,
        }
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range.
        
        Args:
            df: OHLCV dataframe
            period: ATR period
        
        Returns:
            ATR value
        """
        high = df["high"]
        low = df["low"]
        close = df["close"].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        
        return atr
    
    def calculate_targets(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Calculate target levels.
        
        Args:
            symbol: Stock symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            df: OHLCV dataframe (optional)
        
        Returns:
            Target calculation details
        """
        risk = entry_price - stop_loss
        
        # Risk-reward based targets
        target_1 = entry_price + (risk * MIN_RISK_REWARD_RATIO)
        target_2 = entry_price + (risk * DEFAULT_TARGET_MULTIPLIER)
        target_3 = entry_price + (risk * 3)
        
        targets = {
            "risk_reward": {
                "target_1": round(target_1, 2),
                "target_2": round(target_2, 2),
                "target_3": round(target_3, 2),
            }
        }
        
        # Resistance-based targets
        if df is not None and len(df) >= 50:
            resistances = self._find_resistance_levels(df, entry_price)
            if resistances:
                targets["resistance"] = [round(r, 2) for r in resistances[:3]]
        
        # Select recommended targets
        recommended = targets["risk_reward"]
        
        # If resistance is nearby and below RR target, adjust
        if "resistance" in targets and targets["resistance"]:
            first_resistance = targets["resistance"][0]
            if first_resistance < recommended["target_1"]:
                recommended["target_1"] = first_resistance
        
        return {
            "symbol": symbol,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target_1": recommended["target_1"],
            "target_2": recommended["target_2"],
            "target_3": recommended.get("target_3", recommended["target_2"] * 1.5),
            "risk_reward_1": round((recommended["target_1"] - entry_price) / risk, 2),
            "risk_reward_2": round((recommended["target_2"] - entry_price) / risk, 2),
            "all_targets": targets,
        }
    
    def _find_resistance_levels(
        self,
        df: pd.DataFrame,
        current_price: float,
        num_levels: int = 5,
    ) -> List[float]:
        """
        Find resistance levels above current price.
        
        Args:
            df: OHLCV dataframe
            current_price: Current price
            num_levels: Number of levels to find
        
        Returns:
            List of resistance levels
        """
        resistances = []
        
        # Recent highs
        for i in range(2, len(df) - 2):
            if (df["high"].iloc[i] > df["high"].iloc[i-1] and
                df["high"].iloc[i] > df["high"].iloc[i-2] and
                df["high"].iloc[i] > df["high"].iloc[i+1] and
                df["high"].iloc[i] > df["high"].iloc[i+2]):
                if df["high"].iloc[i] > current_price:
                    resistances.append(df["high"].iloc[i])
        
        # Sort and deduplicate
        resistances = sorted(set(resistances))
        
        return resistances[:num_levels]
    
    def calculate_position_size(
        self,
        symbol: str,
        capital: float,
        entry_price: float,
        stop_loss: float,
        risk_percent: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate position size.
        
        Args:
            symbol: Stock symbol
            capital: Available capital
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_percent: Risk percentage (default from settings)
            confidence: Trade confidence for adjustment
        
        Returns:
            Position sizing details
        """
        if risk_percent is None:
            risk_percent = POSITION_SIZING_RULES["risk_per_trade_percent"]
        
        # Risk per share
        risk_per_share = entry_price - stop_loss
        
        if risk_per_share <= 0:
            return {
                "error": "Invalid stop loss - must be below entry",
                "shares": 0,
                "position_value": 0,
            }
        
        # Calculate risk amount
        risk_amount = capital * (risk_percent / 100)
        
        # Calculate shares based on risk
        shares = int(risk_amount / risk_per_share)
        
        # Position value
        position_value = shares * entry_price
        position_pct = position_value / capital * 100
        
        # Check max position constraint
        max_position_value = capital * (MAX_SINGLE_POSITION)
        if position_value > max_position_value:
            position_value = max_position_value
            shares = int(position_value / entry_price)
            position_pct = MAX_SINGLE_POSITION * 100
        
        # Adjust for confidence
        if confidence is not None and POSITION_SIZING_RULES.get("adjust_for_confidence"):
            if confidence >= 80:
                multiplier = 1.0
            elif confidence >= 65:
                multiplier = 0.75
            else:
                multiplier = 0.5
            
            shares = int(shares * multiplier)
            position_value = shares * entry_price
            position_pct = position_value / capital * 100
        
        return {
            "symbol": symbol,
            "shares": shares,
            "entry_price": entry_price,
            "position_value": round(position_value, 2),
            "position_size_pct": round(position_pct, 2),
            "risk_amount": round(shares * risk_per_share, 2),
            "risk_pct_of_capital": round(shares * risk_per_share / capital * 100, 2),
            "confidence_adjustment": confidence is not None,
        }
    
    def calculate_portfolio_risk(
        self,
        portfolio: List[Dict],
        capital: float,
    ) -> Dict[str, Any]:
        """
        Calculate total portfolio risk.
        
        Args:
            portfolio: List of positions
            capital: Total capital
        
        Returns:
            Portfolio risk analysis
        """
        total_risk = 0
        position_risks = []
        
        for position in portfolio:
            entry = position.get("avg_price", 0)
            stop = position.get("stop_loss", entry * 0.95)
            quantity = position.get("quantity", 0)
            
            risk = (entry - stop) * quantity
            total_risk += risk
            
            position_risks.append({
                "symbol": position.get("symbol"),
                "risk_amount": round(risk, 2),
                "risk_pct": round(risk / capital * 100, 2),
            })
        
        portfolio_heat = total_risk / capital * 100
        
        return {
            "total_risk_amount": round(total_risk, 2),
            "portfolio_heat_pct": round(portfolio_heat, 2),
            "max_allowed_heat": MAX_PORTFOLIO_RISK * 100,
            "within_limits": portfolio_heat <= MAX_PORTFOLIO_RISK * 100,
            "position_risks": position_risks,
            "can_add_new_position": portfolio_heat < (MAX_PORTFOLIO_RISK * 100 - 2),
        }
    
    def suggest_risk_adjusted_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        confidence: float,
        capital: float,
        current_portfolio_heat: float = 0,
    ) -> Dict[str, Any]:
        """
        Suggest risk-adjusted position size.
        
        Args:
            symbol: Stock symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            confidence: Trade confidence
            capital: Available capital
            current_portfolio_heat: Current portfolio heat percentage
        
        Returns:
            Suggested position size
        """
        # Available risk capacity
        max_heat = MAX_PORTFOLIO_RISK * 100
        available_heat = max_heat - current_portfolio_heat
        
        if available_heat <= 0:
            return {
                "symbol": symbol,
                "can_trade": False,
                "reason": "Portfolio heat at maximum",
            }
        
        # Base position calculation
        base = self.calculate_position_size(
            symbol=symbol,
            capital=capital,
            entry_price=entry_price,
            stop_loss=stop_loss,
            confidence=confidence,
        )
        
        # Check if within available heat
        trade_heat = base.get("risk_pct_of_capital", 0)
        
        if trade_heat > available_heat:
            # Reduce size to fit available heat
            reduction_factor = available_heat / trade_heat
            adjusted_shares = int(base["shares"] * reduction_factor)
            adjusted_value = adjusted_shares * entry_price
            
            return {
                "symbol": symbol,
                "can_trade": True,
                "shares": adjusted_shares,
                "position_value": round(adjusted_value, 2),
                "position_size_pct": round(adjusted_value / capital * 100, 2),
                "reduced": True,
                "reduction_reason": "Limited by portfolio heat",
            }
        
        return {
            "symbol": symbol,
            "can_trade": True,
            "shares": base["shares"],
            "position_value": base["position_value"],
            "position_size_pct": base["position_size_pct"],
            "reduced": False,
        }
    
    def calculate_risk_parameters(
        self,
        symbol: str,
        entry_price: float,
        df: Optional[pd.DataFrame] = None,
        confidence: float = 60,
        capital: float = 100000,
    ) -> Dict[str, Any]:
        """
        Calculate all risk parameters for a trade.
        
        Args:
            symbol: Stock symbol
            entry_price: Entry price
            df: OHLCV dataframe
            confidence: Trade confidence
            capital: Trading capital
        
        Returns:
            Complete risk parameters
        """
        # Calculate stop loss
        stop_result = self.calculate_stop_loss(
            symbol=symbol,
            entry_price=entry_price,
            df=df,
            method="atr",
        )
        stop_loss = stop_result["stop_loss"]
        
        # Calculate targets
        target_result = self.calculate_targets(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            df=df,
        )
        
        # Calculate position size
        size_result = self.calculate_position_size(
            symbol=symbol,
            capital=capital,
            entry_price=entry_price,
            stop_loss=stop_loss,
            confidence=confidence,
        )
        
        return {
            "symbol": symbol,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target_1": target_result["target_1"],
            "target_2": target_result["target_2"],
            "risk_percent": stop_result["risk_percent"],
            "risk_reward": target_result["risk_reward_1"],
            "shares": size_result["shares"],
            "position_value": size_result["position_value"],
            "position_size_pct": size_result["position_size_pct"],
        }


# Convenience function
def calculate_risk(
    entry_price: float,
    stop_loss: float,
    capital: float = 100000,
) -> Dict:
    """Calculate risk for a trade."""
    calculator = RiskCalculator()
    return calculator.calculate_position_size(
        symbol="",
        capital=capital,
        entry_price=entry_price,
        stop_loss=stop_loss,
    )
