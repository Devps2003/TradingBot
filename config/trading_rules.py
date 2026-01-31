"""
Custom Trading Rules and Preferences.

These rules define your personal trading style and can be adjusted
based on experience and market conditions.
"""

from typing import Dict, List, Any

# ============================================================================
# TRADING STYLE CONFIGURATION
# ============================================================================

TRADING_STYLE = {
    "type": "swing",  # "intraday", "swing", "positional"
    "holding_period_days": {"min": 3, "max": 15, "target": 7},
    "preferred_sectors": [],  # Empty means all sectors
    "excluded_sectors": [],  # Sectors to avoid
    "min_market_cap_cr": 5000,  # Minimum market cap in crores
    "prefer_large_caps": True,
}

# ============================================================================
# ENTRY RULES
# ============================================================================

ENTRY_RULES = {
    # Technical requirements
    "require_trend_alignment": True,  # Stock trend must match market trend
    "min_volume_multiple": 1.0,  # Volume >= X times average
    "require_above_200dma": False,  # Must be above 200 DMA
    "require_above_50dma": True,  # Must be above 50 DMA
    
    # Momentum requirements
    "rsi_range": {"min": 30, "max": 70},  # Avoid extremes
    "avoid_overbought": True,
    "avoid_oversold": False,  # Sometimes oversold can be opportunity
    
    # Pattern requirements
    "require_pattern_confirmation": True,
    "min_pattern_confidence": 60,
    
    # Fundamental requirements
    "require_positive_earnings": True,
    "max_debt_to_equity": 2.0,
    "min_promoter_holding": 25,  # Percentage
    "max_promoter_pledge": 20,  # Percentage of holdings pledged
    
    # Sentiment requirements
    "avoid_negative_sentiment": True,
    "min_sentiment_score": -30,  # Minimum acceptable sentiment
    
    # Market context requirements
    "avoid_high_vix": True,  # VIX > 20
    "vix_threshold": 20,
    "require_fii_buying": False,  # FII net buyers in last 5 days
}

# ============================================================================
# EXIT RULES
# ============================================================================

EXIT_RULES = {
    # Profit booking
    "book_partial_at_target1": True,
    "partial_booking_percent": 50,  # Book 50% at first target
    "trail_remaining": True,  # Trail stop for remaining position
    
    # Stop loss
    "use_atr_stop": True,
    "atr_multiplier": 2.0,
    "max_stop_loss_percent": 8,
    "move_stop_to_breakeven": True,  # After target 1
    
    # Time-based exit
    "max_holding_days": 15,
    "exit_if_no_progress_days": 7,  # Exit if stock doesn't move
    
    # Technical exits
    "exit_on_trend_break": True,
    "exit_on_pattern_failure": True,
    
    # Event-based exits
    "exit_before_earnings": True,
    "days_before_earnings_exit": 3,
}

# ============================================================================
# POSITION SIZING RULES
# ============================================================================

POSITION_SIZING_RULES = {
    "method": "risk_based",  # "fixed", "risk_based", "kelly"
    
    # Fixed method settings
    "fixed_position_percent": 5,  # 5% of capital per trade
    
    # Risk-based method settings
    "risk_per_trade_percent": 2,  # Risk 2% of capital
    "max_position_percent": 15,  # Never exceed 15%
    "min_position_percent": 2,  # At least 2%
    
    # Kelly criterion settings (advanced)
    "use_kelly": False,
    "kelly_fraction": 0.5,  # Use half Kelly for safety
    
    # Confidence-based adjustment
    "adjust_for_confidence": True,
    "confidence_multipliers": {
        "high": 1.0,  # confidence >= 80
        "medium": 0.75,  # confidence >= 65
        "low": 0.5,  # confidence >= 50
    },
    
    # Volatility-based adjustment
    "adjust_for_volatility": True,
    "volatility_multipliers": {
        "low": 1.2,  # Low volatility = larger position
        "normal": 1.0,
        "high": 0.7,  # High volatility = smaller position
    },
}

# ============================================================================
# PORTFOLIO RULES
# ============================================================================

PORTFOLIO_RULES = {
    # Diversification
    "max_positions": 10,
    "max_per_sector": 3,
    "max_sector_allocation": 30,  # Percent
    
    # Correlation
    "avoid_high_correlation": True,
    "max_correlation": 0.7,
    
    # Portfolio heat (total risk)
    "max_portfolio_heat": 10,  # Maximum % of capital at risk
    "reduce_new_positions_at_heat": 8,  # Start being cautious
    
    # Cash management
    "min_cash_percent": 20,  # Always keep some cash
    "reserve_for_opportunities": 10,  # Keep extra for great setups
}

# ============================================================================
# SECTOR PREFERENCES
# ============================================================================

SECTOR_PREFERENCES = {
    "BANKING": {
        "weight": 1.0,  # Normal weight
        "max_allocation": 25,
        "preferred_for": ["trending_markets"],
    },
    "IT": {
        "weight": 1.0,
        "max_allocation": 25,
        "preferred_for": ["global_strength", "rupee_weakness"],
    },
    "PHARMA": {
        "weight": 0.9,
        "max_allocation": 20,
        "preferred_for": ["defensive"],
    },
    "FMCG": {
        "weight": 0.8,
        "max_allocation": 15,
        "preferred_for": ["defensive", "volatile_markets"],
    },
    "AUTO": {
        "weight": 1.0,
        "max_allocation": 20,
        "preferred_for": ["economic_recovery"],
    },
    "METALS": {
        "weight": 0.9,
        "max_allocation": 15,
        "preferred_for": ["commodity_rally", "global_recovery"],
    },
    "REALTY": {
        "weight": 0.7,
        "max_allocation": 10,
        "preferred_for": ["rate_cut_cycle"],
    },
}

# ============================================================================
# MARKET CONDITION RULES
# ============================================================================

MARKET_CONDITION_RULES = {
    "bullish_market": {
        "description": "Nifty above 20 DMA, FII buying, VIX < 15",
        "strategy": "aggressive",
        "position_size_multiplier": 1.2,
        "sectors": ["BANKING", "AUTO", "REALTY", "METALS"],
    },
    "neutral_market": {
        "description": "Nifty between 20 and 50 DMA, mixed signals",
        "strategy": "selective",
        "position_size_multiplier": 1.0,
        "sectors": ["IT", "PHARMA", "FMCG"],
    },
    "bearish_market": {
        "description": "Nifty below 50 DMA, FII selling, VIX > 20",
        "strategy": "defensive",
        "position_size_multiplier": 0.5,
        "sectors": ["PHARMA", "FMCG", "IT"],
    },
    "highly_volatile": {
        "description": "VIX > 25, large daily swings",
        "strategy": "avoid_new_positions",
        "position_size_multiplier": 0.3,
        "sectors": [],
    },
}

# ============================================================================
# BLACKLIST / WATCHLIST RULES
# ============================================================================

# Stocks to never trade (poor experience, manipulation, etc.)
BLACKLISTED_STOCKS: List[str] = [
    # Add stocks you want to avoid
    # "YESBANK",
    # "RCOM",
]

# Special handling stocks (extra caution)
CAUTION_STOCKS: Dict[str, str] = {
    # "ADANI*": "High volatility, use smaller positions",
}

# ============================================================================
# ALERT RULES
# ============================================================================

ALERT_RULES = {
    # Price alerts
    "alert_near_target": True,
    "target_proximity_percent": 2,  # Alert when 2% from target
    
    "alert_near_stop": True,
    "stop_proximity_percent": 2,  # Alert when 2% from stop
    
    # Volume alerts
    "alert_unusual_volume": True,
    "unusual_volume_threshold": 2.0,  # 2x average
    
    # News alerts
    "alert_negative_news": True,
    "alert_earnings_approaching": True,
    "earnings_alert_days": 5,
    
    # Technical alerts
    "alert_trend_break": True,
    "alert_pattern_completion": True,
    "alert_indicator_divergence": True,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_position_size_multiplier(confidence: float, volatility: str) -> float:
    """
    Calculate position size multiplier based on confidence and volatility.
    """
    # Confidence multiplier
    if confidence >= 80:
        conf_mult = POSITION_SIZING_RULES["confidence_multipliers"]["high"]
    elif confidence >= 65:
        conf_mult = POSITION_SIZING_RULES["confidence_multipliers"]["medium"]
    else:
        conf_mult = POSITION_SIZING_RULES["confidence_multipliers"]["low"]
    
    # Volatility multiplier
    vol_mult = POSITION_SIZING_RULES["volatility_multipliers"].get(volatility, 1.0)
    
    return conf_mult * vol_mult


def should_avoid_stock(symbol: str) -> tuple[bool, str]:
    """
    Check if a stock should be avoided based on blacklist rules.
    """
    if symbol in BLACKLISTED_STOCKS:
        return True, "Stock is blacklisted"
    
    for pattern, reason in CAUTION_STOCKS.items():
        if pattern.endswith("*"):
            if symbol.startswith(pattern[:-1]):
                return False, f"Caution: {reason}"
    
    return False, ""


def get_sector_rules(sector: str) -> Dict[str, Any]:
    """
    Get trading rules for a specific sector.
    """
    return SECTOR_PREFERENCES.get(sector, {
        "weight": 1.0,
        "max_allocation": 20,
        "preferred_for": [],
    })
