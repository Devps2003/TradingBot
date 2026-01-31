"""
General helper utilities for the Indian Market Trading Agent.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import pickle


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Dictionary with JSON data
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return {}
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to JSON file
        indent: Indentation level
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def cache_data(
    data: Any,
    cache_key: str,
    cache_dir: Union[str, Path],
    duration_minutes: int = 15,
) -> None:
    """
    Cache data with expiration.
    
    Args:
        data: Data to cache
        cache_key: Unique key for this cache
        cache_dir: Directory to store cache
        duration_minutes: Cache duration in minutes
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create safe filename from key
    safe_key = hashlib.md5(cache_key.encode()).hexdigest()
    cache_file = cache_dir / f"{safe_key}.cache"
    
    cache_data = {
        "data": data,
        "timestamp": datetime.now().isoformat(),
        "expires": (datetime.now() + timedelta(minutes=duration_minutes)).isoformat(),
        "key": cache_key,
    }
    
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f)


def get_cached_data(
    cache_key: str,
    cache_dir: Union[str, Path],
) -> Optional[Any]:
    """
    Get cached data if not expired.
    
    Args:
        cache_key: Unique key for this cache
        cache_dir: Directory to store cache
    
    Returns:
        Cached data or None if expired/not found
    """
    cache_dir = Path(cache_dir)
    safe_key = hashlib.md5(cache_key.encode()).hexdigest()
    cache_file = cache_dir / f"{safe_key}.cache"
    
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)
        
        expires = datetime.fromisoformat(cache_data["expires"])
        if datetime.now() > expires:
            cache_file.unlink()  # Delete expired cache
            return None
        
        return cache_data["data"]
    except Exception:
        return None


def format_number(value: float, decimal_places: int = 2) -> str:
    """
    Format a number with commas and decimal places (Indian format).
    
    Args:
        value: Number to format
        decimal_places: Number of decimal places
    
    Returns:
        Formatted string
    """
    if value is None:
        return "N/A"
    
    # Handle crores and lakhs
    if abs(value) >= 1e7:
        return f"₹{value/1e7:.{decimal_places}f} Cr"
    elif abs(value) >= 1e5:
        return f"₹{value/1e5:.{decimal_places}f} L"
    else:
        return f"₹{value:,.{decimal_places}f}"


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    Format a percentage value.
    
    Args:
        value: Percentage value
        decimal_places: Number of decimal places
    
    Returns:
        Formatted string with % sign
    """
    if value is None:
        return "N/A"
    
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimal_places}f}%"


def format_large_number(value: float) -> str:
    """
    Format large numbers in Indian notation (Lakhs, Crores).
    
    Args:
        value: Number to format
    
    Returns:
        Formatted string
    """
    if value is None:
        return "N/A"
    
    if abs(value) >= 1e12:
        return f"{value/1e12:.2f}L Cr"
    elif abs(value) >= 1e7:
        return f"{value/1e7:.2f} Cr"
    elif abs(value) >= 1e5:
        return f"{value/1e5:.2f} L"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.2f}K"
    else:
        return f"{value:.2f}"


def calculate_percentage_change(current: float, previous: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        current: Current value
        previous: Previous value
    
    Returns:
        Percentage change
    """
    if previous == 0:
        return 0.0
    return ((current - previous) / previous) * 100


def is_market_hours() -> bool:
    """
    Check if Indian market is currently open.
    
    Returns:
        True if market is open
    """
    from .indian_market_utils import is_market_open
    return is_market_open()


def get_signal_color(signal: str) -> str:
    """
    Get color for a trading signal.
    
    Args:
        signal: Signal type (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)
    
    Returns:
        Color name for rich console
    """
    colors = {
        "STRONG_BUY": "bold green",
        "BUY": "green",
        "HOLD": "yellow",
        "SELL": "red",
        "STRONG_SELL": "bold red",
    }
    return colors.get(signal.upper(), "white")


def get_trend_color(trend: str) -> str:
    """
    Get color for a trend indicator.
    
    Args:
        trend: Trend type (BULLISH, BEARISH, NEUTRAL)
    
    Returns:
        Color name for rich console
    """
    colors = {
        "BULLISH": "green",
        "BEARISH": "red",
        "NEUTRAL": "yellow",
    }
    return colors.get(trend.upper(), "white")


def truncate_string(s: str, max_length: int = 50) -> str:
    """
    Truncate a string to maximum length with ellipsis.
    
    Args:
        s: String to truncate
        max_length: Maximum length
    
    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[:max_length - 3] + "..."


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division that handles zero denominator.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division fails
    
    Returns:
        Division result or default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def round_to_tick(price: float, tick_size: float = 0.05) -> float:
    """
    Round price to nearest tick size.
    
    Args:
        price: Price to round
        tick_size: Minimum tick size (default 0.05 for NSE)
    
    Returns:
        Rounded price
    """
    return round(price / tick_size) * tick_size


def get_nse_symbol(symbol: str) -> str:
    """
    Convert symbol to NSE format.
    
    Args:
        symbol: Stock symbol
    
    Returns:
        NSE formatted symbol
    """
    symbol = symbol.upper().strip()
    if not symbol.endswith(".NS"):
        return symbol
    return symbol.replace(".NS", "")


def get_yahoo_symbol(symbol: str) -> str:
    """
    Convert symbol to Yahoo Finance format.
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Yahoo Finance formatted symbol
    """
    symbol = symbol.upper().strip()
    
    # Remove existing suffix if any
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol
    
    # Special mappings for symbols that differ on Yahoo Finance
    # Some NSE symbols don't work directly, try alternate suffixes
    special_symbols = {
        "M&M": "M%26M.NS",  # M&M needs URL encoding
        "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
        "L&TFH": "L%26TFH.NS",
    }
    
    if symbol in special_symbols:
        return special_symbols[symbol]
    
    # For most Indian stocks, add .NS suffix (NSE)
    # If .NS doesn't work, you can try .BO (BSE)
    return f"{symbol}.NS"


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for keys
    
    Returns:
        Flattened dictionary
    """
    items: List[tuple] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
