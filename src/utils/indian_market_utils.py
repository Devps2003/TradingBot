"""
Indian market specific utilities.

Handles NSE/BSE specific requirements like:
- Market hours
- Trading holidays
- Symbol formatting
- Lot sizes
"""

from datetime import datetime, date, time, timedelta
from typing import List, Optional, Tuple
import pytz

# Indian timezone
IST = pytz.timezone("Asia/Kolkata")

# Market timing constants
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
PRE_MARKET_START = time(9, 0)
PRE_MARKET_END = time(9, 8)
POST_MARKET_START = time(15, 40)
POST_MARKET_END = time(16, 0)

# NSE Holidays 2025-2026 (update annually)
NSE_HOLIDAYS_2025 = [
    date(2025, 1, 26),   # Republic Day
    date(2025, 2, 26),   # Maha Shivaratri
    date(2025, 3, 14),   # Holi
    date(2025, 3, 31),   # Id-Ul-Fitr
    date(2025, 4, 10),   # Shri Mahavir Jayanti
    date(2025, 4, 14),   # Dr. Ambedkar Jayanti
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 5, 12),   # Buddha Purnima
    date(2025, 6, 7),    # Bakri Id
    date(2025, 7, 6),    # Moharram
    date(2025, 8, 15),   # Independence Day
    date(2025, 8, 16),   # Janmashtami
    date(2025, 9, 5),    # Milad-un-Nabi
    date(2025, 10, 2),   # Mahatma Gandhi Jayanti
    date(2025, 10, 21),  # Dussehra
    date(2025, 10, 22),  # Dussehra Holiday
    date(2025, 11, 1),   # Diwali Laxmi Pujan
    date(2025, 11, 5),   # Diwali Balipratipada
    date(2025, 11, 5),   # Guru Nanak Jayanti
    date(2025, 12, 25),  # Christmas
]

NSE_HOLIDAYS_2026 = [
    date(2026, 1, 26),   # Republic Day
    date(2026, 2, 17),   # Maha Shivaratri
    date(2026, 3, 3),    # Holi
    date(2026, 3, 20),   # Id-Ul-Fitr
    date(2026, 3, 30),   # Shri Mahavir Jayanti
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 31),   # Buddha Purnima
    date(2026, 5, 27),   # Bakri Id
    date(2026, 6, 26),   # Moharram
    date(2026, 8, 15),   # Independence Day
    date(2026, 8, 25),   # Milad-un-Nabi
    date(2026, 9, 5),    # Janmashtami
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 9),   # Dussehra
    date(2026, 10, 20),  # Diwali Laxmi Pujan
    date(2026, 10, 26),  # Guru Nanak Jayanti
    date(2026, 12, 25),  # Christmas
]

ALL_HOLIDAYS = set(NSE_HOLIDAYS_2025 + NSE_HOLIDAYS_2026)


def get_current_ist_time() -> datetime:
    """
    Get current time in IST.
    
    Returns:
        Current datetime in IST
    """
    return datetime.now(IST)


def is_trading_day(check_date: Optional[date] = None) -> bool:
    """
    Check if a given date is a trading day.
    
    Args:
        check_date: Date to check (default: today)
    
    Returns:
        True if trading day
    """
    if check_date is None:
        check_date = get_current_ist_time().date()
    
    # Check if weekend
    if check_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Check if holiday
    if check_date in ALL_HOLIDAYS:
        return False
    
    return True


def is_market_open() -> bool:
    """
    Check if the market is currently open.
    
    Returns:
        True if market is open
    """
    now = get_current_ist_time()
    current_time = now.time()
    current_date = now.date()
    
    # Check if trading day
    if not is_trading_day(current_date):
        return False
    
    # Check if within market hours
    return MARKET_OPEN <= current_time <= MARKET_CLOSE


def is_pre_market() -> bool:
    """
    Check if we're in pre-market session.
    
    Returns:
        True if in pre-market
    """
    now = get_current_ist_time()
    current_time = now.time()
    current_date = now.date()
    
    if not is_trading_day(current_date):
        return False
    
    return PRE_MARKET_START <= current_time <= PRE_MARKET_END


def is_post_market() -> bool:
    """
    Check if we're in post-market session.
    
    Returns:
        True if in post-market
    """
    now = get_current_ist_time()
    current_time = now.time()
    current_date = now.date()
    
    if not is_trading_day(current_date):
        return False
    
    return POST_MARKET_START <= current_time <= POST_MARKET_END


def get_market_status() -> str:
    """
    Get current market status.
    
    Returns:
        Status string: "OPEN", "CLOSED", "PRE_MARKET", "POST_MARKET"
    """
    if is_market_open():
        return "OPEN"
    elif is_pre_market():
        return "PRE_MARKET"
    elif is_post_market():
        return "POST_MARKET"
    else:
        return "CLOSED"


def get_next_trading_day(from_date: Optional[date] = None) -> date:
    """
    Get the next trading day.
    
    Args:
        from_date: Starting date (default: today)
    
    Returns:
        Next trading day
    """
    if from_date is None:
        from_date = get_current_ist_time().date()
    
    next_day = from_date + timedelta(days=1)
    
    while not is_trading_day(next_day):
        next_day += timedelta(days=1)
    
    return next_day


def get_previous_trading_day(from_date: Optional[date] = None) -> date:
    """
    Get the previous trading day.
    
    Args:
        from_date: Starting date (default: today)
    
    Returns:
        Previous trading day
    """
    if from_date is None:
        from_date = get_current_ist_time().date()
    
    prev_day = from_date - timedelta(days=1)
    
    while not is_trading_day(prev_day):
        prev_day -= timedelta(days=1)
    
    return prev_day


def get_trading_days_between(start_date: date, end_date: date) -> List[date]:
    """
    Get all trading days between two dates.
    
    Args:
        start_date: Start date
        end_date: End date
    
    Returns:
        List of trading days
    """
    trading_days = []
    current = start_date
    
    while current <= end_date:
        if is_trading_day(current):
            trading_days.append(current)
        current += timedelta(days=1)
    
    return trading_days


def get_session_info() -> dict:
    """
    Get detailed session information.
    
    Returns:
        Dictionary with session info
    """
    now = get_current_ist_time()
    current_date = now.date()
    
    status = get_market_status()
    
    info = {
        "status": status,
        "current_time": now.strftime("%H:%M:%S"),
        "current_date": current_date.isoformat(),
        "is_trading_day": is_trading_day(current_date),
    }
    
    if status == "CLOSED":
        next_open = get_next_trading_day(current_date)
        info["next_trading_day"] = next_open.isoformat()
        
        if is_trading_day(current_date) and now.time() < MARKET_OPEN:
            # Market hasn't opened today
            info["opens_in"] = str(
                datetime.combine(current_date, MARKET_OPEN) - now.replace(tzinfo=None)
            )
    elif status == "OPEN":
        close_time = datetime.combine(current_date, MARKET_CLOSE)
        info["closes_in"] = str(close_time - now.replace(tzinfo=None))
    
    return info


def format_nse_symbol(symbol: str) -> str:
    """
    Format symbol for NSE.
    
    Args:
        symbol: Raw symbol
    
    Returns:
        NSE formatted symbol
    """
    symbol = symbol.upper().strip()
    # Remove .NS suffix if present
    if symbol.endswith(".NS"):
        symbol = symbol[:-3]
    # Remove .BO suffix if present
    if symbol.endswith(".BO"):
        symbol = symbol[:-3]
    return symbol


def format_yahoo_symbol(symbol: str, exchange: str = "NS") -> str:
    """
    Format symbol for Yahoo Finance.
    
    Args:
        symbol: Raw symbol
        exchange: Exchange suffix (NS for NSE, BO for BSE)
    
    Returns:
        Yahoo Finance formatted symbol
    """
    symbol = format_nse_symbol(symbol)
    return f"{symbol}.{exchange}"


def get_expiry_dates(months_ahead: int = 3) -> List[date]:
    """
    Get F&O expiry dates (last Thursday of month).
    
    Args:
        months_ahead: Number of months to look ahead
    
    Returns:
        List of expiry dates
    """
    expiries = []
    current = get_current_ist_time().date()
    
    for i in range(months_ahead):
        # Get the month
        month = current.month + i
        year = current.year
        if month > 12:
            month -= 12
            year += 1
        
        # Find last Thursday
        last_day = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year + 1, 1, 1) - timedelta(days=1)
        
        # Find last Thursday
        offset = (last_day.weekday() - 3) % 7
        last_thursday = last_day - timedelta(days=offset)
        
        # If it's a holiday, use previous day
        while not is_trading_day(last_thursday):
            last_thursday -= timedelta(days=1)
        
        expiries.append(last_thursday)
    
    return expiries


def get_lot_size(symbol: str) -> int:
    """
    Get F&O lot size for a symbol.
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Lot size (1 if not in F&O)
    """
    # Common lot sizes (update periodically)
    lot_sizes = {
        "NIFTY": 50,
        "BANKNIFTY": 15,
        "FINNIFTY": 40,
        "RELIANCE": 250,
        "TCS": 150,
        "HDFCBANK": 550,
        "INFY": 400,
        "ICICIBANK": 1375,
        "HINDUNILVR": 300,
        "ITC": 1600,
        "SBIN": 1500,
        "BHARTIARTL": 950,
        "KOTAKBANK": 400,
        "LT": 150,
        "HCLTECH": 350,
        "AXISBANK": 900,
        "ASIANPAINT": 200,
        "MARUTI": 50,
        "SUNPHARMA": 700,
        "TITAN": 175,
        "BAJFINANCE": 125,
        "WIPRO": 1500,
        "ULTRACEMCO": 100,
        "TATAMOTORS": 1425,
        "ONGC": 3850,
        "NTPC": 2000,
        "M&M": 350,
        "POWERGRID": 2700,
        "JSWSTEEL": 675,
        "TATASTEEL": 3000,
        "ADANIENT": 500,
        "ADANIPORTS": 1250,
    }
    
    symbol = format_nse_symbol(symbol)
    return lot_sizes.get(symbol, 1)


def get_circuit_limits(price: float) -> Tuple[float, float]:
    """
    Get circuit filter limits based on price band.
    
    Args:
        price: Current price
    
    Returns:
        Tuple of (lower_limit, upper_limit)
    """
    # Default is 20% circuit
    circuit_percent = 0.20
    
    lower = price * (1 - circuit_percent)
    upper = price * (1 + circuit_percent)
    
    return (round(lower, 2), round(upper, 2))


def get_tick_size(price: float) -> float:
    """
    Get tick size based on price.
    
    Args:
        price: Current price
    
    Returns:
        Tick size
    """
    # NSE uses 0.05 tick size for all
    return 0.05
