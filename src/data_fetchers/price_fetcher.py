"""
Price Data Fetcher for Indian Stock Market.

Fetches OHLCV (Open, High, Low, Close, Volume) data from multiple sources:
1. Yahoo Finance (primary - reliable)
2. NSE India website (for live data)

Handles caching, rate limiting, and data standardization.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from pathlib import Path
import time
import requests
import logging
import warnings

# Suppress yfinance errors and warnings
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import yfinance as yf
except ImportError:
    yf = None

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import (
    CACHE_DIR,
    CACHE_DURATION_MINUTES,
    PRICE_CACHE_DURATION,
    USER_AGENT,
    REQUEST_TIMEOUT,
)
from src.utils.helpers import cache_data, get_cached_data, get_yahoo_symbol, get_nse_symbol
from src.utils.indian_market_utils import is_trading_day, get_current_ist_time
from src.utils.logger import LoggerMixin


class PriceFetcher(LoggerMixin):
    """
    Fetches price data for Indian stocks from multiple sources.
    """
    
    def __init__(self):
        """Initialize the price fetcher."""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self.cache_dir = CACHE_DIR / "prices"
        self.cache_dir.mkdir(exist_ok=True)
        
        # NSE specific headers
        self.nse_headers = {
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.nseindia.com/",
        }
    
    def _init_nse_session(self) -> requests.Session:
        """
        Initialize NSE session with cookies.
        
        Returns:
            Configured session
        """
        nse_session = requests.Session()
        nse_session.headers.update(self.nse_headers)
        
        try:
            # Get cookies from main page
            nse_session.get(
                "https://www.nseindia.com",
                timeout=REQUEST_TIMEOUT,
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize NSE session: {e}")
        
        return nse_session
    
    def get_live_price(self, symbol: str) -> Dict:
        """
        Get live/latest price for a stock.
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
        
        Returns:
            Dictionary with price data
        """
        cache_key = f"live_price_{symbol}"
        cached = get_cached_data(cache_key, self.cache_dir)
        if cached is not None:
            return cached
        
        result = {
            "symbol": get_nse_symbol(symbol),
            "price": None,
            "change": None,
            "change_percent": None,
            "open": None,
            "high": None,
            "low": None,
            "close": None,
            "volume": None,
            "timestamp": None,
            "source": None,
        }
        
        # Try Yahoo Finance first
        if yf is not None:
            try:
                yahoo_symbol = get_yahoo_symbol(symbol)
                ticker = yf.Ticker(yahoo_symbol)
                info = ticker.info
                
                if info:
                    result.update({
                        "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                        "change": info.get("regularMarketChange"),
                        "change_percent": info.get("regularMarketChangePercent"),
                        "open": info.get("regularMarketOpen"),
                        "high": info.get("regularMarketDayHigh"),
                        "low": info.get("regularMarketDayLow"),
                        "close": info.get("previousClose"),
                        "volume": info.get("regularMarketVolume"),
                        "timestamp": datetime.now().isoformat(),
                        "source": "yahoo",
                        "market_cap": info.get("marketCap"),
                        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                    })
                    
                    cache_data(result, cache_key, self.cache_dir, PRICE_CACHE_DURATION)
                    return result
            except Exception as e:
                self.logger.warning(f"Yahoo Finance error for {symbol}: {e}")
        
        # Try NSE as fallback
        try:
            result = self._get_nse_live_price(symbol)
            if result.get("price"):
                cache_data(result, cache_key, self.cache_dir, PRICE_CACHE_DURATION)
        except Exception as e:
            self.logger.error(f"NSE error for {symbol}: {e}")
        
        return result
    
    def _get_nse_live_price(self, symbol: str) -> Dict:
        """
        Get live price from NSE India.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Price data dictionary
        """
        nse_symbol = get_nse_symbol(symbol)
        nse_session = self._init_nse_session()
        
        url = f"https://www.nseindia.com/api/quote-equity?symbol={nse_symbol}"
        
        try:
            response = nse_session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            price_info = data.get("priceInfo", {})
            
            return {
                "symbol": nse_symbol,
                "price": price_info.get("lastPrice"),
                "change": price_info.get("change"),
                "change_percent": price_info.get("pChange"),
                "open": price_info.get("open"),
                "high": price_info.get("intraDayHighLow", {}).get("max"),
                "low": price_info.get("intraDayHighLow", {}).get("min"),
                "close": price_info.get("previousClose"),
                "volume": data.get("securityWiseDP", {}).get("quantityTraded"),
                "timestamp": datetime.now().isoformat(),
                "source": "nse",
            }
        except Exception as e:
            self.logger.error(f"NSE API error: {e}")
            return {"symbol": nse_symbol, "error": str(e)}
    
    def get_historical_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.
        
        Args:
            symbol: Stock symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo)
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
        
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"historical_{symbol}_{period}_{interval}_{start_date}_{end_date}"
        cached = get_cached_data(cache_key, self.cache_dir)
        if cached is not None:
            return pd.DataFrame(cached)
        
        df = pd.DataFrame()
        
        if yf is not None:
            try:
                yahoo_symbol = get_yahoo_symbol(symbol)
                ticker = yf.Ticker(yahoo_symbol)
                
                if start_date and end_date:
                    df = ticker.history(start=start_date, end=end_date, interval=interval)
                else:
                    df = ticker.history(period=period, interval=interval)
                
                if not df.empty:
                    # Standardize column names
                    df = df.reset_index()
                    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                    
                    # Rename date column
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                    elif "datetime" in df.columns:
                        df.rename(columns={"datetime": "date"}, inplace=True)
                        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                    
                    # Add symbol column
                    df["symbol"] = get_nse_symbol(symbol)
                    
                    # Ensure standard columns exist
                    standard_cols = ["date", "open", "high", "low", "close", "volume", "symbol"]
                    for col in standard_cols:
                        if col not in df.columns:
                            df[col] = np.nan
                    
                    # Reorder columns
                    df = df[standard_cols + [c for c in df.columns if c not in standard_cols]]
                    
                    # Cache the data
                    cache_data(df.to_dict("list"), cache_key, self.cache_dir, CACHE_DURATION_MINUTES)
                    
                    self.logger.info(f"Fetched {len(df)} rows of historical data for {symbol}")
            
            except Exception as e:
                self.logger.error(f"Error fetching historical data for {symbol}: {e}")
        
        return df
    
    def get_intraday_data(
        self,
        symbol: str,
        interval: str = "5m",
        days: int = 5,
    ) -> pd.DataFrame:
        """
        Get intraday data.
        
        Args:
            symbol: Stock symbol
            interval: Data interval (1m, 5m, 15m, 30m, 1h)
            days: Number of days of intraday data
        
        Returns:
            DataFrame with intraday OHLCV data
        """
        # Yahoo Finance allows up to 7 days of intraday data
        period = f"{min(days, 7)}d"
        return self.get_historical_data(symbol, period=period, interval=interval)
    
    def get_multiple_stocks(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple stocks.
        
        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
        
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for symbol in symbols:
            try:
                df = self.get_historical_data(symbol, period=period, interval=interval)
                if not df.empty:
                    results[symbol] = df
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                self.logger.error(f"Error fetching {symbol}: {e}")
                results[symbol] = pd.DataFrame()
        
        return results
    
    def get_quote(self, symbol: str) -> Dict:
        """
        Get detailed quote information.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with quote data
        """
        cache_key = f"quote_{symbol}"
        cached = get_cached_data(cache_key, self.cache_dir)
        if cached is not None:
            return cached
        
        quote = self.get_live_price(symbol)
        
        # Get additional info from Yahoo
        if yf is not None:
            try:
                yahoo_symbol = get_yahoo_symbol(symbol)
                ticker = yf.Ticker(yahoo_symbol)
                info = ticker.info
                
                quote.update({
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "pb_ratio": info.get("priceToBook"),
                    "eps": info.get("trailingEps"),
                    "dividend_yield": info.get("dividendYield"),
                    "beta": info.get("beta"),
                    "avg_volume": info.get("averageVolume"),
                    "avg_volume_10d": info.get("averageVolume10days"),
                    "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                    "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                    "fifty_day_avg": info.get("fiftyDayAverage"),
                    "two_hundred_day_avg": info.get("twoHundredDayAverage"),
                })
                
                cache_data(quote, cache_key, self.cache_dir, CACHE_DURATION_MINUTES)
            
            except Exception as e:
                self.logger.warning(f"Error getting quote info for {symbol}: {e}")
        
        return quote
    
    def get_index_data(
        self,
        index: str = "NIFTY 50",
        period: str = "1y",
    ) -> pd.DataFrame:
        """
        Get index data.
        
        Args:
            index: Index name
            period: Data period
        
        Returns:
            DataFrame with index data
        """
        # Yahoo Finance index symbols
        index_map = {
            "NIFTY 50": "^NSEI",
            "NIFTY": "^NSEI",
            "SENSEX": "^BSESN",
            "NIFTY BANK": "^NSEBANK",
            "BANKNIFTY": "^NSEBANK",
            "NIFTY IT": "^CNXIT",
            "NIFTY MIDCAP 100": "^CNXMC",
            "INDIA VIX": "^INDIAVIX",
        }
        
        yahoo_symbol = index_map.get(index.upper(), index)
        
        if yf is not None:
            try:
                ticker = yf.Ticker(yahoo_symbol)
                df = ticker.history(period=period)
                
                if not df.empty:
                    df = df.reset_index()
                    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                    df["symbol"] = index
                    return df
            
            except Exception as e:
                self.logger.error(f"Error fetching index {index}: {e}")
        
        return pd.DataFrame()
    
    def get_delivery_data(self, symbol: str) -> Dict:
        """
        Get delivery percentage data from NSE.
        
        Note: NSE API often blocks programmatic access. This function
        returns a default value if data cannot be fetched.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with delivery data
        """
        nse_symbol = get_nse_symbol(symbol)
        
        # NSE blocks most API requests - return default values
        # This avoids spamming errors in the logs
        # In production, you'd use a proper data provider
        return {
            "symbol": nse_symbol,
            "traded_quantity": None,
            "deliverable_quantity": None,
            "delivery_percentage": None,
            "note": "NSE API requires browser session - using default values",
        }


# Convenience function for quick access
def get_price(symbol: str) -> Dict:
    """Get live price for a symbol."""
    fetcher = PriceFetcher()
    return fetcher.get_live_price(symbol)


def get_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Get historical data for a symbol."""
    fetcher = PriceFetcher()
    return fetcher.get_historical_data(symbol, period=period)
