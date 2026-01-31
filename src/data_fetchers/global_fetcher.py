"""
Global Market Data Fetcher.

Fetches global market data that affects Indian markets:
- US Markets (S&P 500, Nasdaq, Dow)
- Asian Markets (Nikkei, Hang Seng, SGX Nifty)
- Commodities (Crude Oil, Gold)
- Currency (USD/INR)
- Volatility (VIX)
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
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
    USER_AGENT,
    REQUEST_TIMEOUT,
)
from src.utils.helpers import cache_data, get_cached_data
from src.utils.logger import LoggerMixin


class GlobalFetcher(LoggerMixin):
    """
    Fetches global market data relevant to Indian markets.
    """
    
    # Yahoo Finance symbols for global indices
    INDICES = {
        # US Markets
        "SP500": "^GSPC",
        "NASDAQ": "^IXIC",
        "DOW": "^DJI",
        "RUSSELL": "^RUT",
        
        # US Futures
        "SP500_FUTURES": "ES=F",
        "NASDAQ_FUTURES": "NQ=F",
        "DOW_FUTURES": "YM=F",
        
        # Asian Markets
        "NIKKEI": "^N225",
        "HANG_SENG": "^HSI",
        "SHANGHAI": "000001.SS",
        "KOSPI": "^KS11",
        "TAIWAN": "^TWII",
        
        # European Markets
        "FTSE": "^FTSE",
        "DAX": "^GDAXI",
        "CAC40": "^FCHI",
        
        # India
        "NIFTY": "^NSEI",
        "SENSEX": "^BSESN",
        "NIFTY_BANK": "^NSEBANK",
        "INDIA_VIX": "^INDIAVIX",
        
        # Volatility
        "VIX": "^VIX",
    }
    
    COMMODITIES = {
        "CRUDE_WTI": "CL=F",
        "CRUDE_BRENT": "BZ=F",
        "GOLD": "GC=F",
        "SILVER": "SI=F",
        "COPPER": "HG=F",
        "NATURAL_GAS": "NG=F",
    }
    
    CURRENCIES = {
        "USDINR": "USDINR=X",
        "DXY": "DX-Y.NYB",  # US Dollar Index
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "USDJPY=X",
    }
    
    BONDS = {
        "US10Y": "^TNX",  # 10-Year Treasury Yield
        "US2Y": "^IRX",   # 2-Year Treasury
    }
    
    def __init__(self):
        """Initialize the global fetcher."""
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.cache_dir = CACHE_DIR / "global"
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_ticker_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get current data for a ticker.
        
        Args:
            symbol: Yahoo Finance symbol
        
        Returns:
            Dictionary with ticker data
        """
        if yf is None:
            return {}
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get recent history for change calculation
            hist = ticker.history(period="2d")
            
            current = info.get("regularMarketPrice") or info.get("previousClose")
            prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose")
            
            if hist is not None and len(hist) >= 2:
                current = hist["Close"].iloc[-1]
                prev_close = hist["Close"].iloc[-2]
            
            change = current - prev_close if current and prev_close else None
            change_pct = (change / prev_close * 100) if change and prev_close else None
            
            return {
                "price": current,
                "prev_close": prev_close,
                "change": round(change, 2) if change else None,
                "change_pct": round(change_pct, 2) if change_pct else None,
                "high": info.get("regularMarketDayHigh"),
                "low": info.get("regularMarketDayLow"),
                "timestamp": datetime.now().isoformat(),
            }
        
        except Exception as e:
            self.logger.warning(f"Error fetching {symbol}: {e}")
            return {}
    
    def get_global_indices(self) -> Dict[str, Any]:
        """
        Get current data for major global indices.
        
        Returns:
            Dictionary with index data
        """
        cache_key = "global_indices"
        cached = get_cached_data(cache_key, self.cache_dir)
        if cached is not None:
            return cached
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "us_markets": {},
            "asian_markets": {},
            "european_markets": {},
            "indian_markets": {},
        }
        
        # US Markets
        for name in ["SP500", "NASDAQ", "DOW"]:
            result["us_markets"][name.lower()] = self._get_ticker_data(self.INDICES[name])
        
        # US Futures (for pre-market indication)
        for name in ["SP500_FUTURES", "NASDAQ_FUTURES"]:
            clean_name = name.lower().replace("_futures", "_fut")
            result["us_markets"][clean_name] = self._get_ticker_data(self.INDICES[name])
        
        # Asian Markets
        for name in ["NIKKEI", "HANG_SENG", "SHANGHAI"]:
            result["asian_markets"][name.lower()] = self._get_ticker_data(self.INDICES[name])
        
        # European Markets
        for name in ["FTSE", "DAX"]:
            result["european_markets"][name.lower()] = self._get_ticker_data(self.INDICES[name])
        
        # Indian Markets
        for name in ["NIFTY", "SENSEX", "NIFTY_BANK"]:
            result["indian_markets"][name.lower()] = self._get_ticker_data(self.INDICES[name])
        
        cache_data(result, cache_key, self.cache_dir, 5)  # 5 minute cache
        
        return result
    
    def get_commodities(self) -> Dict[str, Any]:
        """
        Get current commodity prices.
        
        Returns:
            Dictionary with commodity data
        """
        cache_key = "commodities"
        cached = get_cached_data(cache_key, self.cache_dir)
        if cached is not None:
            return cached
        
        result = {
            "timestamp": datetime.now().isoformat(),
        }
        
        for name, symbol in self.COMMODITIES.items():
            result[name.lower()] = self._get_ticker_data(symbol)
        
        cache_data(result, cache_key, self.cache_dir, 5)
        
        return result
    
    def get_currency_data(self) -> Dict[str, Any]:
        """
        Get currency exchange rates.
        
        Returns:
            Dictionary with currency data
        """
        cache_key = "currencies"
        cached = get_cached_data(cache_key, self.cache_dir)
        if cached is not None:
            return cached
        
        result = {
            "timestamp": datetime.now().isoformat(),
        }
        
        for name, symbol in self.CURRENCIES.items():
            result[name.lower()] = self._get_ticker_data(symbol)
        
        cache_data(result, cache_key, self.cache_dir, 5)
        
        return result
    
    def get_vix_data(self) -> Dict[str, Any]:
        """
        Get volatility index data.
        
        Returns:
            Dictionary with VIX data
        """
        cache_key = "vix_data"
        cached = get_cached_data(cache_key, self.cache_dir)
        if cached is not None:
            return cached
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "us_vix": self._get_ticker_data(self.INDICES["VIX"]),
            "india_vix": self._get_ticker_data(self.INDICES["INDIA_VIX"]),
        }
        
        # Add interpretation
        us_vix = result["us_vix"].get("price", 0)
        india_vix = result["india_vix"].get("price", 0)
        
        result["us_vix_level"] = self._interpret_vix(us_vix)
        result["india_vix_level"] = self._interpret_vix(india_vix)
        
        cache_data(result, cache_key, self.cache_dir, 5)
        
        return result
    
    def _interpret_vix(self, vix_value: float) -> str:
        """
        Interpret VIX level.
        
        Args:
            vix_value: VIX value
        
        Returns:
            Interpretation string
        """
        if vix_value is None or vix_value == 0:
            return "UNKNOWN"
        elif vix_value < 12:
            return "VERY_LOW"
        elif vix_value < 16:
            return "LOW"
        elif vix_value < 20:
            return "NORMAL"
        elif vix_value < 25:
            return "ELEVATED"
        elif vix_value < 30:
            return "HIGH"
        else:
            return "VERY_HIGH"
    
    def get_sgx_nifty(self) -> Dict[str, Any]:
        """
        Get SGX Nifty (Singapore Nifty futures) as pre-market indicator.
        
        Returns:
            Dictionary with SGX Nifty data
        """
        # Note: SGX Nifty has been discontinued, using alternative
        # This would need to be updated with a valid data source
        cache_key = "sgx_nifty"
        cached = get_cached_data(cache_key, self.cache_dir)
        if cached is not None:
            return cached
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "note": "SGX Nifty discontinued. Using GIFT Nifty instead.",
            "gift_nifty": self._get_ticker_data("^NSEI"),  # Placeholder
        }
        
        cache_data(result, cache_key, self.cache_dir, 5)
        
        return result
    
    def get_bond_yields(self) -> Dict[str, Any]:
        """
        Get bond yield data.
        
        Returns:
            Dictionary with bond yield data
        """
        cache_key = "bond_yields"
        cached = get_cached_data(cache_key, self.cache_dir)
        if cached is not None:
            return cached
        
        result = {
            "timestamp": datetime.now().isoformat(),
        }
        
        for name, symbol in self.BONDS.items():
            result[name.lower()] = self._get_ticker_data(symbol)
        
        cache_data(result, cache_key, self.cache_dir, 5)
        
        return result
    
    def get_global_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of global markets.
        
        Returns:
            Dictionary with global market summary
        """
        indices = self.get_global_indices()
        commodities = self.get_commodities()
        currencies = self.get_currency_data()
        vix = self.get_vix_data()
        bonds = self.get_bond_yields()
        
        # Calculate sentiment indicators
        us_sentiment = self._calculate_region_sentiment(indices.get("us_markets", {}))
        asia_sentiment = self._calculate_region_sentiment(indices.get("asian_markets", {}))
        
        return {
            "timestamp": datetime.now().isoformat(),
            "indices": indices,
            "commodities": commodities,
            "currencies": currencies,
            "vix": vix,
            "bonds": bonds,
            "sentiment": {
                "us_markets": us_sentiment,
                "asian_markets": asia_sentiment,
                "overall": self._combine_sentiment(us_sentiment, asia_sentiment),
            },
            "key_highlights": self._generate_highlights(indices, commodities, currencies, vix),
        }
    
    def _calculate_region_sentiment(self, markets: Dict) -> str:
        """
        Calculate sentiment for a region based on market changes.
        
        Args:
            markets: Dictionary of market data
        
        Returns:
            Sentiment string
        """
        positive = 0
        negative = 0
        
        for market_data in markets.values():
            if isinstance(market_data, dict):
                change = market_data.get("change_pct", 0) or 0
                if change > 0.5:
                    positive += 1
                elif change < -0.5:
                    negative += 1
        
        if positive > negative + 1:
            return "BULLISH"
        elif negative > positive + 1:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _combine_sentiment(self, *sentiments: str) -> str:
        """
        Combine multiple sentiments into overall sentiment.
        
        Args:
            sentiments: Multiple sentiment strings
        
        Returns:
            Combined sentiment
        """
        bullish = sum(1 for s in sentiments if s == "BULLISH")
        bearish = sum(1 for s in sentiments if s == "BEARISH")
        
        if bullish > bearish:
            return "BULLISH"
        elif bearish > bullish:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _generate_highlights(
        self,
        indices: Dict,
        commodities: Dict,
        currencies: Dict,
        vix: Dict,
    ) -> List[str]:
        """
        Generate key market highlights.
        
        Args:
            indices: Index data
            commodities: Commodity data
            currencies: Currency data
            vix: VIX data
        
        Returns:
            List of highlight strings
        """
        highlights = []
        
        # US Market highlights
        sp500 = indices.get("us_markets", {}).get("sp500", {})
        if sp500.get("change_pct"):
            change = sp500["change_pct"]
            if abs(change) > 1:
                direction = "up" if change > 0 else "down"
                highlights.append(f"S&P 500 {direction} {abs(change):.1f}%")
        
        # Crude oil
        crude = commodities.get("crude_wti", {})
        if crude.get("change_pct"):
            change = crude["change_pct"]
            if abs(change) > 2:
                direction = "surging" if change > 0 else "falling"
                highlights.append(f"Crude oil {direction} {abs(change):.1f}%")
        
        # USD/INR
        usdinr = currencies.get("usdinr", {})
        if usdinr.get("price"):
            highlights.append(f"USD/INR at {usdinr['price']:.2f}")
        
        # VIX
        india_vix = vix.get("india_vix", {})
        if india_vix.get("price"):
            level = vix.get("india_vix_level", "")
            if level in ["HIGH", "VERY_HIGH"]:
                highlights.append(f"India VIX elevated at {india_vix['price']:.1f}")
            elif level in ["LOW", "VERY_LOW"]:
                highlights.append(f"India VIX low at {india_vix['price']:.1f} - favorable for longs")
        
        return highlights
    
    def get_market_cues_for_india(self) -> Dict[str, Any]:
        """
        Get specific global cues relevant for Indian market open.
        
        Returns:
            Dictionary with cues for Indian market
        """
        summary = self.get_global_summary()
        
        # Extract key cues
        us_markets = summary.get("indices", {}).get("us_markets", {})
        asia_markets = summary.get("indices", {}).get("asian_markets", {})
        indian_markets = summary.get("indices", {}).get("indian_markets", {})
        commodities = summary.get("commodities", {})
        currencies = summary.get("currencies", {})
        vix = summary.get("vix", {})
        
        # Calculate expected Nifty impact
        sp500_change = us_markets.get("sp500", {}).get("change_pct", 0) or 0
        nasdaq_change = us_markets.get("nasdaq", {}).get("change_pct", 0) or 0
        asia_avg = sum(
            (m.get("change_pct", 0) or 0)
            for m in asia_markets.values()
            if isinstance(m, dict)
        ) / max(len(asia_markets), 1)
        
        # Simple model for expected impact
        expected_impact = (sp500_change * 0.4 + nasdaq_change * 0.3 + asia_avg * 0.3)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "indian_markets": indian_markets,  # Nifty, Sensex, BankNifty
            "us_close": {
                "sp500": us_markets.get("sp500", {}),
                "nasdaq": us_markets.get("nasdaq", {}),
                "dow": us_markets.get("dow", {}),
            },
            "us_futures": {
                "sp500_fut": us_markets.get("sp500_fut", {}),
                "nasdaq_fut": us_markets.get("nasdaq_fut", {}),
            },
            "asian_markets": asia_markets,
            "crude_oil": commodities.get("crude_wti", {}),
            "gold": commodities.get("gold", {}),
            "usdinr": currencies.get("usdinr", {}),
            "dxy": currencies.get("dxy", {}),
            "india_vix": vix.get("india_vix", {}),
            "expected_nifty_impact": round(expected_impact, 2),
            "expected_gap": "GAP_UP" if expected_impact > 0.3 else "GAP_DOWN" if expected_impact < -0.3 else "FLAT",
            "overall_sentiment": summary.get("sentiment", {}).get("overall", "NEUTRAL"),
            "key_highlights": summary.get("key_highlights", []),
        }


# Convenience functions
def get_global_markets() -> Dict:
    """Get global market data."""
    fetcher = GlobalFetcher()
    return fetcher.get_global_indices()


def get_market_cues() -> Dict:
    """Get global cues for Indian market."""
    fetcher = GlobalFetcher()
    return fetcher.get_market_cues_for_india()
