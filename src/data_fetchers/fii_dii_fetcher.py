"""
FII/DII Data Fetcher for Indian Stock Market.

Fetches Foreign Institutional Investor (FII) and 
Domestic Institutional Investor (DII) data from NSE.

This data is crucial for understanding institutional money flow.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import requests
from bs4 import BeautifulSoup

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import (
    CACHE_DIR,
    CACHE_DURATION_MINUTES,
    USER_AGENT,
    REQUEST_TIMEOUT,
)
from src.utils.helpers import cache_data, get_cached_data
from src.utils.indian_market_utils import get_previous_trading_day, is_trading_day
from src.utils.logger import LoggerMixin


class FIIDIIFetcher(LoggerMixin):
    """
    Fetches FII/DII activity data from NSE and other sources.
    """
    
    def __init__(self):
        """Initialize the FII/DII fetcher."""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
        })
        self.cache_dir = CACHE_DIR / "fii_dii"
        self.cache_dir.mkdir(exist_ok=True)
    
    def _init_nse_session(self) -> requests.Session:
        """
        Initialize NSE session with cookies.
        
        Returns:
            Configured session
        """
        try:
            self.session.get("https://www.nseindia.com", timeout=REQUEST_TIMEOUT)
        except Exception as e:
            self.logger.warning(f"Failed to initialize NSE session: {e}")
        
        return self.session
    
    def get_daily_fii_dii(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get daily FII/DII cash market data.
        
        Args:
            date: Date in YYYY-MM-DD format (default: latest available)
        
        Returns:
            Dictionary with FII/DII data
        """
        cache_key = f"fii_dii_daily_{date or 'latest'}"
        cached = get_cached_data(cache_key, self.cache_dir)
        if cached is not None:
            return cached
        
        result = {
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "timestamp": datetime.now().isoformat(),
            "fii": {
                "buy_value": None,
                "sell_value": None,
                "net_value": None,
            },
            "dii": {
                "buy_value": None,
                "sell_value": None,
                "net_value": None,
            },
            "source": "nse",
        }
        
        try:
            self._init_nse_session()
            
            # NSE FII/DII API
            url = "https://www.nseindia.com/api/fiidiiTradeReact"
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data:
                    category = item.get("category", "").upper()
                    
                    if "FII" in category or "FPI" in category:
                        result["fii"] = {
                            "buy_value": self._parse_crores(item.get("buyValue")),
                            "sell_value": self._parse_crores(item.get("sellValue")),
                            "net_value": self._parse_crores(item.get("netValue")),
                        }
                    elif "DII" in category:
                        result["dii"] = {
                            "buy_value": self._parse_crores(item.get("buyValue")),
                            "sell_value": self._parse_crores(item.get("sellValue")),
                            "net_value": self._parse_crores(item.get("netValue")),
                        }
                
                # Calculate totals
                fii_net = result["fii"]["net_value"] or 0
                dii_net = result["dii"]["net_value"] or 0
                
                result["total_net"] = fii_net + dii_net
                result["fii_dii_sentiment"] = self._get_sentiment(fii_net, dii_net)
                
                cache_data(result, cache_key, self.cache_dir, CACHE_DURATION_MINUTES * 2)
        
        except Exception as e:
            self.logger.error(f"Error fetching FII/DII data: {e}")
            result["error"] = str(e)
        
        return result
    
    def _parse_crores(self, value: Any) -> Optional[float]:
        """
        Parse value to crores.
        
        Args:
            value: Value string or number
        
        Returns:
            Value in crores
        """
        if value is None:
            return None
        
        try:
            if isinstance(value, str):
                value = value.replace(",", "").replace("₹", "").strip()
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _get_sentiment(self, fii_net: float, dii_net: float) -> str:
        """
        Determine overall sentiment based on FII/DII activity.
        
        Args:
            fii_net: FII net value
            dii_net: DII net value
        
        Returns:
            Sentiment string
        """
        total = fii_net + dii_net
        
        if fii_net > 0 and dii_net > 0:
            return "STRONGLY_BULLISH"
        elif fii_net > 0 and dii_net < 0:
            if fii_net > abs(dii_net):
                return "BULLISH"
            return "NEUTRAL"
        elif fii_net < 0 and dii_net > 0:
            if dii_net > abs(fii_net):
                return "NEUTRAL"
            return "CAUTIOUS"
        elif fii_net < 0 and dii_net < 0:
            return "STRONGLY_BEARISH"
        else:
            return "NEUTRAL"
    
    def get_fii_dii_trend(self, days: int = 30) -> pd.DataFrame:
        """
        Get FII/DII trend for multiple days.
        
        Args:
            days: Number of days
        
        Returns:
            DataFrame with daily FII/DII data
        """
        cache_key = f"fii_dii_trend_{days}"
        cached = get_cached_data(cache_key, self.cache_dir)
        if cached is not None:
            return pd.DataFrame(cached)
        
        # For historical data, we'll try to scrape from Moneycontrol
        data = self._fetch_historical_fii_dii(days)
        
        if data:
            df = pd.DataFrame(data)
            cache_data(df.to_dict("list"), cache_key, self.cache_dir, CACHE_DURATION_MINUTES * 4)
            return df
        
        return pd.DataFrame()
    
    def _fetch_historical_fii_dii(self, days: int = 30) -> List[Dict]:
        """
        Fetch historical FII/DII data.
        
        Args:
            days: Number of days
        
        Returns:
            List of daily data dictionaries
        """
        # This would typically scrape from Moneycontrol or NSE archives
        # For now, return empty - in production, implement scraping
        self.logger.info(f"Historical FII/DII data for {days} days not yet implemented")
        return []
    
    def get_stock_institutional_holding(self, symbol: str) -> Dict[str, Any]:
        """
        Get institutional holding for a specific stock.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with holding data
        """
        cache_key = f"institutional_holding_{symbol}"
        cached = get_cached_data(cache_key, self.cache_dir)
        if cached is not None:
            return cached
        
        result = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "fii_holding_percent": None,
            "dii_holding_percent": None,
            "mf_holding_percent": None,
            "fii_shares": None,
            "dii_shares": None,
        }
        
        try:
            self._init_nse_session()
            
            # NSE shareholding API
            url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                
                # Get shareholding info
                security_info = data.get("securityInfo", {})
                
                # Note: Actual holding data requires parsing shareholding pattern
                # This is simplified - in production, parse full shareholding
                
                cache_data(result, cache_key, self.cache_dir, CACHE_DURATION_MINUTES * 4)
        
        except Exception as e:
            self.logger.error(f"Error fetching institutional holding for {symbol}: {e}")
            result["error"] = str(e)
        
        return result
    
    def get_fii_derivative_data(self) -> Dict[str, Any]:
        """
        Get FII derivative (F&O) data.
        
        Returns:
            Dictionary with F&O data
        """
        cache_key = "fii_derivative_data"
        cached = get_cached_data(cache_key, self.cache_dir)
        if cached is not None:
            return cached
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "index_futures": {
                "long": None,
                "short": None,
                "net": None,
            },
            "index_options": {
                "long": None,
                "short": None,
                "net": None,
            },
            "stock_futures": {
                "long": None,
                "short": None,
                "net": None,
            },
            "source": "nse",
        }
        
        try:
            self._init_nse_session()
            
            # NSE FII derivative API
            url = "https://www.nseindia.com/api/fiidiiTradeReact"
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                # Parse derivative data from response
                # Note: Actual implementation depends on NSE API response format
                pass
        
        except Exception as e:
            self.logger.error(f"Error fetching FII derivative data: {e}")
            result["error"] = str(e)
        
        cache_data(result, cache_key, self.cache_dir, CACHE_DURATION_MINUTES)
        
        return result
    
    def get_market_sentiment_indicator(self) -> Dict[str, Any]:
        """
        Get overall market sentiment based on FII/DII activity.
        
        Returns:
            Dictionary with sentiment indicators
        """
        daily = self.get_daily_fii_dii()
        
        fii_net = daily.get("fii", {}).get("net_value", 0) or 0
        dii_net = daily.get("dii", {}).get("net_value", 0) or 0
        total_net = fii_net + dii_net
        
        # Calculate sentiment score (-100 to +100)
        # Normalize based on typical daily flows
        typical_flow = 2000  # Approximate typical daily flow in crores
        sentiment_score = (total_net / typical_flow) * 100
        sentiment_score = max(-100, min(100, sentiment_score))
        
        return {
            "date": daily.get("date"),
            "fii_net": fii_net,
            "dii_net": dii_net,
            "total_net": total_net,
            "sentiment_score": round(sentiment_score, 2),
            "sentiment": daily.get("fii_dii_sentiment", "NEUTRAL"),
            "interpretation": self._get_interpretation(fii_net, dii_net),
        }
    
    def _get_interpretation(self, fii_net: float, dii_net: float) -> str:
        """
        Get human-readable interpretation of FII/DII activity.
        
        Args:
            fii_net: FII net value
            dii_net: DII net value
        
        Returns:
            Interpretation string
        """
        if fii_net > 1000 and dii_net > 0:
            return "Strong institutional buying - bullish outlook"
        elif fii_net > 500:
            return "FII inflows positive - cautiously bullish"
        elif fii_net < -1000 and dii_net < 0:
            return "Heavy institutional selling - bearish pressure"
        elif fii_net < -500:
            return "FII outflows - watch for support"
        elif abs(fii_net) < 200 and abs(dii_net) < 200:
            return "Muted institutional activity - range-bound market likely"
        elif fii_net < 0 and dii_net > abs(fii_net):
            return "DII buying absorbing FII selling - supportive"
        else:
            return "Mixed signals - trade with caution"


# Convenience functions
def get_fii_dii_today() -> Dict:
    """Get today's FII/DII data."""
    fetcher = FIIDIIFetcher()
    return fetcher.get_daily_fii_dii()


def get_market_sentiment() -> Dict:
    """Get market sentiment based on institutional activity."""
    fetcher = FIIDIIFetcher()
    return fetcher.get_market_sentiment_indicator()
