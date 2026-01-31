"""
Insider Trading Data Fetcher for Indian Stock Market.

Fetches SAST (Substantial Acquisition of Shares and Takeovers) data:
- Promoter buying/selling
- Key management personnel trades
- Large shareholder movements

This data provides insights into insider confidence.
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
from src.utils.helpers import cache_data, get_cached_data, get_nse_symbol
from src.utils.logger import LoggerMixin


class InsiderFetcher(LoggerMixin):
    """
    Fetches insider trading data from NSE and other sources.
    """
    
    def __init__(self):
        """Initialize the insider fetcher."""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
        })
        self.cache_dir = CACHE_DIR / "insider"
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
    
    def get_insider_trades(self, days: int = 30) -> pd.DataFrame:
        """
        Get all insider trades for recent days.
        
        Args:
            days: Number of days to fetch
        
        Returns:
            DataFrame with insider trades
        """
        cache_key = f"insider_trades_{days}"
        cached = get_cached_data(cache_key, self.cache_dir)
        if cached is not None:
            return pd.DataFrame(cached)
        
        trades = []
        
        try:
            self._init_nse_session()
            
            # NSE Insider Trading API
            url = "https://www.nseindia.com/api/corporates-pit"
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get("data", []):
                    trade = {
                        "broadcast_date": item.get("date"),
                        "symbol": item.get("symbol"),
                        "company": item.get("company"),
                        "regulation": item.get("anession"),
                        "person_name": item.get("personName"),
                        "person_category": item.get("personCategory"),
                        "transaction_type": item.get("acqMode"),
                        "securities_type": item.get("secType"),
                        "quantity": self._parse_number(item.get("secAcq")),
                        "value": self._parse_number(item.get("secVal")),
                        "holding_before": item.get("befAcqSharesNo"),
                        "holding_after": item.get("aftAcqSharesNo"),
                        "holding_before_pct": self._parse_number(item.get("befAcqSharesPer")),
                        "holding_after_pct": self._parse_number(item.get("aftAcqSharesPer")),
                    }
                    trades.append(trade)
                
                if trades:
                    df = pd.DataFrame(trades)
                    df["broadcast_date"] = pd.to_datetime(df["broadcast_date"])
                    
                    # Filter by days
                    cutoff = datetime.now() - timedelta(days=days)
                    df = df[df["broadcast_date"] >= cutoff]
                    df = df.sort_values("broadcast_date", ascending=False)
                    
                    cache_data(df.to_dict("list"), cache_key, self.cache_dir, CACHE_DURATION_MINUTES * 2)
                    return df
        
        except Exception as e:
            self.logger.error(f"Error fetching insider trades: {e}")
        
        return pd.DataFrame(trades)
    
    def _parse_number(self, value: Any) -> float:
        """
        Parse number from string.
        
        Args:
            value: Value to parse
        
        Returns:
            Parsed float
        """
        if value is None:
            return 0.0
        
        try:
            if isinstance(value, str):
                value = value.replace(",", "").replace("-", "0").strip()
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def get_stock_insider_activity(
        self,
        symbol: str,
        days: int = 90,
    ) -> pd.DataFrame:
        """
        Get insider activity for a specific stock.
        
        Args:
            symbol: Stock symbol
            days: Days to look back
        
        Returns:
            DataFrame with insider activity
        """
        nse_symbol = get_nse_symbol(symbol)
        
        all_trades = self.get_insider_trades(days)
        
        if all_trades.empty:
            return pd.DataFrame()
        
        stock_trades = all_trades[all_trades["symbol"] == nse_symbol]
        return stock_trades
    
    def get_promoter_transactions(
        self,
        symbol: str,
        days: int = 180,
    ) -> pd.DataFrame:
        """
        Get promoter buying/selling transactions.
        
        Args:
            symbol: Stock symbol
            days: Days to look back
        
        Returns:
            DataFrame with promoter transactions
        """
        stock_trades = self.get_stock_insider_activity(symbol, days)
        
        if stock_trades.empty:
            return pd.DataFrame()
        
        # Filter for promoter category
        promoter_keywords = ["promoter", "promoter group", "pag"]
        promoter_trades = stock_trades[
            stock_trades["person_category"].str.lower().str.contains(
                "|".join(promoter_keywords), na=False
            )
        ]
        
        return promoter_trades
    
    def analyze_insider_sentiment(
        self,
        symbol: str,
        days: int = 90,
    ) -> Dict[str, Any]:
        """
        Analyze insider sentiment based on trading activity.
        
        Args:
            symbol: Stock symbol
            days: Days to analyze
        
        Returns:
            Dictionary with analysis
        """
        trades = self.get_stock_insider_activity(symbol, days)
        
        result = {
            "symbol": symbol,
            "period_days": days,
            "total_transactions": len(trades),
            "buy_count": 0,
            "sell_count": 0,
            "net_shares": 0,
            "net_value": 0,
            "sentiment": "NEUTRAL",
            "promoter_activity": "NONE",
            "key_transactions": [],
            "insight": "",
        }
        
        if trades.empty:
            result["insight"] = "No insider transactions in the analysis period"
            return result
        
        # Categorize transactions
        buys = trades[trades["transaction_type"].str.upper().str.contains("BUY|ACQUISITION", na=False)]
        sells = trades[trades["transaction_type"].str.upper().str.contains("SELL|DISPOSAL|SALE", na=False)]
        
        result["buy_count"] = len(buys)
        result["sell_count"] = len(sells)
        result["buy_value"] = buys["value"].sum() if not buys.empty else 0
        result["sell_value"] = sells["value"].sum() if not sells.empty else 0
        result["net_value"] = result["buy_value"] - result["sell_value"]
        
        # Check promoter activity
        promoter_trades = self.get_promoter_transactions(symbol, days)
        if not promoter_trades.empty:
            prom_buys = promoter_trades[
                promoter_trades["transaction_type"].str.upper().str.contains("BUY|ACQUISITION", na=False)
            ]
            prom_sells = promoter_trades[
                promoter_trades["transaction_type"].str.upper().str.contains("SELL|DISPOSAL", na=False)
            ]
            
            if len(prom_buys) > len(prom_sells):
                result["promoter_activity"] = "BUYING"
            elif len(prom_sells) > len(prom_buys):
                result["promoter_activity"] = "SELLING"
            else:
                result["promoter_activity"] = "MIXED"
        
        # Key transactions (largest by value)
        trades_sorted = trades.sort_values("value", ascending=False)
        for _, row in trades_sorted.head(5).iterrows():
            result["key_transactions"].append({
                "date": str(row.get("broadcast_date", ""))[:10],
                "person": row.get("person_name", "Unknown"),
                "category": row.get("person_category", ""),
                "type": row.get("transaction_type", ""),
                "value": row.get("value", 0),
            })
        
        # Determine sentiment
        if result["net_value"] > 1e7:  # > 1 crore net buying
            if result["promoter_activity"] == "BUYING":
                result["sentiment"] = "STRONGLY_BULLISH"
                result["insight"] = "Significant insider and promoter buying - high conviction signal"
            else:
                result["sentiment"] = "BULLISH"
                result["insight"] = "Net insider buying indicates positive outlook"
        elif result["net_value"] < -1e7:  # > 1 crore net selling
            if result["promoter_activity"] == "SELLING":
                result["sentiment"] = "STRONGLY_BEARISH"
                result["insight"] = "Promoter selling is a warning sign - exercise caution"
            else:
                result["sentiment"] = "BEARISH"
                result["insight"] = "Net insider selling may indicate concerns"
        else:
            result["sentiment"] = "NEUTRAL"
            result["insight"] = "No significant insider trading pattern detected"
        
        return result
    
    def get_recent_promoter_activity(
        self,
        min_value: float = 1e7,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get recent significant promoter transactions across all stocks.
        
        Args:
            min_value: Minimum transaction value
            days: Days to look back
        
        Returns:
            List of significant promoter transactions
        """
        all_trades = self.get_insider_trades(days)
        
        if all_trades.empty:
            return []
        
        # Filter for promoters
        promoter_keywords = ["promoter", "promoter group", "pag"]
        promoter_trades = all_trades[
            all_trades["person_category"].str.lower().str.contains(
                "|".join(promoter_keywords), na=False
            )
        ]
        
        # Filter by value
        significant = promoter_trades[promoter_trades["value"] >= min_value]
        
        result = []
        for _, row in significant.sort_values("value", ascending=False).iterrows():
            result.append({
                "symbol": row.get("symbol"),
                "company": row.get("company"),
                "date": str(row.get("broadcast_date", ""))[:10],
                "person": row.get("person_name"),
                "type": row.get("transaction_type"),
                "value": row.get("value"),
                "holding_change": row.get("holding_after_pct", 0) - row.get("holding_before_pct", 0),
            })
        
        return result
    
    def get_insider_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get summary of insider activity across the market.
        
        Args:
            days: Days to analyze
        
        Returns:
            Dictionary with summary
        """
        trades = self.get_insider_trades(days)
        
        if trades.empty:
            return {"period": f"Last {days} days", "total_transactions": 0}
        
        buys = trades[trades["transaction_type"].str.upper().str.contains("BUY|ACQUISITION", na=False)]
        sells = trades[trades["transaction_type"].str.upper().str.contains("SELL|DISPOSAL", na=False)]
        
        return {
            "period": f"Last {days} days",
            "total_transactions": len(trades),
            "total_buys": len(buys),
            "total_sells": len(sells),
            "buy_value": buys["value"].sum() if not buys.empty else 0,
            "sell_value": sells["value"].sum() if not sells.empty else 0,
            "net_value": (buys["value"].sum() if not buys.empty else 0) - 
                        (sells["value"].sum() if not sells.empty else 0),
            "stocks_with_buying": buys["symbol"].nunique() if not buys.empty else 0,
            "stocks_with_selling": sells["symbol"].nunique() if not sells.empty else 0,
            "market_sentiment": "BULLISH" if len(buys) > len(sells) else "BEARISH" if len(sells) > len(buys) else "NEUTRAL",
        }


# Convenience functions
def get_insider_trades(days: int = 30) -> pd.DataFrame:
    """Get recent insider trades."""
    fetcher = InsiderFetcher()
    return fetcher.get_insider_trades(days)


def analyze_insider(symbol: str) -> Dict:
    """Analyze insider activity for a stock."""
    fetcher = InsiderFetcher()
    return fetcher.analyze_insider_sentiment(symbol)
