"""
Bulk and Block Deals Fetcher for Indian Stock Market.

Fetches bulk deals and block deals data from NSE.
This is critical for swing trading as it reveals:
- Institutional accumulation/distribution
- Large player activity
- Potential breakout candidates

Bulk Deal: Trade >= 0.5% of equity shares
Block Deal: Trade of 5 lakh shares or Rs. 10 crore, minimum
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import requests

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


class BulkDealsFetcher(LoggerMixin):
    """
    Fetches bulk and block deals data from NSE.
    """
    
    def __init__(self):
        """Initialize the bulk deals fetcher."""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
        })
        self.cache_dir = CACHE_DIR / "bulk_deals"
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
    
    def get_bulk_deals(self, days: int = 30) -> pd.DataFrame:
        """
        Get bulk deals for recent days.
        
        Args:
            days: Number of days to fetch
        
        Returns:
            DataFrame with bulk deals
        """
        cache_key = f"bulk_deals_{days}"
        cached = get_cached_data(cache_key, self.cache_dir)
        if cached is not None:
            return pd.DataFrame(cached)
        
        deals = []
        
        try:
            self._init_nse_session()
            
            # NSE Bulk Deals API
            url = "https://www.nseindia.com/api/snapshot-capital-market-largedeal"
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                
                # Parse bulk deals
                bulk_data = data.get("BULK_DEALS_DATA", [])
                
                for deal in bulk_data:
                    deals.append({
                        "date": deal.get("BD_DT_DATE"),
                        "symbol": deal.get("BD_SYMBOL"),
                        "security_name": deal.get("BD_SCRIP_NAME"),
                        "client_name": deal.get("BD_CLIENT_NAME"),
                        "deal_type": deal.get("BD_BUY_SELL"),
                        "quantity": self._parse_number(deal.get("BD_QTY_TRD")),
                        "price": self._parse_number(deal.get("BD_TP_WATP")),
                        "remarks": deal.get("BD_REMARKS"),
                    })
                
                if deals:
                    df = pd.DataFrame(deals)
                    df["value_cr"] = (df["quantity"] * df["price"]) / 1e7
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values("date", ascending=False)
                    
                    cache_data(df.to_dict("list"), cache_key, self.cache_dir, CACHE_DURATION_MINUTES)
                    return df
        
        except Exception as e:
            self.logger.error(f"Error fetching bulk deals: {e}")
        
        return pd.DataFrame(deals)
    
    def get_block_deals(self, days: int = 30) -> pd.DataFrame:
        """
        Get block deals for recent days.
        
        Args:
            days: Number of days to fetch
        
        Returns:
            DataFrame with block deals
        """
        cache_key = f"block_deals_{days}"
        cached = get_cached_data(cache_key, self.cache_dir)
        if cached is not None:
            return pd.DataFrame(cached)
        
        deals = []
        
        try:
            self._init_nse_session()
            
            # NSE Block Deals API
            url = "https://www.nseindia.com/api/snapshot-capital-market-largedeal"
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                
                # Parse block deals
                block_data = data.get("BLOCK_DEALS_DATA", [])
                
                for deal in block_data:
                    deals.append({
                        "date": deal.get("BD_DT_DATE"),
                        "symbol": deal.get("BD_SYMBOL"),
                        "security_name": deal.get("BD_SCRIP_NAME"),
                        "client_name": deal.get("BD_CLIENT_NAME"),
                        "deal_type": deal.get("BD_BUY_SELL"),
                        "quantity": self._parse_number(deal.get("BD_QTY_TRD")),
                        "price": self._parse_number(deal.get("BD_TP_WATP")),
                    })
                
                if deals:
                    df = pd.DataFrame(deals)
                    df["value_cr"] = (df["quantity"] * df["price"]) / 1e7
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values("date", ascending=False)
                    
                    cache_data(df.to_dict("list"), cache_key, self.cache_dir, CACHE_DURATION_MINUTES)
                    return df
        
        except Exception as e:
            self.logger.error(f"Error fetching block deals: {e}")
        
        return pd.DataFrame(deals)
    
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
                value = value.replace(",", "").strip()
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def get_stock_bulk_deals(
        self,
        symbol: str,
        days: int = 90,
    ) -> pd.DataFrame:
        """
        Get bulk/block deals for a specific stock.
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
        
        Returns:
            DataFrame with deals for the stock
        """
        nse_symbol = get_nse_symbol(symbol)
        
        # Get all deals
        bulk = self.get_bulk_deals(days)
        block = self.get_block_deals(days)
        
        # Filter for specific stock
        stock_bulk = bulk[bulk["symbol"] == nse_symbol] if not bulk.empty else pd.DataFrame()
        stock_block = block[block["symbol"] == nse_symbol] if not block.empty else pd.DataFrame()
        
        # Combine
        if not stock_bulk.empty or not stock_block.empty:
            combined = pd.concat([stock_bulk, stock_block], ignore_index=True)
            combined["deal_category"] = combined.apply(
                lambda x: "BLOCK" if x.name in stock_block.index else "BULK",
                axis=1
            )
            return combined.sort_values("date", ascending=False)
        
        return pd.DataFrame()
    
    def analyze_accumulation_distribution(
        self,
        symbol: str,
        days: int = 90,
    ) -> Dict[str, Any]:
        """
        Analyze accumulation/distribution based on bulk/block deals.
        
        Args:
            symbol: Stock symbol
            days: Days to analyze
        
        Returns:
            Dictionary with analysis results
        """
        deals = self.get_stock_bulk_deals(symbol, days)
        
        result = {
            "symbol": symbol,
            "period_days": days,
            "total_deals": len(deals),
            "buy_count": 0,
            "sell_count": 0,
            "net_quantity": 0,
            "net_value_cr": 0,
            "pattern": "NEUTRAL",
            "major_buyers": [],
            "major_sellers": [],
            "insight": "",
        }
        
        if deals.empty:
            result["insight"] = "No bulk/block deals in the analysis period"
            return result
        
        # Analyze buy vs sell
        buys = deals[deals["deal_type"].str.upper() == "BUY"]
        sells = deals[deals["deal_type"].str.upper() == "SELL"]
        
        result["buy_count"] = len(buys)
        result["sell_count"] = len(sells)
        result["buy_quantity"] = buys["quantity"].sum() if not buys.empty else 0
        result["sell_quantity"] = sells["quantity"].sum() if not sells.empty else 0
        result["net_quantity"] = result["buy_quantity"] - result["sell_quantity"]
        
        result["buy_value_cr"] = buys["value_cr"].sum() if not buys.empty else 0
        result["sell_value_cr"] = sells["value_cr"].sum() if not sells.empty else 0
        result["net_value_cr"] = result["buy_value_cr"] - result["sell_value_cr"]
        
        # Identify major participants
        if not buys.empty:
            buyer_agg = buys.groupby("client_name")["value_cr"].sum().sort_values(ascending=False)
            result["major_buyers"] = [
                {"name": name, "value_cr": round(val, 2)}
                for name, val in buyer_agg.head(3).items()
            ]
        
        if not sells.empty:
            seller_agg = sells.groupby("client_name")["value_cr"].sum().sort_values(ascending=False)
            result["major_sellers"] = [
                {"name": name, "value_cr": round(val, 2)}
                for name, val in seller_agg.head(3).items()
            ]
        
        # Determine pattern
        buy_ratio = result["buy_count"] / max(result["total_deals"], 1)
        
        if result["net_value_cr"] > 10 and buy_ratio > 0.6:
            result["pattern"] = "STRONG_ACCUMULATION"
            result["insight"] = "Significant institutional buying detected - bullish signal"
        elif result["net_value_cr"] > 5 and buy_ratio > 0.5:
            result["pattern"] = "ACCUMULATION"
            result["insight"] = "Net buying activity suggests accumulation phase"
        elif result["net_value_cr"] < -10 and buy_ratio < 0.4:
            result["pattern"] = "STRONG_DISTRIBUTION"
            result["insight"] = "Heavy selling by large players - bearish warning"
        elif result["net_value_cr"] < -5 and buy_ratio < 0.5:
            result["pattern"] = "DISTRIBUTION"
            result["insight"] = "Net selling indicates distribution phase"
        else:
            result["pattern"] = "NEUTRAL"
            result["insight"] = "Mixed activity - no clear accumulation/distribution"
        
        return result
    
    def get_recent_significant_deals(
        self,
        min_value_cr: float = 50,
        days: int = 7,
    ) -> pd.DataFrame:
        """
        Get recent significant deals above a threshold.
        
        Args:
            min_value_cr: Minimum deal value in crores
            days: Days to look back
        
        Returns:
            DataFrame with significant deals
        """
        bulk = self.get_bulk_deals(days)
        block = self.get_block_deals(days)
        
        combined = pd.concat([bulk, block], ignore_index=True)
        
        if combined.empty:
            return pd.DataFrame()
        
        # Filter by value
        significant = combined[combined["value_cr"] >= min_value_cr]
        
        return significant.sort_values("value_cr", ascending=False)
    
    def get_deal_summary(self) -> Dict[str, Any]:
        """
        Get a summary of recent bulk/block deal activity.
        
        Returns:
            Dictionary with deal summary
        """
        bulk = self.get_bulk_deals(7)
        block = self.get_block_deals(7)
        
        return {
            "period": "Last 7 days",
            "total_bulk_deals": len(bulk),
            "total_block_deals": len(block),
            "total_bulk_value_cr": bulk["value_cr"].sum() if not bulk.empty else 0,
            "total_block_value_cr": block["value_cr"].sum() if not block.empty else 0,
            "top_stocks_by_deals": self._get_top_stocks(pd.concat([bulk, block])),
            "buy_sell_ratio": self._get_buy_sell_ratio(pd.concat([bulk, block])),
        }
    
    def _get_top_stocks(self, deals: pd.DataFrame) -> List[Dict]:
        """Get top stocks by deal activity."""
        if deals.empty:
            return []
        
        stock_counts = deals.groupby("symbol").size().sort_values(ascending=False)
        return [
            {"symbol": sym, "deal_count": int(count)}
            for sym, count in stock_counts.head(10).items()
        ]
    
    def _get_buy_sell_ratio(self, deals: pd.DataFrame) -> float:
        """Get buy to sell ratio."""
        if deals.empty:
            return 1.0
        
        buys = len(deals[deals["deal_type"].str.upper() == "BUY"])
        sells = len(deals[deals["deal_type"].str.upper() == "SELL"])
        
        return round(buys / max(sells, 1), 2)


# Convenience functions
def get_bulk_deals(days: int = 30) -> pd.DataFrame:
    """Get recent bulk deals."""
    fetcher = BulkDealsFetcher()
    return fetcher.get_bulk_deals(days)


def analyze_stock_deals(symbol: str) -> Dict:
    """Analyze bulk/block deals for a stock."""
    fetcher = BulkDealsFetcher()
    return fetcher.analyze_accumulation_distribution(symbol)
