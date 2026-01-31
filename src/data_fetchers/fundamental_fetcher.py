"""
Fundamental Data Fetcher for Indian Stock Market.

Fetches fundamental data including:
- Financial ratios (P/E, P/B, ROE, etc.)
- Quarterly and annual results
- Shareholding patterns
- Peer comparisons

Data sources:
1. Yahoo Finance (basic fundamentals)
2. Screener.in (detailed financials via scraping)
3. NSE India (shareholding patterns)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import re
import json

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
from src.utils.helpers import cache_data, get_cached_data, get_yahoo_symbol, get_nse_symbol
from src.utils.logger import LoggerMixin


class FundamentalFetcher(LoggerMixin):
    """
    Fetches fundamental data for Indian stocks.
    """
    
    def __init__(self):
        """Initialize the fundamental fetcher."""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self.cache_dir = CACHE_DIR / "fundamentals"
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive fundamental data for a stock.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with fundamental data
        """
        cache_key = f"fundamentals_{symbol}"
        cached = get_cached_data(cache_key, self.cache_dir)
        if cached is not None:
            return cached
        
        fundamentals = {
            "symbol": get_nse_symbol(symbol),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Get Yahoo Finance data
        yahoo_data = self._get_yahoo_fundamentals(symbol)
        fundamentals.update(yahoo_data)
        
        # Get Screener.in data (if available)
        screener_data = self._get_screener_data(symbol)
        fundamentals["screener"] = screener_data
        
        cache_data(fundamentals, cache_key, self.cache_dir, CACHE_DURATION_MINUTES * 4)
        
        return fundamentals
    
    def _get_yahoo_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Get fundamental data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with Yahoo fundamentals
        """
        if yf is None:
            return {}
        
        try:
            yahoo_symbol = get_yahoo_symbol(symbol)
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info
            
            return {
                # Valuation
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "pb_ratio": info.get("priceToBook"),
                "ps_ratio": info.get("priceToSalesTrailing12Months"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
                "ev_to_revenue": info.get("enterpriseToRevenue"),
                
                # Profitability
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "gross_margin": info.get("grossMargins"),
                "ebitda_margin": info.get("ebitdaMargins"),
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
                
                # Growth
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),
                
                # Financial Health
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
                "debt_to_equity": info.get("debtToEquity"),
                "total_debt": info.get("totalDebt"),
                "total_cash": info.get("totalCash"),
                "free_cashflow": info.get("freeCashflow"),
                "operating_cashflow": info.get("operatingCashflow"),
                
                # Per Share
                "eps_trailing": info.get("trailingEps"),
                "eps_forward": info.get("forwardEps"),
                "book_value": info.get("bookValue"),
                "revenue_per_share": info.get("revenuePerShare"),
                
                # Dividends
                "dividend_rate": info.get("dividendRate"),
                "dividend_yield": info.get("dividendYield"),
                "payout_ratio": info.get("payoutRatio"),
                "five_year_avg_dividend_yield": info.get("fiveYearAvgDividendYield"),
                
                # Company Info
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "full_time_employees": info.get("fullTimeEmployees"),
                "business_summary": info.get("longBusinessSummary"),
            }
        
        except Exception as e:
            self.logger.error(f"Error fetching Yahoo fundamentals for {symbol}: {e}")
            return {}
    
    def _get_screener_data(self, symbol: str) -> Dict[str, Any]:
        """
        Scrape fundamental data from Screener.in.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with Screener data
        """
        nse_symbol = get_nse_symbol(symbol)
        url = f"https://www.screener.in/company/{nse_symbol}/"
        
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            
            if response.status_code != 200:
                self.logger.warning(f"Screener returned {response.status_code} for {symbol}")
                return {}
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            data = {}
            
            # Get key ratios from the top section
            ratios_list = soup.find("ul", {"id": "top-ratios"})
            if ratios_list:
                for li in ratios_list.find_all("li"):
                    name_span = li.find("span", {"class": "name"})
                    value_span = li.find("span", {"class": "number"})
                    if name_span and value_span:
                        name = name_span.text.strip().lower().replace(" ", "_")
                        value = self._parse_value(value_span.text.strip())
                        data[name] = value
            
            # Get quarterly results table
            quarterly_table = soup.find("table", {"class": "data-table"})
            if quarterly_table:
                data["quarterly_results"] = self._parse_table(quarterly_table)
            
            # Get pros and cons
            pros_section = soup.find("div", {"class": "pros"})
            if pros_section:
                pros = [li.text.strip() for li in pros_section.find_all("li")]
                data["pros"] = pros
            
            cons_section = soup.find("div", {"class": "cons"})
            if cons_section:
                cons = [li.text.strip() for li in cons_section.find_all("li")]
                data["cons"] = cons
            
            return data
        
        except Exception as e:
            self.logger.error(f"Error scraping Screener for {symbol}: {e}")
            return {}
    
    def _parse_value(self, value_str: str) -> Optional[float]:
        """
        Parse a value string to float.
        
        Args:
            value_str: Value string
        
        Returns:
            Parsed float or None
        """
        try:
            # Remove commas and percentage signs
            value_str = value_str.replace(",", "").replace("%", "").strip()
            
            # Handle Cr (Crores) and L (Lakhs)
            if "Cr" in value_str:
                value_str = value_str.replace("Cr", "").strip()
                return float(value_str) * 1e7
            elif "L" in value_str:
                value_str = value_str.replace("L", "").strip()
                return float(value_str) * 1e5
            
            return float(value_str)
        except (ValueError, AttributeError):
            return None
    
    def _parse_table(self, table) -> List[Dict]:
        """
        Parse an HTML table to list of dictionaries.
        
        Args:
            table: BeautifulSoup table element
        
        Returns:
            List of row dictionaries
        """
        rows = []
        headers = []
        
        # Get headers
        header_row = table.find("thead")
        if header_row:
            headers = [th.text.strip() for th in header_row.find_all("th")]
        
        # Get data rows
        tbody = table.find("tbody")
        if tbody:
            for tr in tbody.find_all("tr"):
                row_data = {}
                cells = tr.find_all(["td", "th"])
                for i, cell in enumerate(cells):
                    key = headers[i] if i < len(headers) else f"col_{i}"
                    row_data[key] = cell.text.strip()
                rows.append(row_data)
        
        return rows
    
    def get_quarterly_results(self, symbol: str) -> pd.DataFrame:
        """
        Get quarterly financial results.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            DataFrame with quarterly results
        """
        if yf is None:
            return pd.DataFrame()
        
        try:
            yahoo_symbol = get_yahoo_symbol(symbol)
            ticker = yf.Ticker(yahoo_symbol)
            
            # Get quarterly financials
            quarterly = ticker.quarterly_financials
            
            if quarterly is not None and not quarterly.empty:
                df = quarterly.T.reset_index()
                df.columns = ["date"] + list(quarterly.index)
                return df
        
        except Exception as e:
            self.logger.error(f"Error fetching quarterly results for {symbol}: {e}")
        
        return pd.DataFrame()
    
    def get_annual_results(self, symbol: str) -> pd.DataFrame:
        """
        Get annual financial results.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            DataFrame with annual results
        """
        if yf is None:
            return pd.DataFrame()
        
        try:
            yahoo_symbol = get_yahoo_symbol(symbol)
            ticker = yf.Ticker(yahoo_symbol)
            
            # Get annual financials
            annual = ticker.financials
            
            if annual is not None and not annual.empty:
                df = annual.T.reset_index()
                df.columns = ["date"] + list(annual.index)
                return df
        
        except Exception as e:
            self.logger.error(f"Error fetching annual results for {symbol}: {e}")
        
        return pd.DataFrame()
    
    def get_shareholding_pattern(self, symbol: str) -> Dict[str, Any]:
        """
        Get shareholding pattern from NSE.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with shareholding data
        """
        cache_key = f"shareholding_{symbol}"
        cached = get_cached_data(cache_key, self.cache_dir)
        if cached is not None:
            return cached
        
        nse_symbol = get_nse_symbol(symbol)
        
        # Initialize NSE session
        nse_session = requests.Session()
        nse_session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
        })
        
        try:
            # Get cookies first
            nse_session.get("https://www.nseindia.com", timeout=REQUEST_TIMEOUT)
            
            # Get shareholding data
            url = f"https://www.nseindia.com/api/quote-equity?symbol={nse_symbol}&section=trade_info"
            response = nse_session.get(url, timeout=REQUEST_TIMEOUT)
            data = response.json()
            
            shareholding = {
                "symbol": nse_symbol,
                "timestamp": datetime.now().isoformat(),
            }
            
            if "shareholdingPattern" in data:
                pattern = data["shareholdingPattern"]
                shareholding.update({
                    "promoter_holding": pattern.get("promoterAndPromoterGroup"),
                    "fii_holding": pattern.get("foreignInstitutions"),
                    "dii_holding": pattern.get("domesticInstitutions"),
                    "public_holding": pattern.get("publicShareholding"),
                })
            
            cache_data(shareholding, cache_key, self.cache_dir, CACHE_DURATION_MINUTES * 4)
            return shareholding
        
        except Exception as e:
            self.logger.error(f"Error fetching shareholding for {symbol}: {e}")
            return {"symbol": nse_symbol, "error": str(e)}
    
    def get_financial_ratios(self, symbol: str) -> Dict[str, float]:
        """
        Get key financial ratios.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with ratios
        """
        fundamentals = self.get_fundamentals(symbol)
        
        return {
            "pe_ratio": fundamentals.get("pe_ratio"),
            "pb_ratio": fundamentals.get("pb_ratio"),
            "ps_ratio": fundamentals.get("ps_ratio"),
            "peg_ratio": fundamentals.get("peg_ratio"),
            "ev_to_ebitda": fundamentals.get("ev_to_ebitda"),
            "debt_to_equity": fundamentals.get("debt_to_equity"),
            "current_ratio": fundamentals.get("current_ratio"),
            "roe": fundamentals.get("roe"),
            "roa": fundamentals.get("roa"),
            "profit_margin": fundamentals.get("profit_margin"),
            "dividend_yield": fundamentals.get("dividend_yield"),
        }
    
    def get_peer_comparison(self, symbol: str, peers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare fundamental metrics with peers.
        
        Args:
            symbol: Stock symbol
            peers: List of peer symbols (auto-detected if None)
        
        Returns:
            DataFrame with peer comparison
        """
        # Get sector peers if not provided
        if peers is None:
            fundamentals = self.get_fundamentals(symbol)
            sector = fundamentals.get("sector", "")
            
            # Get peers from same sector (simplified)
            from config.settings import SECTORS
            for sector_name, stocks in SECTORS.items():
                if symbol in stocks:
                    peers = [s for s in stocks if s != symbol][:5]
                    break
        
        if not peers:
            peers = []
        
        # Get data for all stocks
        all_symbols = [symbol] + peers
        comparison_data = []
        
        for sym in all_symbols:
            try:
                ratios = self.get_financial_ratios(sym)
                ratios["symbol"] = sym
                ratios["is_target"] = sym == symbol
                comparison_data.append(ratios)
            except Exception as e:
                self.logger.warning(f"Error getting data for {sym}: {e}")
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            # Reorder columns
            cols = ["symbol", "is_target"] + [c for c in df.columns if c not in ["symbol", "is_target"]]
            return df[cols]
        
        return pd.DataFrame()
    
    def detect_red_flags(self, symbol: str) -> List[Dict[str, str]]:
        """
        Detect fundamental red flags.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            List of red flags with descriptions
        """
        red_flags = []
        fundamentals = self.get_fundamentals(symbol)
        
        # Check debt levels
        debt_to_equity = fundamentals.get("debt_to_equity")
        if debt_to_equity and debt_to_equity > 2:
            red_flags.append({
                "type": "HIGH_DEBT",
                "severity": "HIGH" if debt_to_equity > 3 else "MEDIUM",
                "description": f"High debt-to-equity ratio: {debt_to_equity:.2f}",
            })
        
        # Check profitability
        profit_margin = fundamentals.get("profit_margin")
        if profit_margin and profit_margin < 0:
            red_flags.append({
                "type": "NEGATIVE_MARGINS",
                "severity": "HIGH",
                "description": f"Negative profit margin: {profit_margin:.2%}",
            })
        
        # Check ROE
        roe = fundamentals.get("roe")
        if roe and roe < 0.05:
            red_flags.append({
                "type": "LOW_ROE",
                "severity": "MEDIUM",
                "description": f"Low return on equity: {roe:.2%}",
            })
        
        # Check current ratio
        current_ratio = fundamentals.get("current_ratio")
        if current_ratio and current_ratio < 1:
            red_flags.append({
                "type": "LIQUIDITY_CONCERN",
                "severity": "MEDIUM",
                "description": f"Low current ratio: {current_ratio:.2f}",
            })
        
        # Check earnings growth
        earnings_growth = fundamentals.get("earnings_quarterly_growth")
        if earnings_growth and earnings_growth < -0.20:
            red_flags.append({
                "type": "EARNINGS_DECLINE",
                "severity": "MEDIUM",
                "description": f"Earnings declining: {earnings_growth:.2%}",
            })
        
        # Check from screener cons
        screener = fundamentals.get("screener", {})
        cons = screener.get("cons", [])
        for con in cons[:3]:  # Top 3 concerns
            red_flags.append({
                "type": "SCREENER_CON",
                "severity": "LOW",
                "description": con,
            })
        
        return red_flags
    
    def calculate_fundamental_score(self, symbol: str) -> float:
        """
        Calculate an overall fundamental score (0-100).
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Fundamental score
        """
        fundamentals = self.get_fundamentals(symbol)
        score = 50  # Start at neutral
        
        # Valuation (max ±15 points)
        pe = fundamentals.get("pe_ratio")
        if pe:
            if pe < 15:
                score += 10
            elif pe < 25:
                score += 5
            elif pe > 40:
                score -= 10
            elif pe > 30:
                score -= 5
        
        # Profitability (max ±15 points)
        roe = fundamentals.get("roe")
        if roe:
            if roe > 0.20:
                score += 10
            elif roe > 0.15:
                score += 5
            elif roe < 0.05:
                score -= 10
        
        profit_margin = fundamentals.get("profit_margin")
        if profit_margin:
            if profit_margin > 0.15:
                score += 5
            elif profit_margin < 0:
                score -= 10
        
        # Financial health (max ±10 points)
        debt_to_equity = fundamentals.get("debt_to_equity")
        if debt_to_equity:
            if debt_to_equity < 0.5:
                score += 5
            elif debt_to_equity > 2:
                score -= 10
            elif debt_to_equity > 1:
                score -= 5
        
        current_ratio = fundamentals.get("current_ratio")
        if current_ratio:
            if current_ratio > 2:
                score += 5
            elif current_ratio < 1:
                score -= 5
        
        # Growth (max ±10 points)
        earnings_growth = fundamentals.get("earnings_growth")
        if earnings_growth:
            if earnings_growth > 0.20:
                score += 10
            elif earnings_growth > 0.10:
                score += 5
            elif earnings_growth < 0:
                score -= 5
        
        # Cap score between 0 and 100
        return max(0, min(100, score))


# Convenience functions
def get_fundamentals(symbol: str) -> Dict:
    """Get fundamentals for a symbol."""
    fetcher = FundamentalFetcher()
    return fetcher.get_fundamentals(symbol)


def get_ratios(symbol: str) -> Dict:
    """Get financial ratios for a symbol."""
    fetcher = FundamentalFetcher()
    return fetcher.get_financial_ratios(symbol)
