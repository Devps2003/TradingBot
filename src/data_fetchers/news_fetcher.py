"""
News Fetcher for Indian Stock Market.

Fetches news from multiple sources for sentiment analysis:
1. Google News RSS
2. Moneycontrol
3. Economic Times
4. Business Standard

Handles rate limiting and caching.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import (
    CACHE_DIR,
    NEWS_CACHE_DURATION,
    USER_AGENT,
    REQUEST_TIMEOUT,
)
from src.utils.helpers import cache_data, get_cached_data, get_nse_symbol
from src.utils.logger import LoggerMixin


class NewsFetcher(LoggerMixin):
    """
    Fetches news from multiple sources for Indian stocks.
    """
    
    def __init__(self):
        """Initialize the news fetcher."""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self.cache_dir = CACHE_DIR / "news"
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_stock_news(
        self,
        symbol: str,
        days: int = 7,
        max_articles: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get news for a specific stock.
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            max_articles: Maximum number of articles to return
        
        Returns:
            List of news articles
        """
        cache_key = f"stock_news_{symbol}_{days}"
        cached = get_cached_data(cache_key, self.cache_dir)
        if cached is not None:
            return cached
        
        nse_symbol = get_nse_symbol(symbol)
        all_news = []
        
        # Get company name variations for better search
        company_names = self._get_company_names(nse_symbol)
        
        # Fetch from Google News
        for name in company_names[:2]:  # Limit to 2 search terms
            google_news = self._fetch_google_news(f"{name} NSE stock", days)
            all_news.extend(google_news)
        
        # Fetch from Moneycontrol
        mc_news = self._fetch_moneycontrol_news(nse_symbol)
        all_news.extend(mc_news)
        
        # Remove duplicates based on title similarity
        unique_news = self._deduplicate_news(all_news)
        
        # Sort by date and limit
        unique_news.sort(key=lambda x: x.get("published_date", ""), reverse=True)
        result = unique_news[:max_articles]
        
        cache_data(result, cache_key, self.cache_dir, NEWS_CACHE_DURATION)
        
        return result
    
    def _get_company_names(self, symbol: str) -> List[str]:
        """
        Get company name variations for a symbol.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            List of company name variations
        """
        # Common mappings
        company_map = {
            "RELIANCE": ["Reliance Industries", "RIL"],
            "TCS": ["Tata Consultancy Services", "TCS"],
            "HDFCBANK": ["HDFC Bank"],
            "INFY": ["Infosys"],
            "ICICIBANK": ["ICICI Bank"],
            "HINDUNILVR": ["Hindustan Unilever", "HUL"],
            "ITC": ["ITC Limited"],
            "SBIN": ["State Bank of India", "SBI"],
            "BHARTIARTL": ["Bharti Airtel", "Airtel"],
            "KOTAKBANK": ["Kotak Mahindra Bank"],
            "LT": ["Larsen & Toubro", "L&T"],
            "HCLTECH": ["HCL Technologies"],
            "AXISBANK": ["Axis Bank"],
            "ASIANPAINT": ["Asian Paints"],
            "MARUTI": ["Maruti Suzuki"],
            "SUNPHARMA": ["Sun Pharma", "Sun Pharmaceutical"],
            "TITAN": ["Titan Company"],
            "BAJFINANCE": ["Bajaj Finance"],
            "WIPRO": ["Wipro"],
            "TATAMOTORS": ["Tata Motors"],
            "TATASTEEL": ["Tata Steel"],
        }
        
        return company_map.get(symbol, [symbol])
    
    def _fetch_google_news(self, query: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Fetch news from Google News RSS.
        
        Args:
            query: Search query
            days: Days to look back
        
        Returns:
            List of news articles
        """
        news_items = []
        encoded_query = quote_plus(query)
        
        # Google News RSS URL
        url = f"https://news.google.com/rss/search?q={encoded_query}+when:{days}d&hl=en-IN&gl=IN&ceid=IN:en"
        
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                
                for item in root.findall(".//item"):
                    title = item.find("title")
                    link = item.find("link")
                    pub_date = item.find("pubDate")
                    source = item.find("source")
                    
                    news_items.append({
                        "title": title.text if title is not None else "",
                        "url": link.text if link is not None else "",
                        "published_date": self._parse_date(pub_date.text) if pub_date is not None else "",
                        "source": source.text if source is not None else "Google News",
                        "summary": "",
                        "raw_text": title.text if title is not None else "",
                    })
        
        except Exception as e:
            self.logger.warning(f"Error fetching Google News: {e}")
        
        return news_items
    
    def _fetch_moneycontrol_news(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetch news from Moneycontrol.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            List of news articles
        """
        news_items = []
        
        # Moneycontrol search URL
        url = f"https://www.moneycontrol.com/news/tags/{symbol.lower()}.html"
        
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Find news articles
                articles = soup.find_all("li", {"class": "clearfix"})[:10]
                
                for article in articles:
                    title_elem = article.find("h2")
                    link_elem = article.find("a")
                    date_elem = article.find("span")
                    summary_elem = article.find("p")
                    
                    if title_elem:
                        news_items.append({
                            "title": title_elem.text.strip() if title_elem else "",
                            "url": link_elem.get("href", "") if link_elem else "",
                            "published_date": date_elem.text.strip() if date_elem else "",
                            "source": "Moneycontrol",
                            "summary": summary_elem.text.strip() if summary_elem else "",
                            "raw_text": f"{title_elem.text.strip()} {summary_elem.text.strip() if summary_elem else ''}",
                        })
        
        except Exception as e:
            self.logger.warning(f"Error fetching Moneycontrol news: {e}")
        
        return news_items
    
    def _parse_date(self, date_str: str) -> str:
        """
        Parse date string to ISO format.
        
        Args:
            date_str: Date string
        
        Returns:
            ISO formatted date string
        """
        try:
            # Try common formats
            formats = [
                "%a, %d %b %Y %H:%M:%S %Z",
                "%a, %d %b %Y %H:%M:%S %z",
                "%Y-%m-%dT%H:%M:%S",
                "%d %b %Y",
                "%B %d, %Y",
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(date_str.strip(), fmt)
                    return dt.isoformat()
                except ValueError:
                    continue
            
            return date_str
        except Exception:
            return date_str
    
    def _deduplicate_news(self, news_list: List[Dict]) -> List[Dict]:
        """
        Remove duplicate news articles based on title similarity.
        
        Args:
            news_list: List of news articles
        
        Returns:
            Deduplicated list
        """
        unique = []
        seen_titles = set()
        
        for news in news_list:
            title = news.get("title", "").lower()
            # Simplify title for comparison
            simple_title = re.sub(r'[^\w\s]', '', title)[:50]
            
            if simple_title not in seen_titles:
                seen_titles.add(simple_title)
                unique.append(news)
        
        return unique
    
    def get_sector_news(self, sector: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get news for a sector.
        
        Args:
            sector: Sector name (e.g., "BANKING", "IT")
            days: Number of days to look back
        
        Returns:
            List of news articles
        """
        cache_key = f"sector_news_{sector}_{days}"
        cached = get_cached_data(cache_key, self.cache_dir)
        if cached is not None:
            return cached
        
        sector_queries = {
            "BANKING": "Indian banking sector NSE",
            "IT": "Indian IT sector technology Infosys TCS",
            "PHARMA": "Indian pharma sector pharmaceutical",
            "AUTO": "Indian auto sector automobile",
            "FMCG": "Indian FMCG consumer goods",
            "METALS": "Indian metals steel sector",
            "ENERGY": "Indian energy sector oil gas",
            "REALTY": "Indian real estate sector",
        }
        
        query = sector_queries.get(sector.upper(), f"Indian {sector} sector")
        news = self._fetch_google_news(query, days)
        
        cache_data(news, cache_key, self.cache_dir, NEWS_CACHE_DURATION)
        
        return news
    
    def get_market_news(self, days: int = 3) -> List[Dict[str, Any]]:
        """
        Get general market news.
        
        Args:
            days: Number of days to look back
        
        Returns:
            List of news articles
        """
        cache_key = f"market_news_{days}"
        cached = get_cached_data(cache_key, self.cache_dir)
        if cached is not None:
            return cached
        
        all_news = []
        
        # Market-wide queries
        queries = [
            "Nifty Sensex stock market India",
            "NSE BSE market today",
            "FII DII Indian market",
        ]
        
        for query in queries:
            news = self._fetch_google_news(query, days)
            all_news.extend(news)
        
        unique_news = self._deduplicate_news(all_news)
        unique_news.sort(key=lambda x: x.get("published_date", ""), reverse=True)
        
        cache_data(unique_news[:20], cache_key, self.cache_dir, NEWS_CACHE_DURATION)
        
        return unique_news[:20]
    
    def get_article_content(self, url: str) -> str:
        """
        Fetch full article content from URL.
        
        Args:
            url: Article URL
        
        Returns:
            Article text content
        """
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Remove script and style elements
                for element in soup(["script", "style", "nav", "header", "footer"]):
                    element.decompose()
                
                # Try to find article content
                article = soup.find("article")
                if article:
                    return article.get_text(separator=" ", strip=True)
                
                # Fallback to main content
                main = soup.find("main")
                if main:
                    return main.get_text(separator=" ", strip=True)
                
                # Last resort - body text
                return soup.body.get_text(separator=" ", strip=True)[:5000]
        
        except Exception as e:
            self.logger.warning(f"Error fetching article content: {e}")
        
        return ""
    
    def get_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get social media sentiment (placeholder for future implementation).
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with social sentiment data
        """
        # This would integrate with Twitter/Reddit APIs
        # For now, return placeholder data
        return {
            "symbol": symbol,
            "twitter_sentiment": None,
            "reddit_sentiment": None,
            "mention_count": 0,
            "sentiment_score": 0,
            "trending": False,
            "note": "Social media integration requires API keys",
        }
    
    def analyze_news_volume(
        self,
        symbol: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Analyze news volume and trends.
        
        Args:
            symbol: Stock symbol
            days: Days to analyze
        
        Returns:
            News volume analysis
        """
        news = self.get_stock_news(symbol, days=days, max_articles=50)
        
        if not news:
            return {
                "symbol": symbol,
                "total_articles": 0,
                "avg_per_day": 0,
                "recent_spike": False,
            }
        
        # Count articles by date
        date_counts = {}
        for article in news:
            date_str = article.get("published_date", "")[:10]
            if date_str:
                date_counts[date_str] = date_counts.get(date_str, 0) + 1
        
        total = len(news)
        avg_per_day = total / days if days > 0 else 0
        
        # Check for recent spike
        recent_count = sum(1 for n in news if self._is_recent(n.get("published_date", ""), 3))
        expected_recent = avg_per_day * 3
        recent_spike = recent_count > expected_recent * 2 if expected_recent > 0 else False
        
        return {
            "symbol": symbol,
            "total_articles": total,
            "avg_per_day": round(avg_per_day, 2),
            "recent_count_3d": recent_count,
            "recent_spike": recent_spike,
            "date_distribution": date_counts,
        }
    
    def _is_recent(self, date_str: str, days: int) -> bool:
        """
        Check if a date is within recent days.
        
        Args:
            date_str: ISO date string
            days: Number of days
        
        Returns:
            True if recent
        """
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            cutoff = datetime.now() - timedelta(days=days)
            return dt.replace(tzinfo=None) > cutoff
        except Exception:
            return False


# Convenience functions
def get_news(symbol: str, days: int = 7) -> List[Dict]:
    """Get news for a symbol."""
    fetcher = NewsFetcher()
    return fetcher.get_stock_news(symbol, days=days)


def get_market_news(days: int = 3) -> List[Dict]:
    """Get general market news."""
    fetcher = NewsFetcher()
    return fetcher.get_market_news(days=days)
