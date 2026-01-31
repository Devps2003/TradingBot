"""
Sentiment Analysis Module.

Analyzes sentiment from:
- News articles
- Headlines
- Social media (if available)

Uses NLP models for sentiment classification.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import re

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import SENTIMENT_SETTINGS
from src.data_fetchers.news_fetcher import NewsFetcher
from src.utils.logger import LoggerMixin


class SentimentAnalyzer(LoggerMixin):
    """
    Performs sentiment analysis on text and news.
    """
    
    def __init__(self, model: str = "vader"):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model: Model to use ("vader", "finbert", or "auto")
        """
        self.settings = SENTIMENT_SETTINGS
        self.model_type = model
        self.news_fetcher = NewsFetcher()
        
        # Initialize VADER
        self.vader = None
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
        
        # Initialize FinBERT (if requested and available)
        self.finbert = None
        if model == "finbert" and TRANSFORMERS_AVAILABLE:
            try:
                self.finbert = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    truncation=True,
                    max_length=512,
                )
                self.logger.info("FinBERT model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load FinBERT: {e}")
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Sentiment analysis result
        """
        if not text or not text.strip():
            return {
                "sentiment": "NEUTRAL",
                "score": 0,
                "confidence": 0,
            }
        
        # Clean text
        text = self._clean_text(text)
        
        # Use FinBERT if available
        if self.finbert:
            return self._analyze_with_finbert(text)
        
        # Fall back to VADER
        if self.vader:
            return self._analyze_with_vader(text)
        
        # Last resort: simple keyword analysis
        return self._analyze_with_keywords(text)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text for analysis.
        
        Args:
            text: Raw text
        
        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def _analyze_with_finbert(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using FinBERT.
        
        Args:
            text: Text to analyze
        
        Returns:
            Sentiment result
        """
        try:
            result = self.finbert(text[:512])[0]
            label = result["label"].upper()
            score = result["score"]
            
            # Map FinBERT labels
            if label == "POSITIVE":
                sentiment = "POSITIVE"
                sentiment_score = score * 100
            elif label == "NEGATIVE":
                sentiment = "NEGATIVE"
                sentiment_score = -score * 100
            else:
                sentiment = "NEUTRAL"
                sentiment_score = 0
            
            return {
                "sentiment": sentiment,
                "score": round(sentiment_score, 2),
                "confidence": round(score * 100, 2),
                "model": "finbert",
            }
        except Exception as e:
            self.logger.warning(f"FinBERT error: {e}")
            return self._analyze_with_vader(text)
    
    def _analyze_with_vader(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using VADER.
        
        Args:
            text: Text to analyze
        
        Returns:
            Sentiment result
        """
        scores = self.vader.polarity_scores(text)
        compound = scores["compound"]
        
        if compound >= self.settings["positive_threshold"]:
            sentiment = "POSITIVE"
        elif compound <= self.settings["negative_threshold"]:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"
        
        return {
            "sentiment": sentiment,
            "score": round(compound * 100, 2),
            "confidence": round(abs(compound) * 100, 2),
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
            "model": "vader",
        }
    
    def _analyze_with_keywords(self, text: str) -> Dict[str, Any]:
        """
        Simple keyword-based sentiment analysis.
        
        Args:
            text: Text to analyze
        
        Returns:
            Sentiment result
        """
        text_lower = text.lower()
        
        positive_words = [
            "bullish", "surge", "rally", "gain", "profit", "growth",
            "strong", "upgrade", "outperform", "buy", "positive",
            "beat", "exceed", "record", "high", "breakthrough",
        ]
        
        negative_words = [
            "bearish", "fall", "drop", "decline", "loss", "weak",
            "downgrade", "underperform", "sell", "negative", "miss",
            "concern", "risk", "low", "crash", "plunge",
        ]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return {
                "sentiment": "NEUTRAL",
                "score": 0,
                "confidence": 30,
                "model": "keywords",
            }
        
        score = (pos_count - neg_count) / total * 100
        
        if score > 20:
            sentiment = "POSITIVE"
        elif score < -20:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"
        
        return {
            "sentiment": sentiment,
            "score": round(score, 2),
            "confidence": min(50, total * 10),
            "model": "keywords",
        }
    
    def analyze_news_batch(
        self,
        news_list: List[Dict],
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of multiple news articles.
        
        Args:
            news_list: List of news dictionaries
        
        Returns:
            Aggregated sentiment analysis
        """
        if not news_list:
            return {
                "overall_sentiment": "NEUTRAL",
                "overall_score": 0,
                "article_count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
            }
        
        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for news in news_list:
            # Combine title and summary for analysis
            text = f"{news.get('title', '')} {news.get('summary', '')}"
            result = self.analyze_text_sentiment(text)
            
            # Apply source weight
            source = news.get("source", "unknown").lower()
            weight = self.settings["source_weights"].get(source, 0.7)
            
            weighted_score = result["score"] * weight
            sentiments.append(weighted_score)
            
            if result["sentiment"] == "POSITIVE":
                positive_count += 1
            elif result["sentiment"] == "NEGATIVE":
                negative_count += 1
            else:
                neutral_count += 1
        
        # Calculate overall
        avg_score = sum(sentiments) / len(sentiments) if sentiments else 0
        
        if avg_score > 20:
            overall = "POSITIVE"
        elif avg_score < -20:
            overall = "NEGATIVE"
        else:
            overall = "NEUTRAL"
        
        return {
            "overall_sentiment": overall,
            "overall_score": round(avg_score, 2),
            "article_count": len(news_list),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "sentiment_distribution": {
                "positive": positive_count / len(news_list) * 100,
                "negative": negative_count / len(news_list) * 100,
                "neutral": neutral_count / len(news_list) * 100,
            },
        }
    
    def get_stock_sentiment(
        self,
        symbol: str,
        days: int = 7,
    ) -> Dict[str, Any]:
        """
        Get sentiment analysis for a stock.
        
        Args:
            symbol: Stock symbol
            days: Days of news to analyze
        
        Returns:
            Stock sentiment analysis
        """
        # Fetch news
        news = self.news_fetcher.get_stock_news(symbol, days=days)
        
        if not news:
            return {
                "symbol": symbol,
                "overall_sentiment": "NEUTRAL",
                "overall_score": 0,
                "confidence": 0,
                "news_count": 0,
                "note": "No news found",
            }
        
        # Analyze news batch
        batch_result = self.analyze_news_batch(news)
        
        # Analyze individual headlines
        headline_sentiments = []
        for n in news[:10]:  # Top 10 headlines
            text = n.get("title", "")
            result = self.analyze_text_sentiment(text)
            headline_sentiments.append({
                "headline": text[:100],
                "sentiment": result["sentiment"],
                "score": result["score"],
            })
        
        # Detect sentiment trend
        recent_news = news[:5]
        older_news = news[5:10] if len(news) > 5 else []
        
        recent_result = self.analyze_news_batch(recent_news)
        older_result = self.analyze_news_batch(older_news)
        
        if older_news:
            trend_change = recent_result["overall_score"] - older_result["overall_score"]
            if trend_change > 20:
                trend = "IMPROVING"
            elif trend_change < -20:
                trend = "DETERIORATING"
            else:
                trend = "STABLE"
        else:
            trend = "UNKNOWN"
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "overall_sentiment": batch_result["overall_sentiment"],
            "overall_score": batch_result["overall_score"],
            "news_count": batch_result["article_count"],
            "positive_count": batch_result["positive_count"],
            "negative_count": batch_result["negative_count"],
            "sentiment_trend": trend,
            "confidence": min(90, batch_result["article_count"] * 10),
            "top_headlines": headline_sentiments[:5],
            "distribution": batch_result.get("sentiment_distribution"),
        }
    
    def detect_sentiment_shift(
        self,
        symbol: str,
        days: int = 7,
    ) -> Dict[str, Any]:
        """
        Detect sentiment shift over time.
        
        Args:
            symbol: Stock symbol
            days: Days to analyze
        
        Returns:
            Sentiment shift analysis
        """
        news = self.news_fetcher.get_stock_news(symbol, days=days)
        
        if len(news) < 3:
            return {
                "symbol": symbol,
                "shift_detected": False,
                "note": "Insufficient news for trend analysis",
            }
        
        # Split into recent and older
        mid = len(news) // 2
        recent = news[:mid]
        older = news[mid:]
        
        recent_sentiment = self.analyze_news_batch(recent)
        older_sentiment = self.analyze_news_batch(older)
        
        score_change = recent_sentiment["overall_score"] - older_sentiment["overall_score"]
        
        return {
            "symbol": symbol,
            "recent_sentiment": recent_sentiment["overall_sentiment"],
            "recent_score": recent_sentiment["overall_score"],
            "older_sentiment": older_sentiment["overall_sentiment"],
            "older_score": older_sentiment["overall_score"],
            "score_change": round(score_change, 2),
            "shift_detected": abs(score_change) > 30,
            "shift_direction": "POSITIVE" if score_change > 30 else "NEGATIVE" if score_change < -30 else "NONE",
        }
    
    def extract_key_events(
        self,
        news_list: List[Dict],
    ) -> List[Dict[str, Any]]:
        """
        Extract key events from news.
        
        Args:
            news_list: List of news articles
        
        Returns:
            List of key events
        """
        events = []
        
        event_keywords = {
            "earnings": ["quarterly results", "q1", "q2", "q3", "q4", "earnings", "profit"],
            "acquisition": ["acquire", "acquisition", "merger", "takeover", "buyout"],
            "expansion": ["expansion", "new plant", "new facility", "capacity"],
            "management": ["ceo", "cfo", "resign", "appoint", "management change"],
            "rating": ["upgrade", "downgrade", "rating", "target price"],
            "regulatory": ["sebi", "rbi", "regulatory", "compliance", "penalty"],
            "dividend": ["dividend", "bonus", "split", "buyback"],
        }
        
        for news in news_list:
            text = f"{news.get('title', '')} {news.get('summary', '')}".lower()
            
            for event_type, keywords in event_keywords.items():
                if any(kw in text for kw in keywords):
                    events.append({
                        "type": event_type,
                        "headline": news.get("title", ""),
                        "date": news.get("published_date", ""),
                        "source": news.get("source", ""),
                        "impact": self._estimate_event_impact(event_type, text),
                    })
                    break
        
        return events[:10]  # Top 10 events
    
    def _estimate_event_impact(self, event_type: str, text: str) -> str:
        """
        Estimate impact of an event.
        
        Args:
            event_type: Type of event
            text: Event text
        
        Returns:
            Impact level
        """
        high_impact = ["acquisition", "management", "earnings"]
        
        if event_type in high_impact:
            return "HIGH"
        
        # Check for magnitude indicators
        if any(word in text for word in ["major", "significant", "record"]):
            return "HIGH"
        
        return "MEDIUM"
    
    def generate_sentiment_summary(
        self,
        symbol: str,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive sentiment summary.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Sentiment summary dictionary
        """
        stock_sentiment = self.get_stock_sentiment(symbol)
        shift = self.detect_sentiment_shift(symbol)
        
        # Get news for event extraction
        news = self.news_fetcher.get_stock_news(symbol, days=7)
        events = self.extract_key_events(news)
        
        # Calculate sentiment score
        score = 50 + stock_sentiment["overall_score"] / 2
        score = max(0, min(100, score))
        
        # Generate insights
        bullish_signals = []
        bearish_signals = []
        
        if stock_sentiment["overall_sentiment"] == "POSITIVE":
            bullish_signals.append(f"Positive news sentiment ({stock_sentiment['positive_count']} positive articles)")
        elif stock_sentiment["overall_sentiment"] == "NEGATIVE":
            bearish_signals.append(f"Negative news sentiment ({stock_sentiment['negative_count']} negative articles)")
        
        if shift["shift_direction"] == "POSITIVE":
            bullish_signals.append("Sentiment improving vs. last week")
        elif shift["shift_direction"] == "NEGATIVE":
            bearish_signals.append("Sentiment deteriorating vs. last week")
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "overall_sentiment": stock_sentiment["overall_sentiment"],
            "sentiment_score": round(score, 0),
            "news_sentiment": stock_sentiment["overall_score"],
            "sentiment_trend": shift.get("shift_direction", "NONE"),
            "confidence": stock_sentiment["confidence"],
            "key_events": events,
            "bullish_signals": bullish_signals,
            "bearish_signals": bearish_signals,
            "news_count": stock_sentiment["news_count"],
            "top_headlines": stock_sentiment.get("top_headlines", []),
        }


# Convenience function
def analyze_sentiment(symbol: str) -> Dict:
    """Analyze sentiment for a symbol."""
    analyzer = SentimentAnalyzer()
    return analyzer.generate_sentiment_summary(symbol)
