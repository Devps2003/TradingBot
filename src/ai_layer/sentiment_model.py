"""
Sentiment Model Module.

Wrapper for sentiment analysis models:
- FinBERT (financial sentiment)
- VADER (rule-based)
- Custom trained models
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import LoggerMixin


class SentimentModel(LoggerMixin):
    """
    Sentiment analysis model wrapper.
    """
    
    def __init__(self, model_type: str = "auto"):
        """
        Initialize the sentiment model.
        
        Args:
            model_type: Model to use (finbert, vader, auto)
        """
        self.model_type = model_type
        self.finbert = None
        self.vader = None
        
        # Initialize VADER (always available if installed)
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
            self.logger.info("VADER sentiment analyzer initialized")
        
        # Initialize FinBERT if available and requested
        if model_type in ["finbert", "auto"] and TRANSFORMERS_AVAILABLE:
            try:
                self.finbert = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    truncation=True,
                    max_length=512,
                )
                self.logger.info("FinBERT model loaded")
                self.model_type = "finbert"
            except Exception as e:
                self.logger.warning(f"Could not load FinBERT: {e}")
                if self.vader:
                    self.model_type = "vader"
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Sentiment analysis result
        """
        if not text or not text.strip():
            return {
                "label": "NEUTRAL",
                "score": 0,
                "confidence": 0,
                "model": None,
            }
        
        # Use FinBERT if available
        if self.finbert and self.model_type == "finbert":
            return self._analyze_finbert(text)
        
        # Fall back to VADER
        if self.vader:
            return self._analyze_vader(text)
        
        # Last resort: keyword-based
        return self._analyze_keywords(text)
    
    def _analyze_finbert(self, text: str) -> Dict[str, Any]:
        """
        Analyze using FinBERT.
        
        Args:
            text: Text to analyze
        
        Returns:
            Sentiment result
        """
        try:
            # Truncate text if too long
            text = text[:500]
            
            result = self.finbert(text)[0]
            label = result["label"].upper()
            score = result["score"]
            
            # Normalize score to -100 to 100 scale
            if label == "POSITIVE":
                normalized_score = score * 100
            elif label == "NEGATIVE":
                normalized_score = -score * 100
            else:
                normalized_score = 0
            
            return {
                "label": label,
                "score": round(normalized_score, 2),
                "confidence": round(score * 100, 2),
                "model": "finbert",
            }
        except Exception as e:
            self.logger.warning(f"FinBERT error: {e}")
            return self._analyze_vader(text)
    
    def _analyze_vader(self, text: str) -> Dict[str, Any]:
        """
        Analyze using VADER.
        
        Args:
            text: Text to analyze
        
        Returns:
            Sentiment result
        """
        scores = self.vader.polarity_scores(text)
        compound = scores["compound"]
        
        if compound >= 0.05:
            label = "POSITIVE"
        elif compound <= -0.05:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        
        return {
            "label": label,
            "score": round(compound * 100, 2),
            "confidence": round(abs(compound) * 100, 2),
            "model": "vader",
            "details": {
                "positive": scores["pos"],
                "negative": scores["neg"],
                "neutral": scores["neu"],
            },
        }
    
    def _analyze_keywords(self, text: str) -> Dict[str, Any]:
        """
        Simple keyword-based sentiment analysis.
        
        Args:
            text: Text to analyze
        
        Returns:
            Sentiment result
        """
        text_lower = text.lower()
        
        positive_words = {
            "bullish", "surge", "rally", "gain", "profit", "growth",
            "strong", "upgrade", "outperform", "buy", "beat", "exceed",
            "record", "high", "breakout", "momentum", "opportunity"
        }
        
        negative_words = {
            "bearish", "fall", "drop", "decline", "loss", "weak",
            "downgrade", "underperform", "sell", "miss", "concern",
            "risk", "low", "crash", "plunge", "warning", "fear"
        }
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        
        if total == 0:
            return {
                "label": "NEUTRAL",
                "score": 0,
                "confidence": 20,
                "model": "keywords",
            }
        
        score = (pos_count - neg_count) / total * 100
        
        if score > 30:
            label = "POSITIVE"
        elif score < -30:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        
        return {
            "label": label,
            "score": round(score, 2),
            "confidence": min(50, total * 10),
            "model": "keywords",
        }
    
    def analyze_batch(
        self,
        texts: List[str],
    ) -> Dict[str, Any]:
        """
        Analyze multiple texts.
        
        Args:
            texts: List of texts
        
        Returns:
            Aggregated sentiment result
        """
        if not texts:
            return {
                "overall_label": "NEUTRAL",
                "overall_score": 0,
                "count": 0,
            }
        
        results = [self.analyze(text) for text in texts]
        
        # Aggregate
        scores = [r["score"] for r in results]
        avg_score = sum(scores) / len(scores)
        
        positive_count = sum(1 for r in results if r["label"] == "POSITIVE")
        negative_count = sum(1 for r in results if r["label"] == "NEGATIVE")
        neutral_count = sum(1 for r in results if r["label"] == "NEUTRAL")
        
        if avg_score > 20:
            overall_label = "POSITIVE"
        elif avg_score < -20:
            overall_label = "NEGATIVE"
        else:
            overall_label = "NEUTRAL"
        
        return {
            "overall_label": overall_label,
            "overall_score": round(avg_score, 2),
            "count": len(texts),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "distribution": {
                "positive": positive_count / len(texts) * 100,
                "negative": negative_count / len(texts) * 100,
                "neutral": neutral_count / len(texts) * 100,
            },
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Model information
        """
        return {
            "active_model": self.model_type,
            "finbert_available": self.finbert is not None,
            "vader_available": self.vader is not None,
            "transformers_installed": TRANSFORMERS_AVAILABLE,
        }


# Convenience functions
def analyze_sentiment(text: str) -> Dict:
    """Analyze sentiment of text."""
    model = SentimentModel()
    return model.analyze(text)


def analyze_headlines(headlines: List[str]) -> Dict:
    """Analyze sentiment of multiple headlines."""
    model = SentimentModel()
    return model.analyze_batch(headlines)
