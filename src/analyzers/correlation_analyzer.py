"""
Correlation Analysis Module.

Analyzes:
- Stock correlations with indices
- Sector correlations
- Cross-stock correlations
- Global market correlations
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_fetchers.price_fetcher import PriceFetcher
from src.utils.logger import LoggerMixin


class CorrelationAnalyzer(LoggerMixin):
    """
    Performs correlation analysis between stocks and indices.
    """
    
    def __init__(self):
        """Initialize the correlation analyzer."""
        self.price_fetcher = PriceFetcher()
    
    def calculate_correlation(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        column: str = "close",
        period: int = 30,
    ) -> float:
        """
        Calculate correlation between two price series.
        
        Args:
            df1: First dataframe
            df2: Second dataframe
            column: Column to use
            period: Rolling period
        
        Returns:
            Correlation coefficient
        """
        if df1.empty or df2.empty:
            return 0.0
        
        # Align dates
        df1 = df1.set_index("date") if "date" in df1.columns else df1
        df2 = df2.set_index("date") if "date" in df2.columns else df2
        
        # Get returns
        returns1 = df1[column].pct_change().dropna()
        returns2 = df2[column].pct_change().dropna()
        
        # Align and calculate correlation
        aligned = pd.concat([returns1, returns2], axis=1, join="inner")
        aligned.columns = ["s1", "s2"]
        
        if len(aligned) < 10:
            return 0.0
        
        return aligned["s1"].corr(aligned["s2"])
    
    def get_index_correlation(
        self,
        symbol: str,
        index: str = "NIFTY 50",
        period: str = "1y",
    ) -> Dict[str, Any]:
        """
        Calculate correlation with an index.
        
        Args:
            symbol: Stock symbol
            index: Index name
            period: Data period
        
        Returns:
            Correlation analysis
        """
        # Fetch data
        stock_df = self.price_fetcher.get_historical_data(symbol, period=period)
        index_df = self.price_fetcher.get_index_data(index, period=period)
        
        if stock_df.empty or index_df.empty:
            return {"error": "Could not fetch data"}
        
        # Calculate correlations for different periods
        correlations = {}
        for days in [30, 60, 90, 180, 365]:
            if len(stock_df) >= days and len(index_df) >= days:
                stock_recent = stock_df.tail(days)
                index_recent = index_df.tail(days)
                corr = self.calculate_correlation(stock_recent, index_recent)
                correlations[f"{days}d"] = round(corr, 3)
        
        # Get beta
        beta = self._calculate_beta(stock_df, index_df)
        
        # Interpret
        latest_corr = correlations.get("30d", 0)
        if latest_corr > 0.8:
            interpretation = "Highly correlated - moves closely with index"
        elif latest_corr > 0.5:
            interpretation = "Moderately correlated with index"
        elif latest_corr > 0.2:
            interpretation = "Weakly correlated with index"
        elif latest_corr > -0.2:
            interpretation = "Uncorrelated - independent mover"
        else:
            interpretation = "Negatively correlated - moves against index"
        
        return {
            "symbol": symbol,
            "index": index,
            "correlations": correlations,
            "beta": round(beta, 2),
            "interpretation": interpretation,
        }
    
    def _calculate_beta(
        self,
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame,
    ) -> float:
        """
        Calculate beta relative to index.
        
        Args:
            stock_df: Stock dataframe
            index_df: Index dataframe
        
        Returns:
            Beta value
        """
        if stock_df.empty or index_df.empty:
            return 1.0
        
        # Align and get returns
        stock_df = stock_df.set_index("date") if "date" in stock_df.columns else stock_df
        index_df = index_df.set_index("date") if "date" in index_df.columns else index_df
        
        stock_returns = stock_df["close"].pct_change().dropna()
        index_returns = index_df["close"].pct_change().dropna()
        
        aligned = pd.concat([stock_returns, index_returns], axis=1, join="inner")
        aligned.columns = ["stock", "index"]
        
        if len(aligned) < 20:
            return 1.0
        
        # Beta = Cov(stock, index) / Var(index)
        covariance = aligned["stock"].cov(aligned["index"])
        variance = aligned["index"].var()
        
        if variance == 0:
            return 1.0
        
        return covariance / variance
    
    def get_peer_correlations(
        self,
        symbol: str,
        peers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate correlations with peer stocks.
        
        Args:
            symbol: Target stock symbol
            peers: List of peer symbols
        
        Returns:
            Peer correlation analysis
        """
        from config.settings import SECTORS
        
        # Find peers from same sector if not provided
        if peers is None:
            for sector, stocks in SECTORS.items():
                if symbol in stocks:
                    peers = [s for s in stocks if s != symbol][:5]
                    break
        
        if not peers:
            return {"error": "No peers found"}
        
        # Fetch target stock data
        target_df = self.price_fetcher.get_historical_data(symbol)
        
        if target_df.empty:
            return {"error": "Could not fetch target stock data"}
        
        correlations = []
        for peer in peers:
            peer_df = self.price_fetcher.get_historical_data(peer)
            
            if not peer_df.empty:
                corr = self.calculate_correlation(target_df, peer_df)
                correlations.append({
                    "symbol": peer,
                    "correlation": round(corr, 3),
                })
        
        # Sort by correlation
        correlations.sort(key=lambda x: x["correlation"], reverse=True)
        
        return {
            "symbol": symbol,
            "peer_correlations": correlations,
            "most_correlated": correlations[0] if correlations else None,
            "least_correlated": correlations[-1] if correlations else None,
        }
    
    def get_sector_correlations(self) -> Dict[str, Any]:
        """
        Calculate correlations between sectors.
        
        Returns:
            Sector correlation matrix
        """
        from config.settings import SECTORS
        
        # Get representative stock for each sector
        sector_data = {}
        
        for sector, stocks in SECTORS.items():
            if stocks:
                # Use first stock as sector proxy
                df = self.price_fetcher.get_historical_data(stocks[0])
                if not df.empty:
                    sector_data[sector] = df
        
        if len(sector_data) < 2:
            return {"error": "Insufficient sector data"}
        
        # Calculate correlation matrix
        correlations = {}
        sectors = list(sector_data.keys())
        
        for i, sector1 in enumerate(sectors):
            correlations[sector1] = {}
            for sector2 in sectors:
                if sector1 == sector2:
                    correlations[sector1][sector2] = 1.0
                else:
                    corr = self.calculate_correlation(
                        sector_data[sector1],
                        sector_data[sector2]
                    )
                    correlations[sector1][sector2] = round(corr, 3)
        
        return {
            "correlation_matrix": correlations,
            "sectors": sectors,
        }
    
    def analyze_diversification(
        self,
        portfolio: List[str],
    ) -> Dict[str, Any]:
        """
        Analyze portfolio diversification based on correlations.
        
        Args:
            portfolio: List of stock symbols
        
        Returns:
            Diversification analysis
        """
        if len(portfolio) < 2:
            return {
                "portfolio": portfolio,
                "diversification_score": 100,
                "message": "Single stock - no diversification analysis",
            }
        
        # Fetch all stock data
        stock_data = {}
        for symbol in portfolio:
            df = self.price_fetcher.get_historical_data(symbol)
            if not df.empty:
                stock_data[symbol] = df
        
        if len(stock_data) < 2:
            return {"error": "Insufficient data for correlation analysis"}
        
        # Calculate all pairwise correlations
        symbols = list(stock_data.keys())
        correlations = []
        
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                corr = self.calculate_correlation(stock_data[sym1], stock_data[sym2])
                correlations.append({
                    "pair": f"{sym1}-{sym2}",
                    "correlation": round(corr, 3),
                })
        
        # Calculate average correlation
        avg_correlation = sum(c["correlation"] for c in correlations) / len(correlations)
        
        # High correlations (diversification risk)
        high_corr_pairs = [c for c in correlations if c["correlation"] > 0.7]
        
        # Diversification score (lower avg correlation = better diversification)
        div_score = max(0, 100 - (avg_correlation * 100))
        
        return {
            "portfolio": portfolio,
            "pairwise_correlations": correlations,
            "average_correlation": round(avg_correlation, 3),
            "diversification_score": round(div_score, 1),
            "high_correlation_pairs": high_corr_pairs,
            "recommendation": self._get_diversification_recommendation(
                avg_correlation, high_corr_pairs
            ),
        }
    
    def _get_diversification_recommendation(
        self,
        avg_corr: float,
        high_corr_pairs: List[Dict],
    ) -> str:
        """Get diversification recommendation."""
        if avg_corr < 0.3:
            return "Excellent diversification - low average correlation"
        elif avg_corr < 0.5:
            return "Good diversification - moderate correlation"
        elif avg_corr < 0.7:
            base = "Moderate diversification - some concentration risk"
            if high_corr_pairs:
                pairs = [p["pair"] for p in high_corr_pairs[:2]]
                base += f". Consider: {', '.join(pairs)} are highly correlated"
            return base
        else:
            return "Poor diversification - high correlation between holdings"


# Convenience function
def analyze_correlations(symbol: str) -> Dict:
    """Analyze correlations for a symbol."""
    analyzer = CorrelationAnalyzer()
    return analyzer.get_index_correlation(symbol)
