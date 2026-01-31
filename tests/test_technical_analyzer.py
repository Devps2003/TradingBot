"""
Tests for Technical Analyzer module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzers.technical_analyzer import TechnicalAnalyzer


class TestTechnicalAnalyzer:
    """Test cases for TechnicalAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TechnicalAnalyzer()
        
        # Create sample OHLCV data
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        
        prices = 100 + np.cumsum(np.random.randn(100) * 2)
        
        self.sample_df = pd.DataFrame({
            "date": dates,
            "open": prices - np.random.rand(100),
            "high": prices + np.random.rand(100) * 2,
            "low": prices - np.random.rand(100) * 2,
            "close": prices,
            "volume": np.random.randint(100000, 1000000, 100),
        })
    
    def test_calculate_all_indicators(self):
        """Test calculating all indicators."""
        result = self.analyzer.calculate_all_indicators(self.sample_df)
        
        assert result is not None
        assert not result.empty
        # Check that some indicators were added
        assert len(result.columns) > len(self.sample_df.columns)
    
    def test_get_indicator_signals(self):
        """Test getting indicator signals."""
        df_with_indicators = self.analyzer.calculate_all_indicators(self.sample_df)
        signals = self.analyzer.get_indicator_signals(df_with_indicators)
        
        assert isinstance(signals, dict)
    
    def test_identify_support_resistance(self):
        """Test support/resistance identification."""
        result = self.analyzer.identify_support_resistance(self.sample_df)
        
        assert "support" in result
        assert "resistance" in result
        assert "current_price" in result
    
    def test_get_trend_strength(self):
        """Test trend strength calculation."""
        df_with_indicators = self.analyzer.calculate_all_indicators(self.sample_df)
        result = self.analyzer.get_trend_strength(df_with_indicators)
        
        assert "trend" in result
        assert "strength" in result
    
    def test_generate_technical_summary(self):
        """Test generating technical summary."""
        result = self.analyzer.generate_technical_summary(self.sample_df, "TEST")
        
        assert "symbol" in result
        assert "technical_score" in result
        assert "trend" in result
        assert "key_levels" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
