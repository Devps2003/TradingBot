"""
Tests for Price Fetcher module.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_fetchers.price_fetcher import PriceFetcher


class TestPriceFetcher:
    """Test cases for PriceFetcher."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fetcher = PriceFetcher()
    
    def test_get_live_price(self):
        """Test getting live price for a stock."""
        result = self.fetcher.get_live_price("RELIANCE")
        
        assert result is not None
        assert "symbol" in result
        assert result["symbol"] == "RELIANCE"
    
    def test_get_historical_data(self):
        """Test getting historical data."""
        df = self.fetcher.get_historical_data("RELIANCE", period="1mo")
        
        assert df is not None
        if not df.empty:
            assert "close" in df.columns
            assert "volume" in df.columns
    
    def test_get_multiple_stocks(self):
        """Test getting data for multiple stocks."""
        symbols = ["RELIANCE", "TCS"]
        result = self.fetcher.get_multiple_stocks(symbols, period="5d")
        
        assert isinstance(result, dict)
    
    def test_get_index_data(self):
        """Test getting index data."""
        df = self.fetcher.get_index_data("NIFTY 50", period="1mo")
        
        assert df is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
