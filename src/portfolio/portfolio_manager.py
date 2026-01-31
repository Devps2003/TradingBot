"""
Portfolio Manager Module.

Manages user's portfolio:
- Track holdings
- Calculate P&L
- Sector allocation
- Risk metrics
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR
from src.data_fetchers.price_fetcher import PriceFetcher
from src.utils.helpers import load_json, save_json
from src.utils.logger import LoggerMixin


class PortfolioManager(LoggerMixin):
    """
    Manages user's portfolio and holdings.
    """
    
    def __init__(self, portfolio_file: Optional[str] = None):
        """
        Initialize the portfolio manager.
        
        Args:
            portfolio_file: Path to portfolio JSON file
        """
        self.portfolio_file = Path(portfolio_file) if portfolio_file else DATA_DIR / "portfolio.json"
        self.price_fetcher = PriceFetcher()
        self.portfolio = self._load_portfolio()
    
    def _load_portfolio(self) -> Dict[str, Any]:
        """
        Load portfolio from file.
        
        Returns:
            Portfolio dictionary
        """
        default_portfolio = {
            "holdings": [],
            "cash": 100000,
            "total_capital": 500000,
            "last_updated": datetime.now().isoformat(),
        }
        
        if not self.portfolio_file.exists():
            save_json(default_portfolio, self.portfolio_file)
            return default_portfolio
        
        portfolio = load_json(self.portfolio_file)
        return portfolio if portfolio else default_portfolio
    
    def save_portfolio(self) -> None:
        """Save portfolio to file."""
        self.portfolio["last_updated"] = datetime.now().isoformat()
        save_json(self.portfolio, self.portfolio_file)
        self.logger.info("Portfolio saved")
    
    def add_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        stop_loss: Optional[float] = None,
        target: Optional[float] = None,
        reason: str = "",
    ) -> Dict[str, Any]:
        """
        Add a new position or add to existing position.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Buy price per share
            stop_loss: Stop loss price
            target: Target price
            reason: Reason for trade
        
        Returns:
            Updated position
        """
        symbol = symbol.upper()
        
        # Check for existing position
        existing = None
        for i, pos in enumerate(self.portfolio["holdings"]):
            if pos["symbol"] == symbol:
                existing = (i, pos)
                break
        
        if existing:
            # Average into existing position
            idx, pos = existing
            old_value = pos["quantity"] * pos["avg_price"]
            new_value = quantity * price
            total_quantity = pos["quantity"] + quantity
            avg_price = (old_value + new_value) / total_quantity
            
            self.portfolio["holdings"][idx].update({
                "quantity": total_quantity,
                "avg_price": round(avg_price, 2),
                "stop_loss": stop_loss or pos.get("stop_loss"),
                "target": target or pos.get("target"),
                "last_added": datetime.now().isoformat(),
            })
            
            position = self.portfolio["holdings"][idx]
        else:
            # New position
            position = {
                "symbol": symbol,
                "quantity": quantity,
                "avg_price": price,
                "buy_date": datetime.now().isoformat(),
                "stop_loss": stop_loss,
                "target": target,
                "buy_reason": reason,
            }
            self.portfolio["holdings"].append(position)
        
        # Deduct from cash
        cost = quantity * price
        self.portfolio["cash"] = max(0, self.portfolio["cash"] - cost)
        
        self.save_portfolio()
        
        return position
    
    def close_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        reason: str = "",
    ) -> Dict[str, Any]:
        """
        Close or reduce a position.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares to sell
            price: Sell price per share
            reason: Reason for exit
        
        Returns:
            Trade result
        """
        symbol = symbol.upper()
        
        # Find position
        for i, pos in enumerate(self.portfolio["holdings"]):
            if pos["symbol"] == symbol:
                if quantity >= pos["quantity"]:
                    # Close entire position
                    quantity = pos["quantity"]
                    self.portfolio["holdings"].pop(i)
                else:
                    # Partial close
                    self.portfolio["holdings"][i]["quantity"] -= quantity
                
                # Calculate P&L
                pnl = (price - pos["avg_price"]) * quantity
                pnl_pct = (price - pos["avg_price"]) / pos["avg_price"] * 100
                
                # Add to cash
                self.portfolio["cash"] += quantity * price
                
                self.save_portfolio()
                
                return {
                    "symbol": symbol,
                    "quantity_sold": quantity,
                    "sell_price": price,
                    "avg_buy_price": pos["avg_price"],
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "reason": reason,
                }
        
        return {"error": f"Position not found: {symbol}"}
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get portfolio summary with current values.
        
        Returns:
            Portfolio summary
        """
        holdings = []
        total_invested = 0
        total_current = 0
        
        for pos in self.portfolio["holdings"]:
            symbol = pos["symbol"]
            quantity = pos["quantity"]
            avg_price = pos["avg_price"]
            
            # Get current price
            live = self.price_fetcher.get_live_price(symbol)
            current_price = live.get("price", avg_price)
            
            invested = quantity * avg_price
            current = quantity * current_price
            pnl = current - invested
            pnl_pct = (current_price - avg_price) / avg_price * 100
            
            total_invested += invested
            total_current += current
            
            holdings.append({
                "symbol": symbol,
                "quantity": quantity,
                "avg_price": avg_price,
                "current_price": round(current_price, 2),
                "invested": round(invested, 2),
                "current_value": round(current, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "stop_loss": pos.get("stop_loss"),
                "target": pos.get("target"),
                "buy_date": pos.get("buy_date"),
            })
        
        total_pnl = total_current - total_invested
        total_pnl_pct = (total_current - total_invested) / total_invested * 100 if total_invested > 0 else 0
        
        portfolio_value = total_current + self.portfolio["cash"]
        
        return {
            "holdings": holdings,
            "num_holdings": len(holdings),
            "total_invested": round(total_invested, 2),
            "total_current": round(total_current, 2),
            "unrealized_pnl": round(total_pnl, 2),
            "unrealized_pnl_pct": round(total_pnl_pct, 2),
            "cash": round(self.portfolio["cash"], 2),
            "portfolio_value": round(portfolio_value, 2),
            "total_capital": self.portfolio["total_capital"],
            "last_updated": datetime.now().isoformat(),
        }
    
    def get_sector_allocation(self) -> Dict[str, Any]:
        """
        Get sector-wise allocation.
        
        Returns:
            Sector allocation
        """
        from config.settings import SECTORS
        
        # Create reverse mapping: symbol -> sector
        symbol_to_sector = {}
        for sector, stocks in SECTORS.items():
            for stock in stocks:
                symbol_to_sector[stock] = sector
        
        sector_values = {}
        total_value = 0
        
        for pos in self.portfolio["holdings"]:
            symbol = pos["symbol"]
            quantity = pos["quantity"]
            
            # Get current value
            live = self.price_fetcher.get_live_price(symbol)
            current_price = live.get("price", pos["avg_price"])
            value = quantity * current_price
            
            sector = symbol_to_sector.get(symbol, "OTHER")
            sector_values[sector] = sector_values.get(sector, 0) + value
            total_value += value
        
        # Calculate percentages
        allocation = {}
        for sector, value in sector_values.items():
            allocation[sector] = {
                "value": round(value, 2),
                "percentage": round(value / total_value * 100, 2) if total_value > 0 else 0,
            }
        
        return {
            "allocation": allocation,
            "total_invested_value": round(total_value, 2),
            "num_sectors": len(sector_values),
        }
    
    def get_holdings_analysis(self) -> List[Dict[str, Any]]:
        """
        Get detailed analysis for each holding.
        
        Returns:
            List of holding analyses
        """
        from src.signals.signal_generator import SignalGenerator
        
        generator = SignalGenerator()
        analyses = []
        
        for pos in self.portfolio["holdings"]:
            symbol = pos["symbol"]
            
            # Get signal
            signal = generator.generate_signal(symbol)
            
            # Check alerts
            alerts = []
            
            # Near target
            current = signal.get("entry_price", 0)
            target = pos.get("target")
            stop = pos.get("stop_loss")
            
            if target and current >= target * 0.98:
                alerts.append({
                    "type": "NEAR_TARGET",
                    "message": f"Approaching target ₹{target}",
                })
            
            if stop and current <= stop * 1.02:
                alerts.append({
                    "type": "NEAR_STOP",
                    "message": f"Near stop loss ₹{stop}",
                })
            
            # Signal change
            if signal.get("signal") in ["SELL", "STRONG_SELL"]:
                alerts.append({
                    "type": "SIGNAL_CHANGE",
                    "message": f"Signal changed to {signal['signal']}",
                })
            
            analyses.append({
                "position": pos,
                "signal": signal,
                "alerts": alerts,
            })
        
        return analyses
    
    def get_portfolio_risk(self) -> Dict[str, Any]:
        """
        Get portfolio risk metrics.
        
        Returns:
            Risk metrics
        """
        from src.signals.risk_calculator import RiskCalculator
        
        calculator = RiskCalculator()
        
        # Prepare portfolio for risk calculation
        positions = []
        for pos in self.portfolio["holdings"]:
            live = self.price_fetcher.get_live_price(pos["symbol"])
            positions.append({
                "symbol": pos["symbol"],
                "quantity": pos["quantity"],
                "avg_price": pos["avg_price"],
                "stop_loss": pos.get("stop_loss", pos["avg_price"] * 0.95),
                "current_price": live.get("price", pos["avg_price"]),
            })
        
        return calculator.calculate_portfolio_risk(
            positions,
            self.portfolio["total_capital"],
        )
    
    def update_position(
        self,
        symbol: str,
        stop_loss: Optional[float] = None,
        target: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Update position parameters.
        
        Args:
            symbol: Stock symbol
            stop_loss: New stop loss
            target: New target
        
        Returns:
            Updated position
        """
        symbol = symbol.upper()
        
        for i, pos in enumerate(self.portfolio["holdings"]):
            if pos["symbol"] == symbol:
                if stop_loss is not None:
                    self.portfolio["holdings"][i]["stop_loss"] = stop_loss
                if target is not None:
                    self.portfolio["holdings"][i]["target"] = target
                
                self.save_portfolio()
                return self.portfolio["holdings"][i]
        
        return {"error": f"Position not found: {symbol}"}
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific position.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Position or None
        """
        symbol = symbol.upper()
        
        for pos in self.portfolio["holdings"]:
            if pos["symbol"] == symbol:
                # Add current price
                live = self.price_fetcher.get_live_price(symbol)
                current = live.get("price", pos["avg_price"])
                
                return {
                    **pos,
                    "current_price": current,
                    "pnl": (current - pos["avg_price"]) * pos["quantity"],
                    "pnl_pct": (current - pos["avg_price"]) / pos["avg_price"] * 100,
                }
        
        return None


# Convenience functions
def get_portfolio() -> Dict:
    """Get portfolio summary."""
    manager = PortfolioManager()
    return manager.get_portfolio_summary()


def add_stock(symbol: str, quantity: int, price: float) -> Dict:
    """Add stock to portfolio."""
    manager = PortfolioManager()
    return manager.add_position(symbol, quantity, price)
