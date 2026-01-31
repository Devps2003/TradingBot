"""
Smart Portfolio Manager with Full Context.

Tracks:
- All holdings with live P&L
- Trade history
- Win/loss statistics
- AI-powered recommendations based on YOUR portfolio
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_fetchers.price_fetcher import PriceFetcher
from src.utils.helpers import load_json, save_json


class SmartPortfolio:
    """
    Intelligent portfolio manager that tracks everything and provides context.
    """
    
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.portfolio_file = self.data_dir / "portfolio.json"
        self.history_file = self.data_dir / "trade_history.json"
        self.daily_file = self.data_dir / "daily_snapshots.json"
        
        self.price_fetcher = PriceFetcher()
        
        # Load data
        self.portfolio = self._load_portfolio()
        self.history = self._load_history()
        self.daily_snapshots = self._load_daily_snapshots()
    
    def _load_portfolio(self) -> Dict:
        """Load portfolio data."""
        default = {
            "holdings": [],
            "cash": 100000,
            "total_capital": 100000,
            "last_updated": datetime.now().isoformat(),
        }
        
        if self.portfolio_file.exists():
            data = load_json(self.portfolio_file)
            return {**default, **data}
        return default
    
    def _load_history(self) -> List[Dict]:
        """Load trade history."""
        if self.history_file.exists():
            data = load_json(self.history_file)
            # Handle both formats: list or dict with "trades" key
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return data.get("trades", [])
        return []
    
    def _load_daily_snapshots(self) -> List[Dict]:
        """Load daily portfolio snapshots."""
        if self.daily_file.exists():
            data = load_json(self.daily_file)
            # Handle both formats
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return data.get("snapshots", [])
        return []
    
    def _save_portfolio(self):
        """Save portfolio to file."""
        self.portfolio["last_updated"] = datetime.now().isoformat()
        save_json(self.portfolio, self.portfolio_file)
    
    def _save_history(self):
        """Save trade history to file."""
        save_json({"trades": self.history}, self.history_file)
    
    def _save_daily_snapshots(self):
        """Save daily snapshots."""
        save_json({"snapshots": self.daily_snapshots}, self.daily_file)
    
    def buy(
        self,
        symbol: str,
        quantity: int,
        price: float,
        stop_loss: Optional[float] = None,
        target: Optional[float] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Record a buy transaction.
        
        Returns:
            Transaction result with portfolio context
        """
        symbol = symbol.upper()
        total_cost = quantity * price
        
        # Check cash
        if total_cost > self.portfolio["cash"]:
            return {"success": False, "error": "Insufficient cash"}
        
        # Find existing holding
        holding = None
        for h in self.portfolio["holdings"]:
            if h["symbol"] == symbol:
                holding = h
                break
        
        if holding:
            # Average up/down
            old_qty = holding["quantity"]
            old_avg = holding["avg_price"]
            new_qty = old_qty + quantity
            new_avg = (old_qty * old_avg + quantity * price) / new_qty
            
            holding["quantity"] = new_qty
            holding["avg_price"] = round(new_avg, 2)
            holding["last_buy_price"] = price
            holding["last_buy_date"] = datetime.now().isoformat()
            if stop_loss:
                holding["stop_loss"] = stop_loss
            if target:
                holding["target"] = target
        else:
            # New holding
            self.portfolio["holdings"].append({
                "symbol": symbol,
                "quantity": quantity,
                "avg_price": price,
                "buy_date": datetime.now().isoformat(),
                "last_buy_price": price,
                "last_buy_date": datetime.now().isoformat(),
                "stop_loss": stop_loss,
                "target": target,
                "notes": notes,
            })
        
        # Update cash
        self.portfolio["cash"] -= total_cost
        
        # Record trade
        trade = {
            "type": "BUY",
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "total": total_cost,
            "stop_loss": stop_loss,
            "target": target,
            "date": datetime.now().isoformat(),
            "notes": notes,
        }
        self.history.append(trade)
        
        self._save_portfolio()
        self._save_history()
        
        return {
            "success": True,
            "trade": trade,
            "message": f"Bought {quantity} {symbol} @ ₹{price:.2f}",
            "remaining_cash": self.portfolio["cash"],
        }
    
    def sell(
        self,
        symbol: str,
        quantity: int,
        price: float,
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Record a sell transaction.
        
        Returns:
            Transaction result with P&L
        """
        symbol = symbol.upper()
        
        # Find holding
        holding = None
        for h in self.portfolio["holdings"]:
            if h["symbol"] == symbol:
                holding = h
                break
        
        if not holding:
            return {"success": False, "error": f"No holding found for {symbol}"}
        
        if quantity > holding["quantity"]:
            return {"success": False, "error": f"Only {holding['quantity']} shares available"}
        
        # Calculate P&L
        avg_price = holding["avg_price"]
        total_sell = quantity * price
        total_cost = quantity * avg_price
        pnl = total_sell - total_cost
        pnl_pct = (price / avg_price - 1) * 100
        
        # Update holding
        holding["quantity"] -= quantity
        
        if holding["quantity"] == 0:
            self.portfolio["holdings"].remove(holding)
        
        # Update cash
        self.portfolio["cash"] += total_sell
        
        # Determine if win or loss
        result = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"
        
        # Record trade
        trade = {
            "type": "SELL",
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "avg_buy_price": avg_price,
            "total": total_sell,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "result": result,
            "date": datetime.now().isoformat(),
            "holding_days": self._calculate_holding_days(holding),
            "notes": notes,
        }
        self.history.append(trade)
        
        self._save_portfolio()
        self._save_history()
        
        return {
            "success": True,
            "trade": trade,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "result": result,
            "message": f"Sold {quantity} {symbol} @ ₹{price:.2f} | P&L: ₹{pnl:,.0f} ({pnl_pct:+.2f}%)",
        }
    
    def _calculate_holding_days(self, holding: Dict) -> int:
        """Calculate how long a position was held."""
        try:
            buy_date = datetime.fromisoformat(holding.get("buy_date", holding.get("last_buy_date", "")))
            return (datetime.now() - buy_date).days
        except:
            return 0
    
    def get_live_portfolio(self) -> Dict[str, Any]:
        """
        Get portfolio with live prices and P&L.
        """
        holdings_live = []
        total_invested = 0
        total_current = 0
        total_pnl = 0
        
        for h in self.portfolio["holdings"]:
            symbol = h["symbol"]
            quantity = h["quantity"]
            avg_price = h["avg_price"]
            
            # Get live price
            live = self.price_fetcher.get_live_price(symbol)
            current_price = live.get("price") or avg_price
            
            invested = quantity * avg_price
            current = quantity * current_price
            pnl = current - invested
            pnl_pct = (current_price / avg_price - 1) * 100 if avg_price > 0 else 0
            
            # Check stop/target
            stop_loss = h.get("stop_loss")
            target = h.get("target")
            
            alerts = []
            if stop_loss and current_price <= stop_loss:
                alerts.append("⚠️ STOP LOSS HIT")
            elif stop_loss and current_price <= stop_loss * 1.02:
                alerts.append("⚠️ Near stop loss")
            
            if target and current_price >= target:
                alerts.append("🎯 TARGET HIT")
            elif target and current_price >= target * 0.98:
                alerts.append("🎯 Near target")
            
            if pnl_pct >= 10:
                alerts.append("💰 Consider booking profits")
            elif pnl_pct <= -7:
                alerts.append("📉 Review position")
            
            holdings_live.append({
                "symbol": symbol,
                "quantity": quantity,
                "avg_price": avg_price,
                "current_price": round(current_price, 2),
                "invested": round(invested, 2),
                "current_value": round(current, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "stop_loss": stop_loss,
                "target": target,
                "buy_date": h.get("buy_date") or h.get("last_buy_date"),
                "holding_days": self._calculate_holding_days(h),
                "alerts": alerts,
                "change_today": live.get("change_percent", 0),
            })
            
            total_invested += invested
            total_current += current
            total_pnl += pnl
        
        # Calculate stats
        total_pnl_pct = (total_current / total_invested - 1) * 100 if total_invested > 0 else 0
        
        return {
            "holdings": sorted(holdings_live, key=lambda x: x["pnl"], reverse=True),
            "num_holdings": len(holdings_live),
            "total_invested": round(total_invested, 2),
            "total_current": round(total_current, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "cash": self.portfolio["cash"],
            "portfolio_value": round(total_current + self.portfolio["cash"], 2),
            "total_capital": self.portfolio["total_capital"],
            "last_updated": datetime.now().isoformat(),
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get trading performance statistics.
        """
        sells = [t for t in self.history if t.get("type") == "SELL"]
        
        if not sells:
            return {
                "total_trades": 0,
                "message": "No completed trades yet",
            }
        
        wins = [t for t in sells if t.get("result") == "WIN"]
        losses = [t for t in sells if t.get("result") == "LOSS"]
        
        total_profit = sum(t.get("pnl", 0) for t in wins)
        total_loss = sum(t.get("pnl", 0) for t in losses)
        net_pnl = total_profit + total_loss
        
        win_rate = len(wins) / len(sells) * 100 if sells else 0
        avg_win = total_profit / len(wins) if wins else 0
        avg_loss = total_loss / len(losses) if losses else 0
        
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        # Best and worst trades
        best_trade = max(sells, key=lambda x: x.get("pnl", 0)) if sells else None
        worst_trade = min(sells, key=lambda x: x.get("pnl", 0)) if sells else None
        
        # Average holding period
        avg_holding = sum(t.get("holding_days", 0) for t in sells) / len(sells) if sells else 0
        
        return {
            "total_trades": len(sells),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 1),
            "total_profit": round(total_profit, 2),
            "total_loss": round(total_loss, 2),
            "net_pnl": round(net_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "∞",
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "avg_holding_days": round(avg_holding, 1),
        }
    
    def get_context_for_ai(self) -> Dict[str, Any]:
        """
        Get full portfolio context for AI to use in decisions.
        """
        portfolio = self.get_live_portfolio()
        stats = self.get_performance_stats()
        
        # Recent trades
        recent_trades = sorted(
            self.history,
            key=lambda x: x.get("date", ""),
            reverse=True
        )[:10]
        
        # Holdings summary for AI
        holdings_summary = []
        for h in portfolio["holdings"]:
            holdings_summary.append({
                "symbol": h["symbol"],
                "quantity": h["quantity"],
                "avg_price": h["avg_price"],
                "current_price": h["current_price"],
                "pnl_pct": h["pnl_pct"],
                "holding_days": h["holding_days"],
                "alerts": h["alerts"],
            })
        
        return {
            "portfolio_summary": {
                "total_value": portfolio["portfolio_value"],
                "invested": portfolio["total_invested"],
                "pnl": portfolio["total_pnl"],
                "pnl_pct": portfolio["total_pnl_pct"],
                "cash_available": portfolio["cash"],
                "num_holdings": portfolio["num_holdings"],
            },
            "holdings": holdings_summary,
            "performance": {
                "total_trades": stats.get("total_trades", 0),
                "win_rate": stats.get("win_rate", 0),
                "net_pnl": stats.get("net_pnl", 0),
                "avg_holding_days": stats.get("avg_holding_days", 0),
            },
            "recent_trades": recent_trades[:5],
            "alerts": [a for h in portfolio["holdings"] for a in h.get("alerts", [])],
        }
    
    def get_signals_for_holdings(self) -> List[Dict[str, Any]]:
        """
        Get trading signals for current holdings.
        """
        from src.signals.signal_generator import SignalGenerator
        from src.analyzers.advanced_analyzer import get_advanced_analysis
        
        signal_gen = SignalGenerator()
        signals = []
        
        for h in self.portfolio["holdings"]:
            symbol = h["symbol"]
            
            try:
                # Get signal
                signal = signal_gen.generate_signal(symbol)
                advanced = get_advanced_analysis(symbol)
                
                # Get live price
                live = self.price_fetcher.get_live_price(symbol)
                current_price = live.get("price") or h["avg_price"]
                
                pnl_pct = (current_price / h["avg_price"] - 1) * 100
                
                # Determine action
                sig = signal.get("signal", "HOLD")
                action = "HOLD"
                reason = ""
                
                if sig in ["STRONG_SELL", "SELL"]:
                    action = "SELL"
                    reason = "Technical weakness"
                elif pnl_pct >= 15:
                    action = "BOOK_PROFIT"
                    reason = f"Up {pnl_pct:.1f}% - lock in gains"
                elif pnl_pct <= -7 and sig != "STRONG_BUY":
                    action = "EXIT"
                    reason = f"Down {pnl_pct:.1f}% - cut losses"
                elif h.get("stop_loss") and current_price <= h["stop_loss"]:
                    action = "EXIT"
                    reason = "Stop loss hit"
                elif h.get("target") and current_price >= h["target"]:
                    action = "BOOK_PROFIT"
                    reason = "Target achieved"
                elif sig in ["STRONG_BUY", "BUY"] and pnl_pct > 0:
                    action = "ADD"
                    reason = "Trend continues, consider adding"
                
                signals.append({
                    "symbol": symbol,
                    "quantity": h["quantity"],
                    "avg_price": h["avg_price"],
                    "current_price": current_price,
                    "pnl_pct": round(pnl_pct, 2),
                    "signal": sig,
                    "action": action,
                    "reason": reason,
                    "confidence": signal.get("confidence", 50),
                    "stop_loss": h.get("stop_loss"),
                    "target": h.get("target"),
                })
            except Exception as e:
                signals.append({
                    "symbol": symbol,
                    "error": str(e),
                })
        
        return signals
    
    def save_daily_snapshot(self):
        """Save today's portfolio snapshot."""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Check if already saved today
        for snap in self.daily_snapshots:
            if snap.get("date") == today:
                return  # Already saved
        
        portfolio = self.get_live_portfolio()
        
        snapshot = {
            "date": today,
            "portfolio_value": portfolio["portfolio_value"],
            "invested": portfolio["total_invested"],
            "pnl": portfolio["total_pnl"],
            "pnl_pct": portfolio["total_pnl_pct"],
            "cash": portfolio["cash"],
            "num_holdings": portfolio["num_holdings"],
        }
        
        self.daily_snapshots.append(snapshot)
        self._save_daily_snapshots()
    
    def get_portfolio_chart_data(self) -> List[Dict]:
        """Get portfolio value over time for charting."""
        return sorted(self.daily_snapshots, key=lambda x: x.get("date", ""))


def get_smart_portfolio() -> SmartPortfolio:
    """Get smart portfolio instance."""
    return SmartPortfolio()
