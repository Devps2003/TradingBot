"""
Trade Tracker Module.

Tracks and analyzes trades:
- Log trades
- Calculate performance metrics
- Pattern performance analysis
- Improvement suggestions
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR
from src.utils.helpers import load_json, save_json
from src.utils.logger import LoggerMixin


class TradeTracker(LoggerMixin):
    """
    Tracks and analyzes trading history.
    """
    
    def __init__(self, trades_file: Optional[str] = None):
        """
        Initialize trade tracker.
        
        Args:
            trades_file: Path to trades JSON file
        """
        self.trades_file = Path(trades_file) if trades_file else DATA_DIR / "trade_history.json"
        self.trades = self._load_trades()
    
    def _load_trades(self) -> List[Dict[str, Any]]:
        """Load trades from file."""
        if not self.trades_file.exists():
            save_json([], self.trades_file)
            return []
        
        data = load_json(self.trades_file)
        return data if isinstance(data, list) else []
    
    def _save_trades(self) -> None:
        """Save trades to file."""
        save_json(self.trades, self.trades_file)
    
    def log_trade(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        quantity: int,
        entry_date: str,
        exit_date: str,
        trade_type: str = "LONG",
        pattern: Optional[str] = None,
        signal: Optional[str] = None,
        entry_reason: str = "",
        exit_reason: str = "",
    ) -> Dict[str, Any]:
        """
        Log a completed trade.
        
        Args:
            symbol: Stock symbol
            entry_price: Entry price
            exit_price: Exit price
            quantity: Number of shares
            entry_date: Entry date
            exit_date: Exit date
            trade_type: LONG or SHORT
            pattern: Pattern that triggered trade
            signal: Signal that triggered trade
            entry_reason: Reason for entry
            exit_reason: Reason for exit
        
        Returns:
            Trade record
        """
        # Calculate P&L
        if trade_type == "LONG":
            pnl = (exit_price - entry_price) * quantity
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl = (entry_price - exit_price) * quantity
            pnl_pct = (entry_price - exit_price) / entry_price * 100
        
        # Calculate holding period
        try:
            entry_dt = datetime.fromisoformat(entry_date)
            exit_dt = datetime.fromisoformat(exit_date)
            holding_days = (exit_dt - entry_dt).days
        except Exception:
            holding_days = 0
        
        trade = {
            "id": len(self.trades) + 1,
            "symbol": symbol,
            "trade_type": trade_type,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "holding_days": holding_days,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "is_winner": pnl > 0,
            "pattern": pattern,
            "signal": signal,
            "entry_reason": entry_reason,
            "exit_reason": exit_reason,
            "logged_at": datetime.now().isoformat(),
        }
        
        self.trades.append(trade)
        self._save_trades()
        
        self.logger.info(f"Logged trade: {symbol} {pnl_pct:+.2f}%")
        
        return trade
    
    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get trade history with optional filters.
        
        Args:
            symbol: Filter by symbol
            days: Filter by recent days
        
        Returns:
            List of trades
        """
        trades = self.trades
        
        if symbol:
            trades = [t for t in trades if t["symbol"] == symbol.upper()]
        
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            trades = [
                t for t in trades
                if datetime.fromisoformat(t["exit_date"]) >= cutoff
            ]
        
        return sorted(trades, key=lambda x: x["exit_date"], reverse=True)
    
    def analyze_performance(
        self,
        days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Analyze overall trading performance.
        
        Args:
            days: Analyze only recent days
        
        Returns:
            Performance analysis
        """
        trades = self.get_trade_history(days=days)
        
        if not trades:
            return {
                "total_trades": 0,
                "message": "No trades to analyze",
            }
        
        winners = [t for t in trades if t["is_winner"]]
        losers = [t for t in trades if not t["is_winner"]]
        
        total_pnl = sum(t["pnl"] for t in trades)
        
        # Win rate
        win_rate = len(winners) / len(trades) * 100
        
        # Average win/loss
        avg_win = sum(t["pnl_pct"] for t in winners) / len(winners) if winners else 0
        avg_loss = sum(t["pnl_pct"] for t in losers) / len(losers) if losers else 0
        
        # Profit factor
        gross_profit = sum(t["pnl"] for t in winners) if winners else 0
        gross_loss = abs(sum(t["pnl"] for t in losers)) if losers else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
        
        # Expectancy
        expectancy = (win_rate / 100 * avg_win) + ((100 - win_rate) / 100 * avg_loss)
        
        # Best and worst trades
        best = max(trades, key=lambda x: x["pnl_pct"])
        worst = min(trades, key=lambda x: x["pnl_pct"])
        
        # Average holding period
        avg_holding = sum(t["holding_days"] for t in trades) / len(trades)
        
        return {
            "period": f"Last {days} days" if days else "All time",
            "total_trades": len(trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "expectancy": round(expectancy, 2),
            "best_trade": {
                "symbol": best["symbol"],
                "pnl_pct": best["pnl_pct"],
            },
            "worst_trade": {
                "symbol": worst["symbol"],
                "pnl_pct": worst["pnl_pct"],
            },
            "avg_holding_days": round(avg_holding, 1),
        }
    
    def get_pattern_performance(self) -> Dict[str, Any]:
        """
        Analyze performance by pattern.
        
        Returns:
            Pattern performance analysis
        """
        pattern_trades = {}
        
        for trade in self.trades:
            pattern = trade.get("pattern")
            if pattern:
                if pattern not in pattern_trades:
                    pattern_trades[pattern] = []
                pattern_trades[pattern].append(trade)
        
        pattern_stats = {}
        for pattern, trades in pattern_trades.items():
            winners = [t for t in trades if t["is_winner"]]
            win_rate = len(winners) / len(trades) * 100
            avg_pnl = sum(t["pnl_pct"] for t in trades) / len(trades)
            
            pattern_stats[pattern] = {
                "total_trades": len(trades),
                "win_rate": round(win_rate, 2),
                "avg_pnl_pct": round(avg_pnl, 2),
                "total_pnl": round(sum(t["pnl"] for t in trades), 2),
            }
        
        # Sort by performance
        sorted_patterns = sorted(
            pattern_stats.items(),
            key=lambda x: x[1]["avg_pnl_pct"],
            reverse=True
        )
        
        return {
            "patterns": dict(sorted_patterns),
            "best_pattern": sorted_patterns[0] if sorted_patterns else None,
            "worst_pattern": sorted_patterns[-1] if sorted_patterns else None,
        }
    
    def get_symbol_performance(self) -> Dict[str, Any]:
        """
        Analyze performance by symbol.
        
        Returns:
            Symbol performance analysis
        """
        symbol_trades = {}
        
        for trade in self.trades:
            symbol = trade["symbol"]
            if symbol not in symbol_trades:
                symbol_trades[symbol] = []
            symbol_trades[symbol].append(trade)
        
        symbol_stats = {}
        for symbol, trades in symbol_trades.items():
            winners = [t for t in trades if t["is_winner"]]
            win_rate = len(winners) / len(trades) * 100
            total_pnl = sum(t["pnl"] for t in trades)
            
            symbol_stats[symbol] = {
                "total_trades": len(trades),
                "win_rate": round(win_rate, 2),
                "total_pnl": round(total_pnl, 2),
            }
        
        # Sort by P&L
        sorted_symbols = sorted(
            symbol_stats.items(),
            key=lambda x: x[1]["total_pnl"],
            reverse=True
        )
        
        return {
            "symbols": dict(sorted_symbols),
            "most_profitable": sorted_symbols[0] if sorted_symbols else None,
            "least_profitable": sorted_symbols[-1] if sorted_symbols else None,
        }
    
    def get_improvement_suggestions(self) -> List[str]:
        """
        Get suggestions for improving trading.
        
        Returns:
            List of suggestions
        """
        suggestions = []
        
        performance = self.analyze_performance()
        
        if performance.get("total_trades", 0) < 10:
            return ["Insufficient trades for meaningful analysis"]
        
        # Win rate suggestions
        win_rate = performance.get("win_rate", 50)
        if win_rate < 40:
            suggestions.append(
                "Win rate is below 40%. Consider being more selective "
                "with entry points and waiting for stronger confirmations."
            )
        
        # Risk/Reward suggestions
        avg_win = abs(performance.get("avg_win_pct", 0))
        avg_loss = abs(performance.get("avg_loss_pct", 0))
        
        if avg_loss > avg_win:
            suggestions.append(
                "Average loss is larger than average win. "
                "Consider tightening stop losses or widening targets."
            )
        
        # Profit factor
        pf = performance.get("profit_factor", 1)
        if pf < 1:
            suggestions.append(
                "Profit factor is below 1 (losing money overall). "
                "Review your strategy and consider paper trading."
            )
        elif pf < 1.5:
            suggestions.append(
                "Profit factor is marginal. Focus on high-conviction "
                "setups and avoid lower-quality trades."
            )
        
        # Holding period
        avg_holding = performance.get("avg_holding_days", 7)
        if avg_holding < 3:
            suggestions.append(
                "Average holding period is very short. "
                "Consider giving trades more time to develop."
            )
        elif avg_holding > 20:
            suggestions.append(
                "Average holding period is quite long. "
                "Consider time-based exits if targets aren't hit."
            )
        
        # Pattern analysis
        pattern_perf = self.get_pattern_performance()
        worst_pattern = pattern_perf.get("worst_pattern")
        if worst_pattern and worst_pattern[1]["avg_pnl_pct"] < -2:
            suggestions.append(
                f"Pattern '{worst_pattern[0]}' has negative performance. "
                "Consider avoiding or improving entries for this pattern."
            )
        
        if not suggestions:
            suggestions.append(
                "Performance looks healthy! Keep maintaining discipline "
                "and following your strategy."
            )
        
        return suggestions
    
    def get_recent_trades_summary(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get summary of recent trades.
        
        Args:
            n: Number of recent trades
        
        Returns:
            List of trade summaries
        """
        recent = sorted(
            self.trades,
            key=lambda x: x.get("exit_date", ""),
            reverse=True
        )[:n]
        
        return [
            {
                "symbol": t["symbol"],
                "entry": t["entry_price"],
                "exit": t["exit_price"],
                "pnl_pct": t["pnl_pct"],
                "result": "WIN" if t["is_winner"] else "LOSS",
                "date": t["exit_date"][:10],
            }
            for t in recent
        ]


# Convenience functions
def log_trade(symbol: str, entry: float, exit: float, qty: int) -> Dict:
    """Log a trade."""
    tracker = TradeTracker()
    return tracker.log_trade(
        symbol=symbol,
        entry_price=entry,
        exit_price=exit,
        quantity=qty,
        entry_date=datetime.now().isoformat(),
        exit_date=datetime.now().isoformat(),
    )


def get_performance() -> Dict:
    """Get trading performance."""
    tracker = TradeTracker()
    return tracker.analyze_performance()
