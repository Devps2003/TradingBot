"""
Main Trading Agent Orchestrator.

This is the brain of the system that:
- Orchestrates all analysis modules
- Generates morning briefings
- Scans for opportunities
- Provides interactive Q&A
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import NIFTY_50, DATA_DIR
from src.data_fetchers.price_fetcher import PriceFetcher
from src.data_fetchers.global_fetcher import GlobalFetcher
from src.data_fetchers.fii_dii_fetcher import FIIDIIFetcher
from src.analyzers.market_context_analyzer import MarketContextAnalyzer
from src.signals.signal_generator import SignalGenerator
from src.portfolio.portfolio_manager import PortfolioManager
from src.portfolio.trade_tracker import TradeTracker
from src.ai_layer.llm_reasoner import LLMReasoner, SimpleLLMReasoner
from src.utils.indian_market_utils import get_session_info, is_market_open
from src.utils.helpers import load_json
from src.utils.logger import LoggerMixin


class TradingAgent(LoggerMixin):
    """
    Main trading agent that orchestrates all analysis.
    """
    
    def __init__(self):
        """Initialize the trading agent."""
        self.price_fetcher = PriceFetcher()
        self.global_fetcher = GlobalFetcher()
        self.fii_dii_fetcher = FIIDIIFetcher()
        self.market_analyzer = MarketContextAnalyzer()
        self.signal_generator = SignalGenerator()
        self.portfolio_manager = PortfolioManager()
        self.trade_tracker = TradeTracker()
        
        # Try to initialize LLM
        try:
            self.llm = LLMReasoner()
            if self.llm.client is None:
                self.llm = SimpleLLMReasoner()
        except Exception:
            self.llm = SimpleLLMReasoner()
        
        # Load watchlist
        self.watchlist = self._load_watchlist()
    
    def _load_watchlist(self) -> List[str]:
        """Load watchlist from file."""
        watchlist_file = DATA_DIR / "watchlist.json"
        if watchlist_file.exists():
            data = load_json(watchlist_file)
            return data.get("symbols", [])
        return []
    
    def run_morning_briefing(self) -> Dict[str, Any]:
        """
        Generate comprehensive morning briefing.
        
        Returns:
            Morning briefing data
        """
        self.logger.info("Generating morning briefing...")
        
        # Session info
        session = get_session_info()
        
        # Global market cues
        global_cues = self.global_fetcher.get_market_cues_for_india()
        
        # FII/DII data
        fii_dii = self.fii_dii_fetcher.get_market_sentiment_indicator()
        
        # Market context
        market_context = self.market_analyzer.generate_market_context_summary()
        
        # Portfolio summary
        portfolio = self.portfolio_manager.get_portfolio_summary()
        
        # Portfolio alerts
        alerts = self._check_portfolio_alerts()
        
        # Get LLM briefing
        briefing_text = ""
        if hasattr(self.llm, "generate_daily_briefing"):
            briefing_text = self.llm.generate_daily_briefing(
                market_context,
                portfolio.get("holdings", [])
            )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "session": session,
            "global_cues": global_cues,
            "fii_dii": fii_dii,
            "market_context": market_context,
            "portfolio": portfolio,
            "alerts": alerts,
            "briefing": briefing_text,
        }
    
    def _check_portfolio_alerts(self) -> List[Dict[str, Any]]:
        """Check for portfolio alerts."""
        alerts = []
        holdings = self.portfolio_manager.portfolio.get("holdings", [])
        
        for pos in holdings:
            symbol = pos["symbol"]
            live = self.price_fetcher.get_live_price(symbol)
            current = live.get("price", pos["avg_price"])
            
            # Near target
            if pos.get("target") and current >= pos["target"] * 0.98:
                alerts.append({
                    "type": "NEAR_TARGET",
                    "symbol": symbol,
                    "message": f"{symbol} approaching target ₹{pos['target']:.2f}",
                    "priority": "HIGH",
                })
            
            # Near stop loss
            if pos.get("stop_loss") and current <= pos["stop_loss"] * 1.02:
                alerts.append({
                    "type": "NEAR_STOP",
                    "symbol": symbol,
                    "message": f"{symbol} near stop loss ₹{pos['stop_loss']:.2f}",
                    "priority": "CRITICAL",
                })
            
            # Significant P&L
            pnl_pct = (current - pos["avg_price"]) / pos["avg_price"] * 100
            if pnl_pct > 10:
                alerts.append({
                    "type": "PROFIT",
                    "symbol": symbol,
                    "message": f"{symbol} up {pnl_pct:.1f}%. Consider booking partial profits.",
                    "priority": "MEDIUM",
                })
            elif pnl_pct < -5:
                alerts.append({
                    "type": "LOSS",
                    "symbol": symbol,
                    "message": f"{symbol} down {pnl_pct:.1f}%. Review position.",
                    "priority": "HIGH",
                })
        
        return sorted(alerts, key=lambda x: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}.get(x["priority"], 3))
    
    def analyze_portfolio(self) -> Dict[str, Any]:
        """
        Perform comprehensive portfolio analysis.
        
        Returns:
            Portfolio analysis
        """
        self.logger.info("Analyzing portfolio...")
        
        summary = self.portfolio_manager.get_portfolio_summary()
        sector_allocation = self.portfolio_manager.get_sector_allocation()
        risk = self.portfolio_manager.get_portfolio_risk()
        holdings_analysis = self.portfolio_manager.get_holdings_analysis()
        
        # Generate signals for each holding
        holding_signals = []
        for pos in summary.get("holdings", []):
            signal = self.signal_generator.generate_signal(pos["symbol"])
            holding_signals.append({
                "symbol": pos["symbol"],
                "current_signal": signal.get("signal", "HOLD"),
                "confidence": signal.get("confidence", 50),
                "action": self._get_recommended_action(pos, signal),
            })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "sector_allocation": sector_allocation,
            "risk": risk,
            "holdings_analysis": holdings_analysis,
            "signals": holding_signals,
        }
    
    def _get_recommended_action(self, position: Dict, signal: Dict) -> str:
        """Get recommended action for a position."""
        sig = signal.get("signal", "HOLD")
        conf = signal.get("confidence", 50)
        pnl_pct = position.get("pnl_pct", 0)
        
        if sig == "STRONG_SELL" and conf > 70:
            return "EXIT - Strong sell signal"
        elif sig == "SELL" and pnl_pct > 0:
            return "BOOK_PROFITS - Sell signal with gains"
        elif sig == "SELL" and pnl_pct < -5:
            return "CUT_LOSS - Sell signal with losses"
        elif sig in ["STRONG_BUY", "BUY"] and pnl_pct > 5:
            return "TRAIL_STOP - Trend continues"
        elif sig == "HOLD":
            return "HOLD - No action needed"
        else:
            return "MONITOR - Watch closely"
    
    def research_stock(self, symbol: str, deep: bool = True) -> Dict[str, Any]:
        """
        Perform deep research on a specific stock.
        
        Args:
            symbol: Stock symbol
            deep: Use advanced multi-timeframe analysis
        
        Returns:
            Comprehensive stock research
        """
        # Generate full signal (includes all analysis)
        signal = self.signal_generator.generate_signal(symbol)
        
        # Get advanced analysis for deeper insights
        advanced_analysis = {}
        if deep:
            try:
                from src.analyzers.advanced_analyzer import get_advanced_analysis
                advanced_analysis = get_advanced_analysis(symbol)
            except Exception as e:
                pass
        
        # Get additional context
        from src.data_fetchers.bulk_deals_fetcher import BulkDealsFetcher
        from src.data_fetchers.insider_fetcher import InsiderFetcher
        
        bulk_fetcher = BulkDealsFetcher()
        insider_fetcher = InsiderFetcher()
        
        bulk_analysis = bulk_fetcher.analyze_accumulation_distribution(symbol)
        insider_analysis = insider_fetcher.analyze_insider_sentiment(symbol)
        
        # Merge advanced data into signal
        if advanced_analysis:
            signal["advanced"] = advanced_analysis
            signal["timeframes"] = advanced_analysis.get("timeframes", {})
            signal["key_levels"] = advanced_analysis.get("key_levels", {})
            signal["relative_strength"] = advanced_analysis.get("relative_strength", {})
            signal["overall_score"] = advanced_analysis.get("overall_score", signal.get("combined_score", 50))
        
        # Get AI analysis using advanced data
        explanation = ""
        if hasattr(self.llm, "analyze_stock"):
            explanation = self.llm.analyze_stock(symbol, signal)
        elif hasattr(self.llm, "generate_signal_explanation"):
            explanation = self.llm.generate_signal_explanation(signal)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "signal": signal,
            "advanced": advanced_analysis,
            "bulk_deals": bulk_analysis,
            "insider_activity": insider_analysis,
            "ai_analysis": explanation,
        }
    
    def scan_opportunities(
        self,
        n: int = 10,
        universe: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Scan for trading opportunities.
        
        Args:
            n: Number of top opportunities
            universe: Stock universe to scan (None = Nifty 50)
        
        Returns:
            Scan results
        """
        from config.settings import NIFTY_50, FULL_UNIVERSE, ETF_LIST
        
        if universe is None:
            universe = NIFTY_50[:30]  # Default to top 30 Nifty stocks
        
        # Get market context first
        context = self.market_analyzer.is_market_favorable_for_longs()
        
        opportunities = self.signal_generator.get_top_opportunities(n=n, universe=universe)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "market_context": context,
            "scanned_stocks": len(universe),
            "opportunities_found": len(opportunities),
            "top_opportunities": opportunities,
        }
    
    def run_eod_analysis(self) -> Dict[str, Any]:
        """
        Run end-of-day analysis.
        
        Returns:
            EOD analysis
        """
        self.logger.info("Running EOD analysis...")
        
        # Market summary
        market_summary = self.market_analyzer.generate_market_context_summary()
        
        # Sector performance
        sector_perf = self.market_analyzer.get_sector_performance()
        
        # Portfolio performance
        portfolio = self.portfolio_manager.get_portfolio_summary()
        
        # Day's trades
        recent_trades = self.trade_tracker.get_recent_trades_summary(5)
        
        # Tomorrow's watchlist
        watchlist_signals = []
        for symbol in self.watchlist[:10]:
            try:
                signal = self.signal_generator.generate_signal(symbol)
                if signal.get("signal") in ["STRONG_BUY", "BUY"]:
                    watchlist_signals.append({
                        "symbol": symbol,
                        "signal": signal.get("signal"),
                        "confidence": signal.get("confidence"),
                    })
            except Exception:
                continue
        
        return {
            "timestamp": datetime.now().isoformat(),
            "market_summary": market_summary,
            "sector_performance": sector_perf,
            "portfolio": portfolio,
            "todays_trades": recent_trades,
            "tomorrows_watchlist": watchlist_signals[:5],
        }
    
    def add_to_portfolio(
        self,
        symbol: str,
        quantity: int,
        price: float,
        stop_loss: Optional[float] = None,
        target: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Add position to portfolio.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Entry price
            stop_loss: Stop loss price
            target: Target price
        
        Returns:
            Position result
        """
        return self.portfolio_manager.add_position(
            symbol=symbol,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            target=target,
        )
    
    def close_portfolio_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
    ) -> Dict[str, Any]:
        """
        Close portfolio position.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Exit price
        
        Returns:
            Trade result
        """
        result = self.portfolio_manager.close_position(symbol, quantity, price)
        
        # Log the trade
        if "error" not in result:
            position = self.portfolio_manager.get_position(symbol)
            self.trade_tracker.log_trade(
                symbol=symbol,
                entry_price=result.get("avg_buy_price", price),
                exit_price=price,
                quantity=quantity,
                entry_date=datetime.now().isoformat(),
                exit_date=datetime.now().isoformat(),
            )
        
        return result
    
    def add_to_watchlist(self, symbol: str) -> Dict[str, Any]:
        """
        Add symbol to watchlist.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Result
        """
        from src.utils.helpers import save_json
        
        symbol = symbol.upper()
        
        if symbol not in self.watchlist:
            self.watchlist.append(symbol)
            save_json({"symbols": self.watchlist}, DATA_DIR / "watchlist.json")
            return {"success": True, "message": f"Added {symbol} to watchlist"}
        
        return {"success": False, "message": f"{symbol} already in watchlist"}
    
    def remove_from_watchlist(self, symbol: str) -> Dict[str, Any]:
        """
        Remove symbol from watchlist.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Result
        """
        from src.utils.helpers import save_json
        
        symbol = symbol.upper()
        
        if symbol in self.watchlist:
            self.watchlist.remove(symbol)
            save_json({"symbols": self.watchlist}, DATA_DIR / "watchlist.json")
            return {"success": True, "message": f"Removed {symbol} from watchlist"}
        
        return {"success": False, "message": f"{symbol} not in watchlist"}
    
    def get_quick_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get quick quote for a symbol.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Quick quote data
        """
        return self.price_fetcher.get_quote(symbol)
    
    def get_trading_performance(self) -> Dict[str, Any]:
        """
        Get trading performance analysis.
        
        Returns:
            Performance analysis
        """
        performance = self.trade_tracker.analyze_performance()
        pattern_perf = self.trade_tracker.get_pattern_performance()
        suggestions = self.trade_tracker.get_improvement_suggestions()
        
        return {
            "performance": performance,
            "pattern_analysis": pattern_perf,
            "suggestions": suggestions,
        }


# Factory function
def get_agent() -> TradingAgent:
    """Get trading agent instance."""
    return TradingAgent()
