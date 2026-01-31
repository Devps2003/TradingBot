"""
💰 MONEY MAKER - Find the Best Trades

This module scans the market to find:
1. High probability setups (score > 70)
2. Excellent risk-reward (> 1:2)
3. Multiple confirmations
4. Smart money accumulation

The goal: Quality over quantity. Find 2-3 great trades, not 20 mediocre ones.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analyzers.pro_analyzer import ProAnalyzer
from src.utils.logger import LoggerMixin
from config.settings import NIFTY_50, NIFTY_NEXT_50, ETF_LIST, FULL_UNIVERSE


class MoneyMaker(LoggerMixin):
    """
    Scans the market for high-probability, money-making opportunities.
    """
    
    def __init__(self):
        self.analyzer = ProAnalyzer()
    
    def scan_for_opportunities(
        self,
        universe: Optional[List[str]] = None,
        min_score: int = 65,
        min_rr: float = 1.5,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """
        Scan multiple stocks and return the best opportunities.
        
        Args:
            universe: List of stocks to scan (default: Nifty 50)
            min_score: Minimum overall score (0-100)
            min_rr: Minimum risk-reward ratio
            max_results: Maximum opportunities to return
        """
        if universe is None:
            universe = NIFTY_50
        
        self.logger.info(f"Scanning {len(universe)} stocks for opportunities...")
        
        opportunities = []
        errors = []
        
        # Parallel scanning for speed
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self._analyze_stock, symbol): symbol 
                for symbol in universe
            }
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    if result and "error" not in result:
                        # Filter by criteria
                        if self._meets_criteria(result, min_score, min_rr):
                            opportunities.append(result)
                except Exception as e:
                    errors.append({"symbol": symbol, "error": str(e)})
        
        # Sort by score (best first)
        opportunities.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
        
        # Take top results
        top_opportunities = opportunities[:max_results]
        
        # Categorize
        strong_buys = [o for o in top_opportunities if o.get("signal", {}).get("signal") == "STRONG_BUY"]
        buys = [o for o in top_opportunities if o.get("signal", {}).get("signal") == "BUY"]
        watchlist = [o for o in opportunities[max_results:max_results+5] if o.get("overall_score", 0) >= 55]
        
        return {
            "scan_time": datetime.now().isoformat(),
            "universe_size": len(universe),
            "opportunities_found": len(opportunities),
            "strong_buys": strong_buys,
            "buys": buys,
            "watchlist": watchlist,
            "top_opportunities": top_opportunities,
            "errors": errors[:5],  # Limit error reporting
        }
    
    def _analyze_stock(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze a single stock."""
        try:
            return self.analyzer.full_analysis(symbol)
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def _meets_criteria(self, result: Dict, min_score: int, min_rr: float) -> bool:
        """Check if opportunity meets our criteria."""
        score = result.get("overall_score", 0)
        signal = result.get("signal", {})
        rr = signal.get("risk_reward_ratio", 0)
        sig = signal.get("signal", "")
        
        # Must be a buy signal
        if "BUY" not in sig:
            return False
        
        # Must meet minimum score
        if score < min_score:
            return False
        
        # Must have good R:R
        if rr < min_rr:
            return False
        
        # Additional filters
        trend = result.get("trend", {})
        volume = result.get("volume", {})
        
        # Trend should be aligned
        if not trend.get("aligned", False) and score < 75:
            return False
        
        # Smart money should not be distributing
        if volume.get("smart_money") == "DISTRIBUTING":
            return False
        
        return True
    
    def find_breakouts(self, universe: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Find stocks breaking out of consolidation.
        These often lead to big moves.
        """
        if universe is None:
            universe = NIFTY_50 + NIFTY_NEXT_50[:20]
        
        breakouts = []
        
        for symbol in universe[:50]:  # Limit for speed
            try:
                result = self.analyzer.full_analysis(symbol)
                
                if "error" in result:
                    continue
                
                patterns = result.get("patterns", {}).get("patterns", [])
                
                # Check for breakout patterns
                for pattern in patterns:
                    if "Breakout" in pattern.get("name", "") and pattern.get("type") == "BULLISH":
                        breakouts.append({
                            "symbol": symbol,
                            "pattern": pattern["name"],
                            "score": result.get("overall_score", 0),
                            "signal": result.get("signal", {}),
                            "volume": result.get("volume", {}).get("smart_money", "N/A"),
                        })
                        break
            except:
                continue
        
        return sorted(breakouts, key=lambda x: x.get("score", 0), reverse=True)
    
    def find_reversals(self, universe: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Find potential reversal candidates.
        Stocks that are oversold with bullish divergence.
        """
        if universe is None:
            universe = NIFTY_50
        
        reversals = []
        
        for symbol in universe[:50]:
            try:
                result = self.analyzer.full_analysis(symbol)
                
                if "error" in result:
                    continue
                
                momentum = result.get("momentum", {})
                volume = result.get("volume", {})
                
                # Oversold conditions
                rsi = momentum.get("rsi", 50)
                divergence = momentum.get("rsi_divergence", "NONE")
                stoch_oversold = momentum.get("stochastic", {}).get("oversold", False)
                
                if (rsi < 35 or stoch_oversold) and \
                   (divergence == "BULLISH" or volume.get("smart_money") == "ACCUMULATING"):
                    reversals.append({
                        "symbol": symbol,
                        "rsi": rsi,
                        "divergence": divergence,
                        "smart_money": volume.get("smart_money"),
                        "score": result.get("overall_score", 0),
                        "signal": result.get("signal", {}),
                    })
            except:
                continue
        
        return sorted(reversals, key=lambda x: x.get("score", 0), reverse=True)
    
    def find_momentum_leaders(self, universe: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Find stocks with strong momentum and relative strength.
        Leaders tend to keep leading.
        """
        if universe is None:
            universe = NIFTY_50 + NIFTY_NEXT_50[:30]
        
        leaders = []
        
        for symbol in universe[:60]:
            try:
                result = self.analyzer.full_analysis(symbol)
                
                if "error" in result:
                    continue
                
                trend = result.get("trend", {})
                rs = result.get("relative_strength", {})
                momentum = result.get("momentum", {})
                
                # Strong RS + Bullish trend + Good momentum
                rs_score = rs.get("rs_score", 50)
                
                if rs_score >= 65 and \
                   trend.get("daily") == "BULLISH" and \
                   trend.get("above_200ema", False) and \
                   momentum.get("condition") in ["BULLISH", "NEUTRAL"]:
                    leaders.append({
                        "symbol": symbol,
                        "rs_score": rs_score,
                        "alpha_1m": rs.get("alpha_1m", 0),
                        "trend": trend.get("daily"),
                        "score": result.get("overall_score", 0),
                        "signal": result.get("signal", {}),
                    })
            except:
                continue
        
        return sorted(leaders, key=lambda x: x.get("rs_score", 0), reverse=True)[:10]
    
    def get_market_dashboard(self) -> Dict[str, Any]:
        """
        Get complete market dashboard with all opportunities.
        """
        self.logger.info("Generating market dashboard...")
        
        # Run all scans
        opportunities = self.scan_for_opportunities(max_results=5)
        breakouts = self.find_breakouts()[:5]
        reversals = self.find_reversals()[:5]
        leaders = self.find_momentum_leaders()[:5]
        
        return {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "strong_buys": len(opportunities.get("strong_buys", [])),
                "buys": len(opportunities.get("buys", [])),
                "breakouts": len(breakouts),
                "reversals": len(reversals),
                "leaders": len(leaders),
            },
            "opportunities": opportunities.get("top_opportunities", []),
            "breakouts": breakouts,
            "reversals": reversals,
            "momentum_leaders": leaders,
            "watchlist": opportunities.get("watchlist", []),
        }


# Convenience function
def find_money_makers(universe: Optional[List[str]] = None) -> Dict[str, Any]:
    """Find the best money-making opportunities."""
    mm = MoneyMaker()
    return mm.scan_for_opportunities(universe)


def get_dashboard() -> Dict[str, Any]:
    """Get complete market dashboard."""
    mm = MoneyMaker()
    return mm.get_market_dashboard()
