"""
Fundamental Analysis Module.

Analyzes fundamental data including:
- Valuation metrics (P/E, P/B, EV/EBITDA)
- Quality metrics (ROE, ROCE, margins)
- Growth metrics (revenue, earnings growth)
- Financial health (debt, liquidity)
- Red flag detection
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_fetchers.fundamental_fetcher import FundamentalFetcher
from src.utils.logger import LoggerMixin


class FundamentalAnalyzer(LoggerMixin):
    """
    Performs fundamental analysis on stocks.
    """
    
    def __init__(self):
        """Initialize the fundamental analyzer."""
        self.fetcher = FundamentalFetcher()
    
    def analyze_valuation(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze stock valuation.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Valuation analysis dictionary
        """
        fundamentals = self.fetcher.get_fundamentals(symbol)
        
        pe = fundamentals.get("pe_ratio")
        pb = fundamentals.get("pb_ratio")
        ev_ebitda = fundamentals.get("ev_to_ebitda")
        peg = fundamentals.get("peg_ratio")
        ps = fundamentals.get("ps_ratio")
        
        valuation = {
            "pe_ratio": pe,
            "pb_ratio": pb,
            "ev_to_ebitda": ev_ebitda,
            "peg_ratio": peg,
            "ps_ratio": ps,
            "pe_assessment": self._assess_pe(pe),
            "pb_assessment": self._assess_pb(pb),
            "overall_valuation": self._assess_overall_valuation(pe, pb, ev_ebitda),
            "valuation_score": self._calculate_valuation_score(pe, pb, ev_ebitda, peg),
        }
        
        return valuation
    
    def _assess_pe(self, pe: Optional[float]) -> str:
        """Assess P/E ratio."""
        if pe is None:
            return "UNKNOWN"
        if pe < 0:
            return "LOSS_MAKING"
        if pe < 12:
            return "UNDERVALUED"
        if pe < 20:
            return "FAIRLY_VALUED"
        if pe < 35:
            return "GROWTH_PREMIUM"
        return "EXPENSIVE"
    
    def _assess_pb(self, pb: Optional[float]) -> str:
        """Assess P/B ratio."""
        if pb is None:
            return "UNKNOWN"
        if pb < 1:
            return "BELOW_BOOK"
        if pb < 2:
            return "FAIRLY_VALUED"
        if pb < 4:
            return "PREMIUM"
        return "EXPENSIVE"
    
    def _assess_overall_valuation(
        self,
        pe: Optional[float],
        pb: Optional[float],
        ev_ebitda: Optional[float],
    ) -> str:
        """Assess overall valuation."""
        score = 0
        count = 0
        
        if pe is not None and pe > 0:
            if pe < 15:
                score += 2
            elif pe < 25:
                score += 1
            elif pe > 40:
                score -= 1
            count += 1
        
        if pb is not None:
            if pb < 2:
                score += 1
            elif pb > 5:
                score -= 1
            count += 1
        
        if ev_ebitda is not None:
            if ev_ebitda < 10:
                score += 1
            elif ev_ebitda > 20:
                score -= 1
            count += 1
        
        if count == 0:
            return "UNKNOWN"
        
        avg = score / count
        if avg > 1:
            return "UNDERVALUED"
        if avg > 0:
            return "FAIRLY_VALUED"
        if avg > -0.5:
            return "SLIGHTLY_EXPENSIVE"
        return "EXPENSIVE"
    
    def _calculate_valuation_score(
        self,
        pe: Optional[float],
        pb: Optional[float],
        ev_ebitda: Optional[float],
        peg: Optional[float],
    ) -> float:
        """Calculate valuation score (0-100, higher = cheaper)."""
        score = 50
        
        if pe is not None and pe > 0:
            if pe < 12:
                score += 15
            elif pe < 18:
                score += 10
            elif pe < 25:
                score += 5
            elif pe > 40:
                score -= 15
            elif pe > 30:
                score -= 10
        
        if pb is not None:
            if pb < 1:
                score += 10
            elif pb < 2:
                score += 5
            elif pb > 5:
                score -= 10
            elif pb > 3:
                score -= 5
        
        if ev_ebitda is not None:
            if ev_ebitda < 8:
                score += 10
            elif ev_ebitda < 12:
                score += 5
            elif ev_ebitda > 20:
                score -= 10
        
        if peg is not None:
            if peg < 0.5:
                score += 10
            elif peg < 1:
                score += 5
            elif peg > 2:
                score -= 5
        
        return max(0, min(100, score))
    
    def analyze_quality(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze company quality.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Quality analysis dictionary
        """
        fundamentals = self.fetcher.get_fundamentals(symbol)
        
        roe = fundamentals.get("roe")
        roa = fundamentals.get("roa")
        profit_margin = fundamentals.get("profit_margin")
        operating_margin = fundamentals.get("operating_margin")
        
        quality = {
            "roe": roe,
            "roa": roa,
            "profit_margin": profit_margin,
            "operating_margin": operating_margin,
            "roe_assessment": self._assess_roe(roe),
            "margin_assessment": self._assess_margins(profit_margin, operating_margin),
            "quality_score": self._calculate_quality_score(
                roe, roa, profit_margin, operating_margin
            ),
        }
        
        return quality
    
    def _assess_roe(self, roe: Optional[float]) -> str:
        """Assess ROE."""
        if roe is None:
            return "UNKNOWN"
        if roe < 0:
            return "POOR"
        if roe < 0.10:
            return "BELOW_AVERAGE"
        if roe < 0.15:
            return "AVERAGE"
        if roe < 0.20:
            return "GOOD"
        return "EXCELLENT"
    
    def _assess_margins(
        self,
        profit_margin: Optional[float],
        operating_margin: Optional[float],
    ) -> str:
        """Assess profit margins."""
        margin = profit_margin or operating_margin
        if margin is None:
            return "UNKNOWN"
        if margin < 0:
            return "LOSS_MAKING"
        if margin < 0.05:
            return "THIN"
        if margin < 0.10:
            return "MODERATE"
        if margin < 0.20:
            return "HEALTHY"
        return "EXCELLENT"
    
    def _calculate_quality_score(
        self,
        roe: Optional[float],
        roa: Optional[float],
        profit_margin: Optional[float],
        operating_margin: Optional[float],
    ) -> float:
        """Calculate quality score (0-100)."""
        score = 50
        
        if roe is not None:
            if roe > 0.25:
                score += 20
            elif roe > 0.18:
                score += 15
            elif roe > 0.12:
                score += 10
            elif roe < 0.05:
                score -= 15
            elif roe < 0.08:
                score -= 10
        
        if roa is not None:
            if roa > 0.10:
                score += 10
            elif roa > 0.05:
                score += 5
            elif roa < 0:
                score -= 10
        
        if profit_margin is not None:
            if profit_margin > 0.20:
                score += 10
            elif profit_margin > 0.10:
                score += 5
            elif profit_margin < 0:
                score -= 15
        
        return max(0, min(100, score))
    
    def analyze_growth(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze growth metrics.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Growth analysis dictionary
        """
        fundamentals = self.fetcher.get_fundamentals(symbol)
        
        revenue_growth = fundamentals.get("revenue_growth")
        earnings_growth = fundamentals.get("earnings_growth")
        earnings_qtr_growth = fundamentals.get("earnings_quarterly_growth")
        
        growth = {
            "revenue_growth": revenue_growth,
            "earnings_growth": earnings_growth,
            "earnings_quarterly_growth": earnings_qtr_growth,
            "growth_assessment": self._assess_growth(revenue_growth, earnings_growth),
            "growth_score": self._calculate_growth_score(
                revenue_growth, earnings_growth, earnings_qtr_growth
            ),
        }
        
        return growth
    
    def _assess_growth(
        self,
        revenue_growth: Optional[float],
        earnings_growth: Optional[float],
    ) -> str:
        """Assess growth quality."""
        growth = earnings_growth or revenue_growth
        if growth is None:
            return "UNKNOWN"
        if growth > 0.30:
            return "HIGH_GROWTH"
        if growth > 0.15:
            return "GOOD_GROWTH"
        if growth > 0.05:
            return "MODERATE_GROWTH"
        if growth > 0:
            return "SLOW_GROWTH"
        return "DECLINING"
    
    def _calculate_growth_score(
        self,
        revenue_growth: Optional[float],
        earnings_growth: Optional[float],
        earnings_qtr_growth: Optional[float],
    ) -> float:
        """Calculate growth score (0-100)."""
        score = 50
        
        if earnings_growth is not None:
            if earnings_growth > 0.30:
                score += 20
            elif earnings_growth > 0.15:
                score += 15
            elif earnings_growth > 0.05:
                score += 10
            elif earnings_growth < 0:
                score -= 15
        
        if revenue_growth is not None:
            if revenue_growth > 0.20:
                score += 10
            elif revenue_growth > 0.10:
                score += 5
            elif revenue_growth < 0:
                score -= 10
        
        if earnings_qtr_growth is not None:
            if earnings_qtr_growth > 0.20:
                score += 10
            elif earnings_qtr_growth < -0.10:
                score -= 10
        
        return max(0, min(100, score))
    
    def analyze_financial_health(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze financial health.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Financial health dictionary
        """
        fundamentals = self.fetcher.get_fundamentals(symbol)
        
        debt_to_equity = fundamentals.get("debt_to_equity")
        current_ratio = fundamentals.get("current_ratio")
        quick_ratio = fundamentals.get("quick_ratio")
        free_cashflow = fundamentals.get("free_cashflow")
        
        health = {
            "debt_to_equity": debt_to_equity,
            "current_ratio": current_ratio,
            "quick_ratio": quick_ratio,
            "free_cashflow": free_cashflow,
            "debt_assessment": self._assess_debt(debt_to_equity),
            "liquidity_assessment": self._assess_liquidity(current_ratio),
            "health_score": self._calculate_health_score(
                debt_to_equity, current_ratio, free_cashflow
            ),
        }
        
        return health
    
    def _assess_debt(self, debt_to_equity: Optional[float]) -> str:
        """Assess debt levels."""
        if debt_to_equity is None:
            return "UNKNOWN"
        if debt_to_equity < 0.3:
            return "LOW_DEBT"
        if debt_to_equity < 0.7:
            return "MODERATE_DEBT"
        if debt_to_equity < 1.5:
            return "HIGH_DEBT"
        return "VERY_HIGH_DEBT"
    
    def _assess_liquidity(self, current_ratio: Optional[float]) -> str:
        """Assess liquidity."""
        if current_ratio is None:
            return "UNKNOWN"
        if current_ratio < 0.8:
            return "WEAK"
        if current_ratio < 1.2:
            return "ADEQUATE"
        if current_ratio < 2:
            return "GOOD"
        return "STRONG"
    
    def _calculate_health_score(
        self,
        debt_to_equity: Optional[float],
        current_ratio: Optional[float],
        free_cashflow: Optional[float],
    ) -> float:
        """Calculate financial health score (0-100)."""
        score = 50
        
        if debt_to_equity is not None:
            if debt_to_equity < 0.3:
                score += 20
            elif debt_to_equity < 0.7:
                score += 10
            elif debt_to_equity > 2:
                score -= 20
            elif debt_to_equity > 1:
                score -= 10
        
        if current_ratio is not None:
            if current_ratio > 2:
                score += 10
            elif current_ratio > 1.5:
                score += 5
            elif current_ratio < 1:
                score -= 15
        
        if free_cashflow is not None:
            if free_cashflow > 0:
                score += 10
            else:
                score -= 10
        
        return max(0, min(100, score))
    
    def detect_red_flags(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Detect fundamental red flags.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            List of red flags
        """
        return self.fetcher.detect_red_flags(symbol)
    
    def analyze_shareholding(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze shareholding pattern.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Shareholding analysis
        """
        shareholding = self.fetcher.get_shareholding_pattern(symbol)
        
        promoter = shareholding.get("promoter_holding")
        fii = shareholding.get("fii_holding")
        dii = shareholding.get("dii_holding")
        
        analysis = {
            "promoter_holding": promoter,
            "fii_holding": fii,
            "dii_holding": dii,
            "promoter_assessment": self._assess_promoter_holding(promoter),
            "institutional_interest": self._assess_institutional(fii, dii),
        }
        
        return analysis
    
    def _assess_promoter_holding(self, holding: Optional[float]) -> str:
        """Assess promoter holding."""
        if holding is None:
            return "UNKNOWN"
        if holding > 70:
            return "HIGH"
        if holding > 50:
            return "MAJORITY"
        if holding > 30:
            return "SIGNIFICANT"
        return "LOW"
    
    def _assess_institutional(
        self,
        fii: Optional[float],
        dii: Optional[float],
    ) -> str:
        """Assess institutional interest."""
        total = (fii or 0) + (dii or 0)
        if total > 50:
            return "HIGH_INTEREST"
        if total > 30:
            return "MODERATE_INTEREST"
        if total > 15:
            return "LOW_INTEREST"
        return "MINIMAL"
    
    def calculate_fundamental_score(self, symbol: str) -> float:
        """
        Calculate overall fundamental score.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Fundamental score (0-100)
        """
        valuation = self.analyze_valuation(symbol)
        quality = self.analyze_quality(symbol)
        growth = self.analyze_growth(symbol)
        health = self.analyze_financial_health(symbol)
        red_flags = self.detect_red_flags(symbol)
        
        # Weighted average
        score = (
            valuation["valuation_score"] * 0.25 +
            quality["quality_score"] * 0.30 +
            growth["growth_score"] * 0.25 +
            health["health_score"] * 0.20
        )
        
        # Deduct for red flags
        for flag in red_flags:
            severity = flag.get("severity", "LOW")
            if severity == "HIGH":
                score -= 10
            elif severity == "MEDIUM":
                score -= 5
            else:
                score -= 2
        
        return max(0, min(100, score))
    
    def generate_fundamental_summary(self, symbol: str) -> Dict[str, Any]:
        """
        Generate comprehensive fundamental analysis summary.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Fundamental summary dictionary
        """
        fundamentals = self.fetcher.get_fundamentals(symbol)
        
        valuation = self.analyze_valuation(symbol)
        quality = self.analyze_quality(symbol)
        growth = self.analyze_growth(symbol)
        health = self.analyze_financial_health(symbol)
        shareholding = self.analyze_shareholding(symbol)
        red_flags = self.detect_red_flags(symbol)
        
        overall_score = self.calculate_fundamental_score(symbol)
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "fundamental_score": overall_score,
            "valuation": valuation,
            "quality": quality,
            "growth": growth,
            "financial_health": health,
            "shareholding": shareholding,
            "red_flags": red_flags,
            "sector": fundamentals.get("sector"),
            "industry": fundamentals.get("industry"),
            "market_cap": fundamentals.get("market_cap"),
            "summary": self._generate_text_summary(
                valuation, quality, growth, health, red_flags
            ),
        }
    
    def _generate_text_summary(
        self,
        valuation: Dict,
        quality: Dict,
        growth: Dict,
        health: Dict,
        red_flags: List,
    ) -> str:
        """Generate text summary of fundamentals."""
        parts = []
        
        # Valuation
        val_assessment = valuation.get("overall_valuation", "UNKNOWN")
        parts.append(f"Valuation: {val_assessment}")
        
        # Quality
        quality_score = quality.get("quality_score", 50)
        if quality_score > 70:
            parts.append("High quality company")
        elif quality_score > 50:
            parts.append("Moderate quality")
        else:
            parts.append("Quality concerns")
        
        # Growth
        growth_assessment = growth.get("growth_assessment", "UNKNOWN")
        parts.append(f"Growth: {growth_assessment}")
        
        # Health
        debt = health.get("debt_assessment", "UNKNOWN")
        parts.append(f"Debt: {debt}")
        
        # Red flags
        if red_flags:
            parts.append(f"Red flags: {len(red_flags)}")
        
        return " | ".join(parts)


# Convenience function
def analyze_fundamentals(symbol: str) -> Dict:
    """Perform fundamental analysis on a symbol."""
    analyzer = FundamentalAnalyzer()
    return analyzer.generate_fundamental_summary(symbol)
