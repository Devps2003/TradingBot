"""
Main Entry Point and CLI Interface.

Commands:
- python main.py morning        # Morning briefing
- python main.py portfolio      # Portfolio analysis
- python main.py research SYMBOL # Deep research on stock
- python main.py scan           # Scan for opportunities
- python main.py eod            # End of day summary
- python main.py add SYMBOL QTY PRICE  # Add to portfolio
- python main.py remove SYMBOL QTY PRICE # Remove from portfolio
- python main.py watchlist add SYMBOL  # Add to watchlist
- python main.py quote SYMBOL   # Quick quote
"""

import sys
import warnings
import logging
from pathlib import Path

# Suppress warnings and noisy logs
warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich/Typer not installed. Install with: pip install rich typer")

from src.agent import TradingAgent

# Initialize
app = typer.Typer(help="Indian Market Trading Research Agent")
console = Console() if RICH_AVAILABLE else None


def print_header(title: str):
    """Print styled header."""
    if console:
        console.print(Panel(
            Text(title, justify="center", style="bold cyan"),
            box=box.DOUBLE,
            padding=(1, 2),
        ))
    else:
        print(f"\n{'='*60}\n{title:^60}\n{'='*60}")


def print_signal_box(signal: str, confidence: float):
    """Print signal in styled box."""
    colors = {
        "STRONG_BUY": "bold green",
        "BUY": "green",
        "HOLD": "yellow",
        "SELL": "red",
        "STRONG_SELL": "bold red",
    }
    
    if console:
        color = colors.get(signal, "white")
        console.print(Panel(
            Text(f"{signal}\nConfidence: {confidence:.0f}%", justify="center", style=color),
            title="SIGNAL",
            box=box.HEAVY,
        ))
    else:
        print(f"\nSIGNAL: {signal} (Confidence: {confidence:.0f}%)")


@app.command()
def morning():
    """Run morning briefing."""
    agent = TradingAgent()
    
    print_header("MORNING BRIEFING")
    
    if console:
        with console.status("[bold green]Generating morning briefing..."):
            briefing = agent.run_morning_briefing()
    else:
        print("Generating morning briefing...")
        briefing = agent.run_morning_briefing()
    
    # Session info
    session = briefing.get("session", {})
    if console:
        console.print(f"\n[bold]Market Status:[/bold] {session.get('status', 'UNKNOWN')}")
        console.print(f"[bold]Time:[/bold] {session.get('current_time', '')}")
    
    # Global cues
    global_cues = briefing.get("global_cues", {})
    if console and global_cues:
        # Indian Markets Table (Most Important!)
        indian_table = Table(title="🇮🇳 Indian Markets", box=box.ROUNDED)
        indian_table.add_column("Index", style="bold cyan")
        indian_table.add_column("Value", style="white", justify="right")
        indian_table.add_column("Change", justify="right")
        
        # Look in indian_markets first, then asian_markets as fallback
        indian_markets = global_cues.get("indian_markets", {})
        if not indian_markets:
            indian_markets = global_cues.get("asian_markets", {})
        
        has_indian_data = False
        for idx_name in ["nifty", "sensex", "nifty_bank"]:
            data = indian_markets.get(idx_name, {})
            if isinstance(data, dict) and data.get("price"):
                has_indian_data = True
                change_pct = data.get("change_pct", 0) or 0
                color = "green" if change_pct >= 0 else "red"
                indian_table.add_row(
                    idx_name.upper().replace("_", " "),
                    f"{data.get('price', 0):,.2f}",
                    f"[{color}]{change_pct:+.2f}%[/{color}]"
                )
        
        if has_indian_data:
            console.print(indian_table)
        else:
            console.print("[yellow]Indian market data not available (market may be closed)[/yellow]")
        
        # US Markets Table
        us_table = Table(title="🇺🇸 US Markets (Yesterday Close)", box=box.ROUNDED)
        us_table.add_column("Index", style="cyan")
        us_table.add_column("Value", style="white", justify="right")
        us_table.add_column("Change", justify="right")
        
        us_close = global_cues.get("us_close", {})
        for idx, data in us_close.items():
            if isinstance(data, dict) and data.get("price"):
                change_pct = data.get("change_pct", 0) or 0
                color = "green" if change_pct >= 0 else "red"
                us_table.add_row(
                    idx.upper(),
                    f"{data.get('price', 0):,.2f}",
                    f"[{color}]{change_pct:+.2f}%[/{color}]"
                )
        
        console.print(us_table)
        
        # Key levels
        exp_gap = global_cues.get("expected_gap", "FLAT")
        exp_impact = global_cues.get("expected_nifty_impact", 0)
        gap_color = "green" if exp_gap == "GAP_UP" else "red" if exp_gap == "GAP_DOWN" else "yellow"
        console.print(f"\n[bold]Expected Opening:[/bold] [{gap_color}]{exp_gap}[/{gap_color}] ({exp_impact:+.2f}%)")
    
    # FII/DII
    fii_dii = briefing.get("fii_dii", {})
    if console and fii_dii:
        console.print(f"\n[bold]FII Net:[/bold] ₹{fii_dii.get('fii_net', 0):,.0f} Cr")
        console.print(f"[bold]DII Net:[/bold] ₹{fii_dii.get('dii_net', 0):,.0f} Cr")
        console.print(f"[bold]Sentiment:[/bold] {fii_dii.get('sentiment', 'NEUTRAL')}")
    
    # Portfolio summary
    portfolio = briefing.get("portfolio", {})
    if console and portfolio:
        console.print(f"\n[bold cyan]Portfolio Value:[/bold cyan] ₹{portfolio.get('portfolio_value', 0):,.0f}")
        pnl = portfolio.get("unrealized_pnl", 0)
        pnl_pct = portfolio.get("unrealized_pnl_pct", 0)
        color = "green" if pnl >= 0 else "red"
        console.print(f"[bold]Unrealized P&L:[/bold] [{color}]₹{pnl:,.0f} ({pnl_pct:+.2f}%)[/{color}]")
    
    # Alerts
    alerts = briefing.get("alerts", [])
    if console and alerts:
        console.print("\n[bold red]ALERTS:[/bold red]")
        for alert in alerts[:5]:
            icon = "⚠️" if alert["priority"] == "HIGH" else "🔴" if alert["priority"] == "CRITICAL" else "ℹ️"
            console.print(f"  {icon} {alert['message']}")
    
    # AI Briefing
    ai_brief = briefing.get("briefing", "")
    if console and ai_brief:
        console.print(Panel(ai_brief, title="AI Analysis", box=box.ROUNDED))


@app.command()
def portfolio():
    """Analyze portfolio."""
    agent = TradingAgent()
    
    print_header("PORTFOLIO ANALYSIS")
    
    if console:
        with console.status("[bold green]Analyzing portfolio..."):
            analysis = agent.analyze_portfolio()
    else:
        analysis = agent.analyze_portfolio()
    
    summary = analysis.get("summary", {})
    holdings = summary.get("holdings", [])
    
    if console and holdings:
        table = Table(title="Holdings", box=box.ROUNDED)
        table.add_column("Symbol", style="cyan")
        table.add_column("Qty", justify="right")
        table.add_column("Avg", justify="right")
        table.add_column("CMP", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("P&L%", justify="right")
        table.add_column("Signal", justify="center")
        
        signals = {s["symbol"]: s for s in analysis.get("signals", [])}
        
        for h in holdings:
            pnl_color = "green" if h["pnl"] >= 0 else "red"
            signal_info = signals.get(h["symbol"], {})
            signal = signal_info.get("current_signal", "HOLD")
            signal_color = "green" if "BUY" in signal else "red" if "SELL" in signal else "yellow"
            
            table.add_row(
                h["symbol"],
                str(h["quantity"]),
                f"₹{h['avg_price']:.2f}",
                f"₹{h['current_price']:.2f}",
                f"[{pnl_color}]₹{h['pnl']:,.0f}[/{pnl_color}]",
                f"[{pnl_color}]{h['pnl_pct']:+.2f}%[/{pnl_color}]",
                f"[{signal_color}]{signal}[/{signal_color}]",
            )
        
        console.print(table)
        
        # Summary
        console.print(f"\n[bold]Total Invested:[/bold] ₹{summary.get('total_invested', 0):,.0f}")
        console.print(f"[bold]Current Value:[/bold] ₹{summary.get('total_current', 0):,.0f}")
        
        pnl = summary.get("unrealized_pnl", 0)
        color = "green" if pnl >= 0 else "red"
        console.print(f"[bold]Total P&L:[/bold] [{color}]₹{pnl:,.0f}[/{color}]")
    else:
        print("No holdings in portfolio")


@app.command()
def research(symbol: str):
    """Deep research on a stock with multi-timeframe analysis."""
    agent = TradingAgent()
    
    print_header(f"RESEARCH: {symbol.upper()}")
    
    if console:
        with console.status(f"[bold green]Running advanced analysis for {symbol}..."):
            result = agent.research_stock(symbol.upper(), deep=True)
    else:
        result = agent.research_stock(symbol.upper(), deep=True)
    
    signal = result.get("signal", {})
    advanced = result.get("advanced", {})
    
    if console:
        # Signal box
        overall_score = advanced.get("overall_score", signal.get("combined_score", 50))
        sig_strength = advanced.get("signal_strength", signal.get("signal", "HOLD"))
        print_signal_box(sig_strength, signal.get("confidence", 50))
        
        # Key levels from advanced analysis
        levels = advanced.get("key_levels", {}) or signal.get("key_levels", {})
        if levels:
            console.print("\n[bold cyan]═══ KEY LEVELS ═══[/bold cyan]")
            console.print(f"  Current:    ₹{levels.get('current', signal.get('entry_price', 0)):.2f}")
            console.print(f"  Resistance: ₹{levels.get('resistance_1', 0):.2f}")
            console.print(f"  Support:    ₹{levels.get('support_1', 0):.2f}")
            console.print(f"  Stop Loss:  ₹{signal.get('stop_loss', 0):.2f}")
            console.print(f"  Target:     ₹{signal.get('target_1', 0):.2f}")
            console.print(f"  52W Range:  ₹{levels.get('low_52w', 0):.2f} - ₹{levels.get('high_52w', 0):.2f}")
        
        # Multi-timeframe analysis
        timeframes = advanced.get("timeframes", {})
        if timeframes:
            console.print("\n[bold cyan]═══ MULTI-TIMEFRAME ═══[/bold cyan]")
            tf_table = Table(box=box.SIMPLE, show_header=True)
            tf_table.add_column("Timeframe", style="cyan")
            tf_table.add_column("Trend", justify="center")
            tf_table.add_column("RSI", justify="right")
            tf_table.add_column("MACD", justify="center")
            
            for tf in ["daily", "weekly", "monthly"]:
                data = timeframes.get(tf, {})
                trend = data.get("trend", "N/A")
                trend_color = "green" if "BULLISH" in trend else "red" if "BEARISH" in trend else "yellow"
                tf_table.add_row(
                    tf.capitalize(),
                    f"[{trend_color}]{trend}[/{trend_color}]",
                    f"{data.get('rsi', 'N/A')}",
                    data.get("macd_signal", "N/A"),
                )
            
            alignment = timeframes.get("alignment", {})
            align_status = alignment.get("status", "N/A")
            align_color = "green" if "BULLISH" in align_status else "red" if "BEARISH" in align_status else "yellow"
            
            console.print(tf_table)
            console.print(f"  Alignment: [{align_color}]{align_status}[/{align_color}]")
        
        # Relative Strength
        rs = advanced.get("relative_strength", {})
        if rs:
            rs_score = rs.get("rs_score", 50)
            rs_color = "green" if rs_score > 60 else "red" if rs_score < 40 else "yellow"
            console.print(f"\n[bold cyan]═══ RELATIVE STRENGTH ═══[/bold cyan]")
            console.print(f"  RS Score: [{rs_color}]{rs_score:.0f}/100[/{rs_color}]")
            console.print(f"  vs Nifty: {rs.get('vs_market', 'N/A')}")
            console.print(f"  20D Return: Stock {rs.get('stock_return_20d', 0):+.1f}% vs Nifty {rs.get('nifty_return_20d', 0):+.1f}%")
        
        # Momentum & Risk
        momentum = advanced.get("momentum", {})
        risk = advanced.get("risk", {})
        if momentum or risk:
            console.print(f"\n[bold cyan]═══ MOMENTUM & RISK ═══[/bold cyan]")
            if momentum:
                cond = momentum.get("condition", "N/A")
                cond_color = "red" if cond == "OVERBOUGHT" else "green" if cond == "OVERSOLD" else "yellow"
                console.print(f"  RSI: {momentum.get('rsi', 'N/A')} | Stochastic: {momentum.get('stochastic_k', 'N/A')}")
                console.print(f"  Condition: [{cond_color}]{cond}[/{cond_color}]")
            if risk:
                risk_lvl = risk.get("risk_level", "N/A")
                risk_color = "red" if risk_lvl == "HIGH" else "green" if risk_lvl == "LOW" else "yellow"
                console.print(f"  Risk Level: [{risk_color}]{risk_lvl}[/{risk_color}]")
                console.print(f"  Volatility: {risk.get('volatility_20d', 'N/A')}%")
                console.print(f"  Suggested Stop: {risk.get('suggested_stop_pct', 'N/A')}%")
        
        # Breakout
        breakout = advanced.get("breakout", {})
        if breakout:
            status = breakout.get("status", "N/A")
            if status != "NEUTRAL":
                status_color = "green" if "BREAKOUT" in status else "red" if "BREAKDOWN" in status else "cyan"
                console.print(f"\n[{status_color}]⚡ {status}[/{status_color}]")
        
        # Scores
        scores = signal.get("scores", {})
        if scores:
            console.print(f"\n[bold cyan]═══ ANALYSIS SCORES ═══[/bold cyan]")
            score_table = Table(box=box.SIMPLE, show_header=False)
            score_table.add_column("Component", style="cyan")
            score_table.add_column("Score", justify="right")
            score_table.add_column("Bar")
            
            for name, score in scores.items():
                color = "green" if score > 60 else "red" if score < 40 else "yellow"
                bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
                score_table.add_row(name.title(), f"[{color}]{score:.0f}[/{color}]", f"[{color}]{bar}[/{color}]")
            
            console.print(score_table)
            console.print(f"\n  [bold]Overall Score: {overall_score:.0f}/100[/bold]")
        
        # AI Analysis
        ai = result.get("ai_analysis", "")
        if ai:
            console.print(Panel(ai, title="🤖 AI Analysis", box=box.ROUNDED, border_style="blue"))


@app.command()
def scan(
    n: int = 10,
    universe: str = typer.Option("nifty", help="Stock universe: nifty, all, etf")
):
    """Scan for trading opportunities. Use --universe etf for ETFs, --universe all for full market."""
    agent = TradingAgent()
    
    print_header("OPPORTUNITY SCANNER")
    
    # Select universe
    scan_universe = None
    if universe.lower() == "etf":
        from config.settings import ETF_LIST
        scan_universe = ETF_LIST
        if console:
            console.print("[cyan]Scanning ETFs (Gold, Silver, Index)...[/cyan]")
    elif universe.lower() == "all":
        from config.settings import FULL_UNIVERSE
        scan_universe = FULL_UNIVERSE[:50]  # Limit for speed
        if console:
            console.print("[cyan]Scanning full market (50 stocks)...[/cyan]")
    else:
        if console:
            console.print("[cyan]Scanning Nifty 50 stocks...[/cyan]")
    
    if console:
        with console.status("[bold green]Scanning..."):
            result = agent.scan_opportunities(n=n, universe=scan_universe)
    else:
        result = agent.scan_opportunities(n=n, universe=scan_universe)
    
    context = result.get("market_context", {})
    if console:
        favorable = context.get("favorable_for_longs", False)
        color = "green" if favorable else "red"
        console.print(f"\n[bold]Market Favorable:[/bold] [{color}]{favorable}[/{color}]")
        console.print(f"[bold]Stocks Scanned:[/bold] {result.get('scanned_stocks', 0)}")
        console.print(f"[bold]Opportunities Found:[/bold] {result.get('opportunities_found', 0)}")
    
    opportunities = result.get("top_opportunities", [])
    
    if console and opportunities:
        table = Table(title="Top Opportunities", box=box.ROUNDED)
        table.add_column("Symbol", style="cyan")
        table.add_column("Signal", justify="center")
        table.add_column("Confidence", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Target", justify="right")
        table.add_column("Stop", justify="right")
        
        for opp in opportunities:
            signal = opp.get("signal", "HOLD")
            if signal == "STRONG_BUY":
                signal_color = "bold green"
            elif signal == "BUY":
                signal_color = "green"
            elif signal == "WATCHLIST":
                signal_color = "cyan"
            else:
                signal_color = "yellow"
            
            table.add_row(
                opp.get("symbol", ""),
                f"[{signal_color}]{signal}[/{signal_color}]",
                f"{opp.get('confidence', 0):.0f}%",
                f"₹{opp.get('entry_price', 0):.2f}",
                f"₹{opp.get('target_1', 0):.2f}",
                f"₹{opp.get('stop_loss', 0):.2f}",
            )
        
        console.print(table)
    elif console:
        console.print("\n[yellow]No opportunities found matching criteria.[/yellow]")


@app.command()
def eod():
    """End of day analysis."""
    agent = TradingAgent()
    
    print_header("END OF DAY SUMMARY")
    
    if console:
        with console.status("[bold green]Running EOD analysis..."):
            result = agent.run_eod_analysis()
    else:
        result = agent.run_eod_analysis()
    
    # Market summary
    market = result.get("market_summary", {})
    regime = market.get("regime", {})
    
    if console:
        console.print(f"\n[bold]Market Regime:[/bold] {regime.get('regime', 'UNKNOWN')}")
        console.print(f"[bold]Volatility:[/bold] {regime.get('volatility', 'NORMAL')}")
    
    # Sector performance
    sectors = result.get("sector_performance", {})
    if console and sectors.get("sector_performance"):
        table = Table(title="Sector Performance", box=box.SIMPLE)
        table.add_column("Sector", style="cyan")
        table.add_column("Change", justify="right")
        
        for sector, change in list(sectors["sector_performance"].items())[:6]:
            color = "green" if change > 0 else "red"
            table.add_row(sector, f"[{color}]{change:+.2f}%[/{color}]")
        
        console.print(table)
    
    # Tomorrow's watchlist
    watchlist = result.get("tomorrows_watchlist", [])
    if console and watchlist:
        console.print("\n[bold cyan]Tomorrow's Watchlist:[/bold cyan]")
        for item in watchlist:
            console.print(f"  📌 {item['symbol']} - {item['signal']} ({item['confidence']:.0f}%)")


@app.command()
def add(symbol: str, quantity: int, price: float, stop: float = None, target: float = None):
    """Add position to portfolio."""
    agent = TradingAgent()
    
    result = agent.add_to_portfolio(
        symbol=symbol.upper(),
        quantity=quantity,
        price=price,
        stop_loss=stop,
        target=target,
    )
    
    if console:
        console.print(f"[green]Added {quantity} shares of {symbol.upper()} at ₹{price}[/green]")
        if stop:
            console.print(f"Stop Loss: ₹{stop}")
        if target:
            console.print(f"Target: ₹{target}")


@app.command()
def remove(symbol: str, quantity: int, price: float):
    """Remove/sell position from portfolio."""
    agent = TradingAgent()
    
    result = agent.close_portfolio_position(
        symbol=symbol.upper(),
        quantity=quantity,
        price=price,
    )
    
    if "error" not in result:
        pnl = result.get("pnl", 0)
        pnl_pct = result.get("pnl_pct", 0)
        color = "green" if pnl >= 0 else "red"
        
        if console:
            console.print(f"[{color}]Sold {quantity} shares of {symbol.upper()} at ₹{price}[/{color}]")
            console.print(f"[{color}]P&L: ₹{pnl:,.0f} ({pnl_pct:+.2f}%)[/{color}]")
    else:
        if console:
            console.print(f"[red]Error: {result['error']}[/red]")


@app.command()
def quote(symbol: str):
    """Get quick quote for a stock."""
    agent = TradingAgent()
    
    result = agent.get_quick_quote(symbol.upper())
    
    if console:
        price = result.get("price", 0)
        change = result.get("change_percent", 0) or 0
        color = "green" if change >= 0 else "red"
        
        console.print(f"\n[bold cyan]{symbol.upper()}[/bold cyan]")
        console.print(f"Price: ₹{price:.2f}")
        console.print(f"Change: [{color}]{change:+.2f}%[/{color}]")
        console.print(f"High: ₹{result.get('high', 0):.2f} | Low: ₹{result.get('low', 0):.2f}")
        console.print(f"52W High: ₹{result.get('fifty_two_week_high', 0):.2f}")
        console.print(f"52W Low: ₹{result.get('fifty_two_week_low', 0):.2f}")


@app.command()
def performance():
    """Show trading performance."""
    agent = TradingAgent()
    
    print_header("TRADING PERFORMANCE")
    
    result = agent.get_trading_performance()
    perf = result.get("performance", {})
    
    if console and perf.get("total_trades", 0) > 0:
        console.print(f"\n[bold]Total Trades:[/bold] {perf.get('total_trades', 0)}")
        console.print(f"[bold]Win Rate:[/bold] {perf.get('win_rate', 0):.1f}%")
        
        pnl = perf.get("total_pnl", 0)
        color = "green" if pnl >= 0 else "red"
        console.print(f"[bold]Total P&L:[/bold] [{color}]₹{pnl:,.0f}[/{color}]")
        
        console.print(f"[bold]Avg Win:[/bold] {perf.get('avg_win_pct', 0):+.2f}%")
        console.print(f"[bold]Avg Loss:[/bold] {perf.get('avg_loss_pct', 0):.2f}%")
        console.print(f"[bold]Profit Factor:[/bold] {perf.get('profit_factor', 0):.2f}")
        
        # Suggestions
        suggestions = result.get("suggestions", [])
        if suggestions:
            console.print("\n[bold yellow]Suggestions:[/bold yellow]")
            for s in suggestions:
                console.print(f"  💡 {s}")
    else:
        console.print("[yellow]No trades to analyze yet.[/yellow]")


@app.command()
def chat(question: str = typer.Argument(None)):
    """Chat with AI about trading. Ask any question!"""
    from src.ai_layer.llm_reasoner import get_reasoner
    
    ai = get_reasoner()
    
    if question:
        # Single question mode
        if console:
            with console.status("[cyan]Thinking...[/cyan]"):
                response = ai.chat(question)
            console.print(Panel(response, title="🤖 AI", box=box.ROUNDED))
    else:
        # Interactive chat mode
        if console:
            console.print("\n[bold cyan]🤖 AI Trading Assistant[/bold cyan]")
            console.print("[dim]Ask anything about stocks, trading, or markets. Type 'quit' to exit.[/dim]\n")
            
            while True:
                try:
                    q = input("You: ").strip()
                    if q.lower() in ["quit", "exit", "q", ""]:
                        break
                    
                    with console.status("[cyan]Thinking...[/cyan]"):
                        response = ai.chat(q)
                    console.print(Panel(response, title="🤖 AI", box=box.ROUNDED))
                except KeyboardInterrupt:
                    break
            
            console.print("[yellow]Goodbye![/yellow]")


@app.command()
def predict(symbol: str):
    """Get AI price prediction for a stock."""
    from src.ai_layer.llm_reasoner import get_reasoner
    
    agent = TradingAgent()
    ai = get_reasoner()
    
    print_header(f"AI PREDICTION: {symbol.upper()}")
    
    if console:
        with console.status(f"[cyan]Analyzing {symbol}...[/cyan]"):
            result = agent.research_stock(symbol.upper())
            prediction = ai.predict_price_movement(symbol.upper(), result.get("signal", {}))
        
        console.print(Panel(prediction, title=f"🔮 {symbol.upper()} Prediction", box=box.ROUNDED))


@app.command()
def etf():
    """Show available ETFs for trading."""
    print_header("AVAILABLE ETFs")
    
    if console:
        console.print("\n[bold cyan]Gold ETFs:[/bold cyan]")
        console.print("  GOLDBEES  - Nippon Gold ETF")
        console.print("  GOLDCASE  - ICICI Gold ETF")
        
        console.print("\n[bold cyan]Silver ETFs:[/bold cyan]")
        console.print("  SILVERBEES - Nippon Silver ETF")
        console.print("  SILVERETF  - ICICI Silver ETF")
        
        console.print("\n[bold cyan]Index ETFs:[/bold cyan]")
        console.print("  NIFTYBEES  - Nifty 50 ETF")
        console.print("  BANKBEES   - Bank Nifty ETF")
        console.print("  ITBEES     - IT Index ETF")
        console.print("  JUNIORBEES - Nifty Next 50 ETF")
        
        console.print("\n[bold cyan]Sector ETFs:[/bold cyan]")
        console.print("  PHARMABEES  - Pharma Index")
        console.print("  PSUBNKBEES  - PSU Bank Index")
        console.print("  INFRABEES   - Infrastructure")
        
        console.print("\n[bold cyan]International:[/bold cyan]")
        console.print("  N100   - Nasdaq 100 ETF")
        console.print("  MAFANG - FAANG Stocks ETF")
        
        console.print("\n[dim]Use 'python run.py research GOLDBEES' to analyze any ETF[/dim]")


if __name__ == "__main__":
    if RICH_AVAILABLE:
        app()
    else:
        print("Please install rich and typer: pip install rich typer")
