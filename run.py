#!/usr/bin/env python3
"""
Simple runner script for the Indian Market Trading Agent.

Usage:
    python run.py                  # Interactive menu
    python run.py morning          # Morning briefing
    python run.py scan             # Scan for opportunities
    python run.py research SYMBOL  # Research a stock
    python run.py portfolio        # Portfolio analysis
"""

import sys
import warnings
import logging
from pathlib import Path

# Suppress all warnings and noisy logs
warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main entry point."""
    from src.main import app
    app()


def interactive():
    """Interactive AI-powered mode."""
    try:
        from rich.console import Console
        from rich.prompt import Prompt
        from rich.panel import Panel
        from rich.markdown import Markdown
        console = Console()
    except ImportError:
        print("Please install rich: pip install rich")
        return
    
    from src.agent import TradingAgent
    from src.ai_layer.llm_reasoner import get_reasoner
    
    agent = TradingAgent()
    ai = get_reasoner()
    
    console.print(Panel.fit(
        "[bold cyan]🤖 Indian Market Trading Agent[/bold cyan]\n"
        "[dim]AI-Powered Research & Analysis[/dim]",
        border_style="cyan"
    ))
    console.print("\n[bold]Commands:[/bold] morning, scan, research, quote, portfolio, chat, etf, help, quit")
    console.print("[dim]Or just type a question to chat with AI![/dim]\n")
    
    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]").strip()
            
            if not user_input:
                continue
            
            command = user_input.lower().split()[0]
            
            if command in ["quit", "exit", "q"]:
                console.print("[yellow]Goodbye! Happy trading! 📈[/yellow]")
                break
            
            elif command == "help":
                console.print("""
[bold cyan]📋 Commands:[/bold cyan]
  [green]morning[/green]     - Morning market briefing
  [green]scan[/green]        - Scan for buy opportunities  
  [green]scan etf[/green]    - Scan ETFs (Gold, Silver, Index)
  [green]scan all[/green]    - Scan full market (100+ stocks)
  [green]research[/green]    - Deep research on a stock
  [green]quote[/green]       - Quick price quote
  [green]portfolio[/green]   - Portfolio analysis
  [green]etf[/green]         - List available ETFs
  [green]predict[/green]     - AI price prediction
  [green]chat[/green]        - Chat with AI about trading
  [green]help[/green]        - Show this help
  [green]quit[/green]        - Exit

[bold cyan]💬 Or just ask anything![/bold cyan]
  "Should I buy Reliance?"
  "What's happening with gold?"
  "Best IT stocks for swing trading?"
  "Explain MACD crossover"
                """)
            
            elif command == "morning":
                with console.status("[cyan]Generating morning briefing...[/cyan]"):
                    briefing = agent.run_morning_briefing()
                session = briefing.get("session", {})
                console.print(f"\n[bold]Market:[/bold] {session.get('status', 'UNKNOWN')}")
                
                # AI briefing
                ai_brief = briefing.get("briefing", "")
                if ai_brief:
                    console.print(Panel(ai_brief, title="🤖 AI Analysis", border_style="blue"))
            
            elif command == "scan":
                parts = user_input.lower().split()
                scan_type = parts[1] if len(parts) > 1 else "default"
                
                if scan_type == "etf":
                    console.print("[cyan]Scanning ETFs...[/cyan]")
                    from config.settings import ETF_LIST
                    result = agent.scan_opportunities(n=10, universe=ETF_LIST)
                elif scan_type == "all":
                    console.print("[cyan]Scanning full market (this may take a while)...[/cyan]")
                    from config.settings import FULL_UNIVERSE
                    result = agent.scan_opportunities(n=15, universe=FULL_UNIVERSE[:50])
                else:
                    with console.status("[cyan]Scanning Nifty stocks...[/cyan]"):
                        result = agent.scan_opportunities(n=10)
                
                opps = result.get("top_opportunities", [])
                if opps:
                    console.print(f"\n[bold green]Found {len(opps)} opportunities:[/bold green]")
                    for opp in opps:
                        sig = opp.get('signal', 'HOLD')
                        color = "green" if "BUY" in sig else "cyan" if sig == "WATCHLIST" else "yellow"
                        console.print(f"  [{color}]{opp['symbol']:12}[/{color}] {sig:12} {opp['confidence']:.0f}% | Entry: ₹{opp.get('entry_price', 0):.2f}")
                else:
                    console.print("[yellow]No strong opportunities found. Try 'scan all' or 'scan etf'[/yellow]")
            
            elif command == "research":
                parts = user_input.split()
                symbol = parts[1].upper() if len(parts) > 1 else Prompt.ask("Enter symbol")
                
                with console.status(f"[cyan]Running advanced multi-timeframe analysis for {symbol}...[/cyan]"):
                    result = agent.research_stock(symbol.upper(), deep=True)
                
                signal = result.get("signal", {})
                advanced = result.get("advanced", {})
                
                overall_score = advanced.get("overall_score", signal.get("combined_score", 50))
                sig = advanced.get("signal_strength", signal.get('signal', 'HOLD'))
                conf = signal.get('confidence', 50)
                
                sig_color = "green" if "BUY" in sig else "red" if "SELL" in sig else "yellow"
                
                # Header
                console.print(f"\n[bold cyan]{'═'*50}[/bold cyan]")
                console.print(f"[bold]{symbol}[/bold] - Score: {overall_score:.0f}/100")
                console.print(f"[bold cyan]{'═'*50}[/bold cyan]")
                
                # Signal
                console.print(f"\n[bold]Signal:[/bold] [{sig_color}]{sig}[/{sig_color}] ({conf:.0f}% confidence)")
                
                # Key levels
                levels = advanced.get("key_levels", {})
                console.print(f"[bold]Entry:[/bold] ₹{levels.get('current', signal.get('entry_price', 0)):.2f}")
                console.print(f"[bold]Stop:[/bold] ₹{signal.get('stop_loss', 0):.2f} | [bold]Target:[/bold] ₹{signal.get('target_1', 0):.2f}")
                
                # Timeframe summary
                tf = advanced.get("timeframes", {})
                if tf:
                    daily = tf.get("daily", {}).get("trend", "N/A")
                    weekly = tf.get("weekly", {}).get("trend", "N/A")
                    alignment = tf.get("alignment", {}).get("status", "N/A")
                    console.print(f"\n[bold]Timeframes:[/bold] Daily={daily} | Weekly={weekly}")
                    console.print(f"[bold]Alignment:[/bold] {alignment}")
                
                # Relative strength
                rs = advanced.get("relative_strength", {})
                if rs:
                    rs_score = rs.get("rs_score", 50)
                    vs_market = rs.get("vs_market", "N/A")
                    console.print(f"[bold]RS vs Nifty:[/bold] {rs_score:.0f}/100 ({vs_market})")
                
                # Risk
                risk = advanced.get("risk", {})
                if risk:
                    console.print(f"[bold]Risk Level:[/bold] {risk.get('risk_level', 'N/A')} (Vol: {risk.get('volatility_20d', 0):.1f}%)")
                
                # AI explanation
                ai_analysis = result.get("ai_analysis", "")
                if ai_analysis:
                    console.print(Panel(ai_analysis, title="🤖 AI Analysis", border_style="blue"))
            
            elif command == "predict":
                parts = user_input.split()
                symbol = parts[1].upper() if len(parts) > 1 else Prompt.ask("Enter symbol")
                
                with console.status(f"[cyan]AI analyzing {symbol}...[/cyan]"):
                    result = agent.research_stock(symbol)
                    prediction = ai.predict_price_movement(symbol, result.get("signal", {}))
                
                console.print(Panel(prediction, title=f"🔮 AI Prediction: {symbol}", border_style="magenta"))
            
            elif command == "quote":
                parts = user_input.split()
                symbol = parts[1].upper() if len(parts) > 1 else Prompt.ask("Enter symbol")
                
                result = agent.get_quick_quote(symbol.upper())
                price = result.get('price', 0)
                change = result.get('change_percent', 0) or 0
                color = "green" if change >= 0 else "red"
                
                console.print(f"\n[bold]{symbol}[/bold]: ₹{price:.2f} [{color}]{change:+.2f}%[/{color}]")
                console.print(f"High: ₹{result.get('high', 0):.2f} | Low: ₹{result.get('low', 0):.2f}")
            
            elif command == "portfolio":
                with console.status("[cyan]Analyzing portfolio...[/cyan]"):
                    analysis = agent.analyze_portfolio()
                summary = analysis.get("summary", {})
                console.print(f"\n[bold]Holdings:[/bold] {summary.get('num_holdings', 0)}")
                console.print(f"[bold]Value:[/bold] ₹{summary.get('portfolio_value', 0):,.0f}")
                pnl = summary.get('unrealized_pnl', 0)
                color = "green" if pnl >= 0 else "red"
                console.print(f"[bold]P&L:[/bold] [{color}]₹{pnl:,.0f}[/{color}]")
            
            elif command == "etf":
                from config.settings import ETFS
                console.print("\n[bold cyan]Available ETFs:[/bold cyan]")
                console.print("\n[bold]Gold:[/bold] GOLDBEES, GOLDCASE")
                console.print("[bold]Silver:[/bold] SILVERBEES, SILVERETF")
                console.print("[bold]Index:[/bold] NIFTYBEES, BANKBEES, ITBEES, JUNIORBEES")
                console.print("[bold]Sector:[/bold] PHARMABEES, PSUBNKBEES, INFRABEES")
                console.print("[bold]International:[/bold] N100 (Nasdaq), MAFANG (FAANG)")
                console.print("\n[dim]Use 'research GOLDBEES' to analyze an ETF[/dim]")
            
            elif command == "chat":
                console.print("[cyan]Chat mode! Type your questions. Type 'back' to return.[/cyan]")
                while True:
                    question = Prompt.ask("[bold blue]Ask[/bold blue]")
                    if question.lower() in ["back", "exit", "quit"]:
                        break
                    response = ai.chat(question)
                    console.print(Panel(response, title="🤖 AI", border_style="blue"))
            
            else:
                # Treat as natural language query - chat with AI
                with console.status("[cyan]Thinking...[/cyan]"):
                    response = ai.chat(user_input)
                console.print(Panel(response, title="🤖 AI", border_style="blue"))
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Type 'quit' to exit[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run CLI
        main()
    else:
        # Interactive mode
        interactive()
