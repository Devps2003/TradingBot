"""
🚀 Indian Market Trading Agent - Professional Dashboard v2.0

A stunning, money-making trading interface with full portfolio context.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import sys
import os
import hashlib
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="Trading Agent 🚀",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# 🔐 PASSWORD PROTECTION - Only YOU can access
# ============================================================================
# Set your password via environment variable or change the default below
# To set: export TRADING_BOT_PASSWORD="your-secret-password"

def check_password():
    """Returns `True` if the user has the correct password."""
    
    # Get password from env or use default
    correct_password = os.getenv("TRADING_BOT_PASSWORD", "trader123")
    
    # Initialize session state
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = None
    
    # Already logged in
    if st.session_state["password_correct"] == True:
        return True
    
    # Show login page
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(180deg, #0a0a1a 0%, #0f0f2a 100%); }
        .login-box {
            max-width: 400px;
            margin: 100px auto;
            padding: 40px;
            background: rgba(255,255,255,0.03);
            border-radius: 24px;
            border: 1px solid rgba(255,255,255,0.1);
            text-align: center;
        }
    </style>
    <div class="login-box">
        <div style="font-size: 4rem; margin-bottom: 20px;">🔐</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: #fff; margin-bottom: 10px;">
            Trading Agent
        </div>
        <div style="color: rgba(255,255,255,0.5); margin-bottom: 30px;">
            Enter your password to continue
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        password_input = st.text_input(
            "Password", type="password", 
            placeholder="Enter password...",
            key="password_input"
        )
        
        if st.button("Login", use_container_width=True):
            if password_input and hashlib.sha256(password_input.encode()).hexdigest() == \
               hashlib.sha256(correct_password.encode()).hexdigest():
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("😕 Incorrect password")
        
        if st.session_state["password_correct"] == False:
            st.error("😕 Incorrect password")
        
        st.caption("Default: `trader123`")
    
    return False

# Check password before showing anything
if not check_password():
    st.stop()

# Beautiful dark theme CSS
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(180deg, #0a0a1a 0%, #0f0f2a 50%, #0a0a1a 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12122a 0%, #0a0a1a 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Glass cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 12px 0;
    }
    
    .glass-card:hover {
        border: 1px solid rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    
    /* Gradient text */
    .gradient-text {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    
    .gradient-text-green {
        background: linear-gradient(135deg, #00ff88 0%, #00d4aa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .gradient-text-red {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff4757 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Signal cards */
    .signal-buy {
        background: linear-gradient(135deg, rgba(0,255,136,0.15) 0%, rgba(0,212,170,0.1) 100%);
        border: 2px solid rgba(0,255,136,0.4);
        border-radius: 24px;
        padding: 30px;
        text-align: center;
    }
    
    .signal-sell {
        background: linear-gradient(135deg, rgba(255,71,87,0.15) 0%, rgba(255,107,107,0.1) 100%);
        border: 2px solid rgba(255,71,87,0.4);
        border-radius: 24px;
        padding: 30px;
        text-align: center;
    }
    
    .signal-hold {
        background: linear-gradient(135deg, rgba(255,217,61,0.15) 0%, rgba(255,193,7,0.1) 100%);
        border: 2px solid rgba(255,217,61,0.4);
        border-radius: 24px;
        padding: 30px;
        text-align: center;
    }
    
    /* Metrics */
    .big-metric {
        font-size: 2.5rem;
        font-weight: 700;
        color: #fff;
        margin: 5px 0;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }
    
    .positive { color: #00ff88 !important; }
    .negative { color: #ff4757 !important; }
    
    /* Holding cards */
    .holding-card {
        background: rgba(255,255,255,0.02);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .holding-card.profit {
        border-left-color: #00ff88;
    }
    
    .holding-card.loss {
        border-left-color: #ff4757;
    }
    
    /* Alert badges */
    .alert-badge {
        background: rgba(255,71,87,0.2);
        color: #ff4757;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        margin: 2px;
        display: inline-block;
    }
    
    .alert-badge.success {
        background: rgba(0,255,136,0.2);
        color: #00ff88;
    }
    
    /* Progress bars */
    .progress-bar {
        height: 8px;
        background: rgba(255,255,255,0.1);
        border-radius: 4px;
        overflow: hidden;
        margin: 8px 0;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Action buttons */
    .action-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .action-btn:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(102,126,234,0.4);
    }
    
    /* Chat bubbles */
    .ai-bubble {
        background: linear-gradient(135deg, rgba(102,126,234,0.2) 0%, rgba(118,75,162,0.15) 100%);
        border-radius: 20px 20px 20px 4px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid rgba(102,126,234,0.3);
    }
    
    .user-bubble {
        background: rgba(255,255,255,0.05);
        border-radius: 20px 20px 4px 20px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: right;
    }
    
    /* Tabs override */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.02);
        padding: 8px;
        border-radius: 16px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: rgba(255,255,255,0.6);
        padding: 12px 24px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .pulse {
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-in {
        animation: slideIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)


# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_symbol" not in st.session_state:
    st.session_state.current_symbol = "RELIANCE"


# Helper functions
@st.cache_resource
def get_smart_portfolio():
    from src.portfolio.smart_portfolio import SmartPortfolio
    return SmartPortfolio()


@st.cache_resource
def get_agent():
    from src.agent import TradingAgent
    return TradingAgent()


@st.cache_resource
def get_ai():
    from src.ai_layer.llm_reasoner import get_reasoner
    return get_reasoner()


@st.cache_data(ttl=300)
def get_stock_data(symbol: str):
    agent = get_agent()
    return agent.research_stock(symbol, deep=True)


@st.cache_data(ttl=300)
def get_historical(symbol: str, period: str = "6mo"):
    from src.data_fetchers.price_fetcher import PriceFetcher
    return PriceFetcher().get_historical_data(symbol, period=period)


@st.cache_data(ttl=60)
def get_market_cues():
    from src.data_fetchers.global_fetcher import GlobalFetcher
    return GlobalFetcher().get_market_cues_for_india()


def create_chart(df, symbol):
    if df.empty:
        return None
    
    fig = go.Figure()
    
    # Candlestick
    colors = ['#00ff88' if c >= o else '#ff4757' 
              for o, c in zip(df['open'], df['close'])]
    
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4757',
        increasing_fillcolor='#00ff88',
        decreasing_fillcolor='#ff4757',
    ))
    
    # EMAs
    if len(df) >= 20:
        ema20 = df['close'].ewm(span=20).mean()
        fig.add_trace(go.Scatter(x=df['date'], y=ema20, name='EMA 20',
                                  line=dict(color='#ffd93d', width=1.5)))
    if len(df) >= 50:
        ema50 = df['close'].ewm(span=50).mean()
        fig.add_trace(go.Scatter(x=df['date'], y=ema50, name='EMA 50',
                                  line=dict(color='#667eea', width=1.5)))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_rangeslider_visible=False,
        height=450,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=1.1),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
    )
    
    return fig


def render_pnl_display(pnl, pnl_pct):
    color_class = "positive" if pnl >= 0 else "negative"
    arrow = "↑" if pnl >= 0 else "↓"
    return f'<span class="{color_class}">{arrow} ₹{abs(pnl):,.0f} ({pnl_pct:+.2f}%)</span>'


# Sidebar
with st.sidebar:
    st.markdown('<div class="gradient-text" style="font-size: 1.8rem;">📈 TradingBot</div>', unsafe_allow_html=True)
    st.markdown('<span style="color: rgba(255,255,255,0.5);">AI-Powered • Portfolio Aware</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "💼 My Portfolio", "🔍 Research", "📊 Scanner", "🤖 AI Advisor", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Portfolio summary in sidebar
    portfolio = get_smart_portfolio()
    live_portfolio = portfolio.get_live_portfolio()
    
    total_pnl = live_portfolio.get("total_pnl", 0)
    pnl_color = "#00ff88" if total_pnl >= 0 else "#ff4757"
    
    st.markdown(f"""
    <div class="glass-card" style="padding: 16px;">
        <div class="metric-label">Portfolio P&L</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: {pnl_color};">
            {"+" if total_pnl >= 0 else ""}₹{total_pnl:,.0f}
        </div>
        <div style="color: {pnl_color}; font-size: 0.9rem;">
            {live_portfolio.get("total_pnl_pct", 0):+.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Market status
    market_cues = get_market_cues()
    sentiment = market_cues.get("overall_sentiment", "NEUTRAL")
    sent_color = "#00ff88" if sentiment == "BULLISH" else "#ff4757" if sentiment == "BEARISH" else "#ffd93d"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 12px; margin-top: 15px;">
        <div class="metric-label">Market</div>
        <div style="color: {sent_color}; font-weight: 600;">{sentiment}</div>
    </div>
    """, unsafe_allow_html=True)


# Main content
if page == "🏠 Dashboard":
    # Header
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="gradient-text slide-in">Good ' + 
                    ('Morning' if datetime.now().hour < 12 else 'Afternoon' if datetime.now().hour < 17 else 'Evening') + 
                    '! 👋</div>', unsafe_allow_html=True)
        st.markdown(f'<span style="color: rgba(255,255,255,0.6);">{datetime.now().strftime("%A, %B %d, %Y")}</span>', 
                    unsafe_allow_html=True)
    
    with col2:
        # Quick stats
        st.markdown(f"""
        <div style="text-align: right;">
            <span class="metric-label">Portfolio Value</span><br>
            <span style="font-size: 1.8rem; font-weight: 700; color: #fff;">
                ₹{live_portfolio.get("portfolio_value", 0):,.0f}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        indian = market_cues.get("indian_markets", {})
        nifty = indian.get("nifty", {})
        nifty_change = nifty.get("change_pct", 0) or 0
        color = "positive" if nifty_change >= 0 else "negative"
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Nifty 50</div>
            <div class="big-metric">{nifty.get("price", 0):,.0f}</div>
            <div class="{color}">{nifty_change:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        sensex = indian.get("sensex", {})
        sensex_change = sensex.get("change_pct", 0) or 0
        color = "positive" if sensex_change >= 0 else "negative"
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Sensex</div>
            <div class="big-metric">{sensex.get("price", 0):,.0f}</div>
            <div class="{color}">{sensex_change:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">My Holdings</div>
            <div class="big-metric">{live_portfolio.get("num_holdings", 0)}</div>
            <div style="color: rgba(255,255,255,0.5);">Active positions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        exp_gap = market_cues.get("expected_gap", "FLAT")
        gap_color = "#00ff88" if exp_gap == "GAP_UP" else "#ff4757" if exp_gap == "GAP_DOWN" else "#ffd93d"
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Expected Open</div>
            <div class="big-metric" style="color: {gap_color}; font-size: 1.5rem;">{exp_gap}</div>
            <div style="color: rgba(255,255,255,0.5);">{market_cues.get("expected_nifty_impact", 0):+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Holdings with signals
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 My Holdings - Live Status")
        
        holdings = live_portfolio.get("holdings", [])
        
        if holdings:
            for h in holdings:
                pnl = h.get("pnl", 0)
                pnl_pct = h.get("pnl_pct", 0)
                card_class = "profit" if pnl >= 0 else "loss"
                pnl_color = "#00ff88" if pnl >= 0 else "#ff4757"
                
                alerts_html = ""
                for alert in h.get("alerts", []):
                    badge_class = "success" if "TARGET" in alert or "profit" in alert.lower() else ""
                    alerts_html += f'<span class="alert-badge {badge_class}">{alert}</span> '
                
                st.markdown(f"""
                <div class="holding-card {card_class}">
                    <div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #fff;">{h["symbol"]}</div>
                        <div style="color: rgba(255,255,255,0.5);">{h["quantity"]} shares @ ₹{h["avg_price"]:.2f}</div>
                        <div style="margin-top: 5px;">{alerts_html}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.3rem; font-weight: 600; color: {pnl_color};">
                            {"+" if pnl >= 0 else ""}₹{pnl:,.0f}
                        </div>
                        <div style="color: {pnl_color};">{pnl_pct:+.2f}%</div>
                        <div style="color: rgba(255,255,255,0.4); font-size: 0.8rem;">
                            CMP: ₹{h["current_price"]:.2f}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No holdings yet. Start by adding a position!")
    
    with col2:
        st.markdown("### 🎯 Quick Actions")
        
        # Add position form
        with st.expander("➕ Add New Position", expanded=False):
            new_sym = st.text_input("Symbol", placeholder="RELIANCE")
            new_qty = st.number_input("Quantity", min_value=1, value=10)
            new_price = st.number_input("Buy Price", min_value=0.01, value=100.0, step=0.05)
            new_sl = st.number_input("Stop Loss (optional)", min_value=0.0, value=0.0)
            new_tgt = st.number_input("Target (optional)", min_value=0.0, value=0.0)
            
            if st.button("Buy", use_container_width=True):
                if new_sym and new_qty and new_price:
                    result = portfolio.buy(
                        new_sym.upper(), new_qty, new_price,
                        stop_loss=new_sl if new_sl > 0 else None,
                        target=new_tgt if new_tgt > 0 else None
                    )
                    if result["success"]:
                        st.success(result["message"])
                        st.rerun()
                    else:
                        st.error(result.get("error"))
        
        # AI Portfolio Advice
        st.markdown("### 🤖 AI Advice")
        if st.button("Get Portfolio Advice", use_container_width=True):
            with st.spinner("AI analyzing your portfolio..."):
                ai = get_ai()
                context = portfolio.get_context_for_ai()
                advice = ai.suggest_portfolio_action(context, market_cues)
            st.markdown(f'<div class="ai-bubble">{advice}</div>', unsafe_allow_html=True)


elif page == "💼 My Portfolio":
    st.markdown('<div class="gradient-text">My Portfolio</div>', unsafe_allow_html=True)
    
    portfolio = get_smart_portfolio()
    live = portfolio.get_live_portfolio()
    stats = portfolio.get_performance_stats()
    
    # Top stats
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Total Value</div>
            <div class="big-metric" style="font-size: 1.8rem;">₹{live["portfolio_value"]:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        color = "positive" if live["total_pnl"] >= 0 else "negative"
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Total P&L</div>
            <div class="big-metric {color}" style="font-size: 1.8rem;">₹{live["total_pnl"]:+,.0f}</div>
            <div class="{color}">{live["total_pnl_pct"]:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Cash</div>
            <div class="big-metric" style="font-size: 1.8rem;">₹{live["cash"]:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        win_rate = stats.get("win_rate", 0)
        wr_color = "#00ff88" if win_rate >= 50 else "#ff4757"
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Win Rate</div>
            <div class="big-metric" style="font-size: 1.8rem; color: {wr_color};">{win_rate:.0f}%</div>
            <div style="color: rgba(255,255,255,0.5);">{stats.get("total_trades", 0)} trades</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        net = stats.get("net_pnl", 0)
        net_color = "#00ff88" if net >= 0 else "#ff4757"
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Realized P&L</div>
            <div class="big-metric" style="font-size: 1.8rem; color: {net_color};">₹{net:+,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Holdings with signals
    tab1, tab2, tab3 = st.tabs(["📊 Holdings", "📈 Signals", "📜 History"])
    
    with tab1:
        holdings = live.get("holdings", [])
        if holdings:
            for h in holdings:
                pnl = h["pnl"]
                pnl_color = "#00ff88" if pnl >= 0 else "#ff4757"
                
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div style="padding: 10px 0;">
                        <div style="font-size: 1.2rem; font-weight: 700;">{h["symbol"]}</div>
                        <div style="color: rgba(255,255,255,0.5);">{h["quantity"]} @ ₹{h["avg_price"]:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="padding: 10px 0;">
                        <div style="color: rgba(255,255,255,0.5);">Current</div>
                        <div style="font-weight: 600;">₹{h["current_price"]:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style="padding: 10px 0;">
                        <div style="color: rgba(255,255,255,0.5);">P&L</div>
                        <div style="font-weight: 600; color: {pnl_color};">₹{pnl:+,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    if st.button("Sell", key=f"sell_{h['symbol']}"):
                        st.session_state[f"selling_{h['symbol']}"] = True
                
                # Sell modal
                if st.session_state.get(f"selling_{h['symbol']}"):
                    with st.expander(f"Sell {h['symbol']}", expanded=True):
                        sell_qty = st.number_input("Quantity", 1, h["quantity"], h["quantity"], key=f"sq_{h['symbol']}")
                        sell_price = st.number_input("Price", value=h["current_price"], key=f"sp_{h['symbol']}")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("Confirm Sell", key=f"cs_{h['symbol']}"):
                                result = portfolio.sell(h["symbol"], sell_qty, sell_price)
                                if result["success"]:
                                    st.success(result["message"])
                                    st.session_state[f"selling_{h['symbol']}"] = False
                                    st.rerun()
                        with col_b:
                            if st.button("Cancel", key=f"cc_{h['symbol']}"):
                                st.session_state[f"selling_{h['symbol']}"] = False
                                st.rerun()
                
                st.markdown("---")
        else:
            st.info("No holdings. Add your first position!")
    
    with tab2:
        st.markdown("### AI-Powered Signals for Your Holdings")
        
        if st.button("🔄 Refresh Signals"):
            st.cache_data.clear()
        
        signals = portfolio.get_signals_for_holdings()
        
        for sig in signals:
            if "error" in sig:
                continue
            
            action = sig.get("action", "HOLD")
            action_colors = {
                "SELL": "#ff4757", "EXIT": "#ff4757",
                "BUY": "#00ff88", "ADD": "#00ff88",
                "BOOK_PROFIT": "#ffd93d",
                "HOLD": "#888",
            }
            action_color = action_colors.get(action, "#888")
            
            st.markdown(f"""
            <div class="glass-card" style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 1.2rem; font-weight: 700;">{sig["symbol"]}</div>
                    <div style="color: rgba(255,255,255,0.5);">
                        P&L: <span style="color: {"#00ff88" if sig["pnl_pct"] >= 0 else "#ff4757"};">{sig["pnl_pct"]:+.2f}%</span>
                    </div>
                </div>
                <div style="text-align: center;">
                    <div style="color: {action_color}; font-size: 1.3rem; font-weight: 700;">{action}</div>
                    <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">{sig.get("reason", "")}</div>
                </div>
                <div style="text-align: right;">
                    <div style="color: rgba(255,255,255,0.5);">Signal: {sig.get("signal", "N/A")}</div>
                    <div style="color: rgba(255,255,255,0.5);">Confidence: {sig.get("confidence", 50):.0f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### Trade History")
        
        history = portfolio.history
        if history:
            for trade in reversed(history[-20:]):
                t_type = trade.get("type", "")
                t_color = "#00ff88" if t_type == "BUY" else "#ff4757"
                
                pnl = trade.get("pnl", 0)
                pnl_html = ""
                if t_type == "SELL":
                    pnl_color = "#00ff88" if pnl >= 0 else "#ff4757"
                    pnl_html = f'<span style="color: {pnl_color};">P&L: ₹{pnl:+,.0f}</span>'
                
                st.markdown(f"""
                <div style="padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <span style="color: {t_color}; font-weight: 600;">{t_type}</span>
                            <span style="color: #fff; font-weight: 600; margin-left: 10px;">{trade.get("symbol", "")}</span>
                            <span style="color: rgba(255,255,255,0.5); margin-left: 10px;">
                                {trade.get("quantity", 0)} @ ₹{trade.get("price", 0):.2f}
                            </span>
                        </div>
                        <div>
                            {pnl_html}
                            <span style="color: rgba(255,255,255,0.3); margin-left: 15px; font-size: 0.85rem;">
                                {trade.get("date", "")[:10]}
                            </span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No trade history yet.")


elif page == "🔍 Research":
    st.markdown('<div class="gradient-text">Stock Research</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    with col1:
        symbol = st.text_input("", value=st.session_state.current_symbol, placeholder="Enter symbol (e.g., RELIANCE, GOLDBEES)")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze = st.button("🔍 Analyze", use_container_width=True)
    
    if analyze and symbol:
        st.session_state.current_symbol = symbol.upper()
    
    if st.session_state.current_symbol:
        with st.spinner(f"Analyzing {st.session_state.current_symbol}..."):
            data = get_stock_data(st.session_state.current_symbol)
            df = get_historical(st.session_state.current_symbol)
        
        signal = data.get("signal", {})
        advanced = data.get("advanced", {})
        
        # Signal display
        sig = advanced.get("signal_strength", signal.get("signal", "HOLD"))
        conf = signal.get("confidence", 50)
        score = advanced.get("overall_score", 50)
        
        sig_class = "signal-buy" if "BUY" in sig else "signal-sell" if "SELL" in sig else "signal-hold"
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            emoji = "🟢" if "BUY" in sig else "🔴" if "SELL" in sig else "🟡"
            st.markdown(f"""
            <div class="{sig_class}">
                <div style="font-size: 4rem;">{emoji}</div>
                <div style="font-size: 2rem; font-weight: 800; color: #fff; margin: 15px 0;">{sig}</div>
                <div style="font-size: 1.2rem; color: rgba(255,255,255,0.7);">Confidence: {conf:.0f}%</div>
                <div style="font-size: 1rem; color: rgba(255,255,255,0.5);">Score: {score:.0f}/100</div>
                <div style="margin-top: 20px; display: flex; justify-content: center; gap: 30px;">
                    <div>
                        <div style="color: rgba(255,255,255,0.5); font-size: 0.8rem;">Entry</div>
                        <div style="color: #fff; font-weight: 600;">₹{signal.get("entry_price", 0):,.2f}</div>
                    </div>
                    <div>
                        <div style="color: rgba(255,255,255,0.5); font-size: 0.8rem;">Stop</div>
                        <div style="color: #ff4757; font-weight: 600;">₹{signal.get("stop_loss", 0):,.2f}</div>
                    </div>
                    <div>
                        <div style="color: rgba(255,255,255,0.5); font-size: 0.8rem;">Target</div>
                        <div style="color: #00ff88; font-weight: 600;">₹{signal.get("target_1", 0):,.2f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add to portfolio button
            if st.button("➕ Add to Portfolio", use_container_width=True):
                st.session_state.add_to_portfolio = st.session_state.current_symbol
        
        with col2:
            chart = create_chart(df, st.session_state.current_symbol)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
        
        # Personalized AI analysis
        st.markdown("### 🤖 Personalized AI Analysis")
        
        if st.button("Get AI Recommendation Based on My Portfolio"):
            with st.spinner("AI thinking..."):
                ai = get_ai()
                portfolio = get_smart_portfolio()
                context = portfolio.get_context_for_ai()
                analysis = ai.get_personalized_signal(
                    st.session_state.current_symbol,
                    signal,
                    context
                )
            st.markdown(f'<div class="ai-bubble">{analysis}</div>', unsafe_allow_html=True)
        
        # Tabs for detailed analysis
        tab1, tab2, tab3 = st.tabs(["📊 Technical", "📈 Momentum", "🎯 Scores"])
        
        with tab1:
            tf = advanced.get("timeframes", {})
            if tf:
                col1, col2, col3 = st.columns(3)
                
                for tf_name, col in zip(["daily", "weekly", "monthly"], [col1, col2, col3]):
                    tf_data = tf.get(tf_name, {})
                    trend = tf_data.get("trend", "N/A")
                    t_color = "#00ff88" if "BULLISH" in trend else "#ff4757" if "BEARISH" in trend else "#ffd93d"
                    
                    with col:
                        st.markdown(f"""
                        <div class="glass-card" style="text-align: center;">
                            <div class="metric-label">{tf_name.upper()}</div>
                            <div style="color: {t_color}; font-size: 1.1rem; font-weight: 600;">{trend}</div>
                            <div style="margin-top: 10px;">
                                <div style="color: rgba(255,255,255,0.5);">RSI: {tf_data.get("rsi", "N/A")}</div>
                                <div style="color: rgba(255,255,255,0.5);">MACD: {tf_data.get("macd_signal", "N/A")}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        
        with tab2:
            momentum = advanced.get("momentum", {})
            rs = advanced.get("relative_strength", {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label">Momentum Indicators</div>
                    <div style="margin-top: 15px;">
                        <div style="display: flex; justify-content: space-between; padding: 8px 0;">
                            <span>RSI (14)</span>
                            <span style="font-weight: 600;">{momentum.get("rsi", "N/A")}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 8px 0;">
                            <span>Stochastic %K</span>
                            <span style="font-weight: 600;">{momentum.get("stochastic_k", "N/A"):.0f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 8px 0;">
                            <span>ROC (20)</span>
                            <span style="font-weight: 600;">{momentum.get("roc_20d", "N/A")}%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 8px 0; border-top: 1px solid rgba(255,255,255,0.1); margin-top: 10px;">
                            <span>Condition</span>
                            <span style="font-weight: 600; color: {"#ff4757" if momentum.get("condition") == "OVERBOUGHT" else "#00ff88" if momentum.get("condition") == "OVERSOLD" else "#ffd93d"};">
                                {momentum.get("condition", "N/A")}
                            </span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                rs_score = rs.get("rs_score", 50)
                rs_color = "#00ff88" if rs_score >= 60 else "#ff4757" if rs_score <= 40 else "#ffd93d"
                
                st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label">Relative Strength vs Nifty</div>
                    <div style="text-align: center; margin: 20px 0;">
                        <div style="font-size: 2.5rem; font-weight: 700; color: {rs_color};">{rs_score:.0f}</div>
                        <div style="color: rgba(255,255,255,0.5);">out of 100</div>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {rs_score}%; background: {rs_color};"></div>
                    </div>
                    <div style="text-align: center; color: {rs_color}; font-weight: 600; margin-top: 10px;">
                        {rs.get("vs_market", "N/A")}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            scores = signal.get("scores", {})
            if scores:
                for name, score in scores.items():
                    color = "#00ff88" if score >= 60 else "#ff4757" if score <= 40 else "#ffd93d"
                    st.markdown(f"""
                    <div style="margin: 15px 0;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span>{name.title()}</span>
                            <span style="color: {color}; font-weight: 600;">{score:.0f}/100</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {score}%; background: {color};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


elif page == "📊 Scanner":
    st.markdown('<div class="gradient-text">Opportunity Scanner</div>', unsafe_allow_html=True)
    st.markdown('<span style="color: rgba(255,255,255,0.5);">Find profitable trades that match your style</span>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        universe = st.selectbox("Universe", ["Nifty 50", "Nifty 100", "ETFs", "All Stocks"])
    with col2:
        min_conf = st.slider("Min Confidence", 40, 80, 55)
    with col3:
        signal_type = st.selectbox("Signal Type", ["All", "Strong Buy", "Buy", "Watchlist"])
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        scan = st.button("🚀 Scan Now", use_container_width=True)
    
    if scan:
        with st.spinner("Scanning market for opportunities..."):
            agent = get_agent()
            
            if universe == "ETFs":
                from config.settings import ETF_LIST
                result = agent.scan_opportunities(n=20, universe=ETF_LIST)
            elif universe == "All Stocks":
                from config.settings import FULL_UNIVERSE
                result = agent.scan_opportunities(n=20, universe=FULL_UNIVERSE[:60])
            else:
                from config.settings import NIFTY_50
                univ = NIFTY_50 if universe == "Nifty 50" else NIFTY_50 + NIFTY_50[:50]
                result = agent.scan_opportunities(n=20, universe=univ)
        
        opps = result.get("top_opportunities", [])
        
        # Filter
        if signal_type == "Strong Buy":
            opps = [o for o in opps if o.get("signal") == "STRONG_BUY"]
        elif signal_type == "Buy":
            opps = [o for o in opps if o.get("signal") in ["BUY", "STRONG_BUY"]]
        
        opps = [o for o in opps if o.get("confidence", 0) >= min_conf]
        
        if opps:
            st.success(f"Found {len(opps)} opportunities!")
            
            for opp in opps:
                sig = opp.get("signal", "HOLD")
                conf = opp.get("confidence", 50)
                sig_color = "#00ff88" if "BUY" in sig else "#ffd93d"
                
                rr = opp.get("risk_reward_ratio", 0)
                
                st.markdown(f"""
                <div class="glass-card" style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 1.3rem; font-weight: 700; color: #fff;">{opp.get("symbol", "")}</div>
                        <div style="color: {sig_color}; font-weight: 600;">{sig}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: rgba(255,255,255,0.5);">Confidence</div>
                        <div style="font-size: 1.5rem; font-weight: 700;">{conf:.0f}%</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: rgba(255,255,255,0.5);">R:R</div>
                        <div style="font-size: 1.2rem; font-weight: 600; color: #00ff88;">1:{rr:.1f}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: #fff;">Entry: ₹{opp.get("entry_price", 0):,.2f}</div>
                        <div style="color: #ff4757;">Stop: ₹{opp.get("stop_loss", 0):,.2f}</div>
                        <div style="color: #00ff88;">Target: ₹{opp.get("target_1", 0):,.2f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No opportunities match your criteria. Try adjusting filters.")


elif page == "🤖 AI Advisor":
    st.markdown('<div class="gradient-text">AI Trading Advisor</div>', unsafe_allow_html=True)
    st.markdown('<span style="color: rgba(255,255,255,0.5);">Your personal AI that knows your portfolio</span>', unsafe_allow_html=True)
    
    # Chat history
    for msg in st.session_state.messages:
        bubble_class = "user-bubble" if msg["role"] == "user" else "ai-bubble"
        prefix = "You: " if msg["role"] == "user" else "🤖 "
        st.markdown(f'<div class="{bubble_class}">{prefix}{msg["content"]}</div>', unsafe_allow_html=True)
    
    # Input
    user_input = st.chat_input("Ask about stocks, your portfolio, or trading strategies...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("AI thinking..."):
            ai = get_ai()
            portfolio = get_smart_portfolio()
            context = portfolio.get_context_for_ai()
            
            # Include portfolio context in every message
            enhanced_query = f"""
User's Portfolio Context:
{json.dumps(context, indent=2, default=str)[:1500]}

User's Question: {user_input}
"""
            response = ai.chat(enhanced_query, context)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    # Quick actions
    st.markdown("---")
    st.markdown("### Quick Questions")
    
    quick_qs = [
        "Analyze my portfolio and suggest changes",
        "What should I buy today?",
        "Which stocks should I exit?",
        "Am I too concentrated in any sector?",
        "Best swing trade setup right now?",
        "What's wrong with my trading?",
    ]
    
    cols = st.columns(3)
    for i, q in enumerate(quick_qs):
        with cols[i % 3]:
            if st.button(q, key=f"q_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                
                with st.spinner("AI analyzing..."):
                    ai = get_ai()
                    portfolio = get_smart_portfolio()
                    context = portfolio.get_context_for_ai()
                    response = ai.chat(q, context)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()


elif page == "⚙️ Settings":
    st.markdown('<div class="gradient-text">Settings</div>', unsafe_allow_html=True)
    
    st.markdown("### 🔑 API Configuration")
    
    import os
    groq_key = os.getenv("GROQ_API_KEY", "")
    
    if groq_key:
        st.success("✅ Groq AI is configured and active")
    else:
        st.warning("⚠️ Set GROQ_API_KEY for AI features")
        st.code("export GROQ_API_KEY='your-key-here'")
        st.markdown("[Get FREE API Key →](https://console.groq.com)")
    
    st.markdown("### 💰 Portfolio Settings")
    
    portfolio = get_smart_portfolio()
    
    new_capital = st.number_input("Total Capital", value=portfolio.portfolio.get("total_capital", 100000), step=10000)
    new_cash = st.number_input("Available Cash", value=portfolio.portfolio.get("cash", 100000), step=1000)
    
    if st.button("Update"):
        portfolio.portfolio["total_capital"] = new_capital
        portfolio.portfolio["cash"] = new_cash
        portfolio._save_portfolio()
        st.success("Updated!")
    
    st.markdown("### 🗑️ Reset")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat cleared!")
    
    if st.button("Reset Portfolio (DANGER)", type="secondary"):
        portfolio.portfolio = {
            "holdings": [],
            "cash": new_capital,
            "total_capital": new_capital,
            "last_updated": datetime.now().isoformat(),
        }
        portfolio._save_portfolio()
        portfolio.history = []
        portfolio._save_history()
        st.success("Portfolio reset!")
        st.rerun()


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.3); padding: 20px;">
    📈 Trading Agent v2.0 • AI-Powered • Portfolio-Aware • Built for Profit
</div>
""", unsafe_allow_html=True)
