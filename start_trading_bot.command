#!/bin/bash
# Double-click this file to start the Trading Bot

cd /Users/devps/Desktop/TradingBot/indian-market-agent
source venv/bin/activate

# Set your API key (edit this line)
export GROQ_API_KEY="your-groq-api-key-here"

echo "🚀 Starting Trading Bot..."
echo "📊 Open http://localhost:8501 in your browser"
echo ""

streamlit run app.py --server.headless true
