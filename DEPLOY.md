# 🚀 Free Deployment Guide

Your private, personalized trading dashboard with persistent data.

---

## Option 1: Run Locally (Simplest - Recommended)

**Best for:** Data always persists, 100% private, no internet needed

### Quick Start
1. Double-click `start_trading_bot.command` on your Desktop
2. Open http://localhost:8501
3. Login with password: `trader123`

### Auto-Start on Login (Mac)
1. Open **System Preferences → Users & Groups → Login Items**
2. Click **+** and add `start_trading_bot.command`
3. Now it starts automatically when you log in!

### Change Password
```bash
export TRADING_BOT_PASSWORD="your-secret-password"
```

---

## Option 2: Railway.app (Free Cloud - Always On)

**Best for:** Access from anywhere, always online, data persists

### Steps:
1. Go to [railway.app](https://railway.app) and sign up with GitHub
2. Create new project → Deploy from GitHub repo
3. Add your project folder to GitHub first:

```bash
cd /Users/devps/Desktop/TradingBot/indian-market-agent
git init
git add .
git commit -m "Initial commit"
# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/trading-bot.git
git push -u origin main
```

4. In Railway, set environment variables:
   - `GROQ_API_KEY` = your-groq-key
   - `TRADING_BOT_PASSWORD` = your-secret-password

5. Railway will give you a URL like `https://trading-bot.up.railway.app`

**Free Tier:** 500 hours/month (enough for personal use)

---

## Option 3: Oracle Cloud (Free Forever VPS)

**Best for:** Always on, unlimited hours, full control

### Steps:
1. Sign up at [cloud.oracle.com](https://cloud.oracle.com) (free tier)
2. Create a VM instance (Always Free: 1GB RAM, 1 OCPU)
3. SSH into your instance and run:

```bash
# Install Python
sudo apt update && sudo apt install python3.11 python3.11-venv -y

# Clone your code
git clone https://github.com/YOUR_USERNAME/trading-bot.git
cd trading-bot

# Setup
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set environment variables
export GROQ_API_KEY="your-key"
export TRADING_BOT_PASSWORD="your-password"

# Run with nohup (keeps running after disconnect)
nohup streamlit run app.py --server.port 8501 --server.headless true &
```

4. Open port 8501 in Oracle Cloud security rules
5. Access via `http://YOUR_VM_IP:8501`

---

## Option 4: Render.com (Simple Cloud)

**Best for:** Easy setup, free tier

### Steps:
1. Push code to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port $PORT --server.headless true`
5. Add environment variables (GROQ_API_KEY, TRADING_BOT_PASSWORD)

**Note:** Free tier sleeps after 15 min inactivity. Data may not persist.

---

## 🔐 Security Tips

1. **Always use a strong password**
   ```bash
   export TRADING_BOT_PASSWORD="MyStr0ng!Pass#2024"
   ```

2. **Never share your GROQ_API_KEY**

3. **For cloud deployments**, use HTTPS (Railway/Render provide this)

4. **Backup your data** periodically:
   ```bash
   cp -r data/ ~/trading_data_backup/
   ```

---

## 📊 Data Storage

Your data is stored in:
- `data/portfolio.json` - Your holdings
- `data/trade_history.json` - All trades
- `data/daily_snapshots.json` - Portfolio history

**To backup:**
```bash
zip -r trading_backup_$(date +%Y%m%d).zip data/
```

**To restore:**
```bash
unzip trading_backup_YYYYMMDD.zip
```

---

## 🎯 Recommended Setup

For a personal trading tool, I recommend:

1. **Daily Use:** Run locally (fastest, most private)
2. **Access from Phone:** Deploy to Railway (free, always on)
3. **Both:** Run locally + backup data to cloud

Enjoy your private trading dashboard! 📈💰
