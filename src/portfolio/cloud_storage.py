"""
☁️ Cloud Storage for Portfolio Data

Keeps your data synced across all devices (Mac, Phone, anywhere!)
Uses Supabase (free) or falls back to local JSON files.

Setup Supabase (2 minutes):
1. Go to supabase.com and create free account
2. Create new project
3. Go to SQL Editor, run the SQL below
4. Get your URL and KEY from Settings > API
5. Set environment variables:
   export SUPABASE_URL="https://xxx.supabase.co"
   export SUPABASE_KEY="your-anon-key"
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# Supabase SQL to create tables (run this once in Supabase SQL Editor):
SETUP_SQL = """
-- Portfolio holdings
CREATE TABLE IF NOT EXISTS portfolio (
    id SERIAL PRIMARY KEY,
    user_id TEXT DEFAULT 'default',
    data JSONB NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Trade history
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    user_id TEXT DEFAULT 'default',
    trade_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Daily snapshots
CREATE TABLE IF NOT EXISTS snapshots (
    id SERIAL PRIMARY KEY,
    user_id TEXT DEFAULT 'default',
    date TEXT NOT NULL,
    snapshot_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_portfolio_user ON portfolio(user_id);
CREATE INDEX IF NOT EXISTS idx_trades_user ON trades(user_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_date ON snapshots(date);
"""


class CloudStorage:
    """
    Hybrid storage: Cloud (Supabase) + Local fallback.
    Your data is always saved locally AND synced to cloud.
    """
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.supabase = None
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        
        # Try to connect to Supabase
        self._init_supabase()
    
    def _init_supabase(self):
        """Initialize Supabase connection if credentials exist."""
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if url and key:
            try:
                from supabase import create_client
                self.supabase = create_client(url, key)
                print("☁️ Connected to Supabase cloud storage")
            except ImportError:
                print("📦 Install supabase: pip install supabase")
            except Exception as e:
                print(f"⚠️ Supabase connection failed: {e}")
        else:
            print("💾 Using local storage (set SUPABASE_URL & SUPABASE_KEY for cloud sync)")
    
    def _local_path(self, name: str) -> Path:
        """Get path to local JSON file."""
        return self.data_dir / f"{name}.json"
    
    def _load_local(self, name: str) -> Any:
        """Load from local JSON file."""
        path = self._local_path(name)
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None
    
    def _save_local(self, name: str, data: Any):
        """Save to local JSON file."""
        path = self._local_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    # =========================================================================
    # PORTFOLIO
    # =========================================================================
    
    def save_portfolio(self, portfolio_data: Dict) -> bool:
        """Save portfolio to cloud + local."""
        # Always save locally
        self._save_local("portfolio", portfolio_data)
        
        # Sync to cloud if available
        if self.supabase:
            try:
                # Upsert (update or insert)
                self.supabase.table("portfolio").upsert({
                    "user_id": self.user_id,
                    "data": portfolio_data,
                    "updated_at": datetime.now().isoformat()
                }, on_conflict="user_id").execute()
                return True
            except Exception as e:
                print(f"⚠️ Cloud sync failed: {e}")
        
        return True
    
    def load_portfolio(self) -> Dict:
        """Load portfolio from cloud (or local fallback)."""
        # Try cloud first
        if self.supabase:
            try:
                result = self.supabase.table("portfolio") \
                    .select("data") \
                    .eq("user_id", self.user_id) \
                    .single() \
                    .execute()
                
                if result.data:
                    # Also update local cache
                    self._save_local("portfolio", result.data["data"])
                    return result.data["data"]
            except Exception as e:
                print(f"⚠️ Cloud load failed, using local: {e}")
        
        # Fallback to local
        local = self._load_local("portfolio")
        return local or {
            "holdings": [],
            "cash": 100000,
            "total_capital": 100000,
            "last_updated": datetime.now().isoformat()
        }
    
    # =========================================================================
    # TRADES
    # =========================================================================
    
    def save_trade(self, trade: Dict) -> bool:
        """Save a new trade to cloud + local."""
        # Load existing trades
        trades = self.load_trades()
        trades.append(trade)
        
        # Save locally
        self._save_local("trade_history", {"trades": trades})
        
        # Sync to cloud
        if self.supabase:
            try:
                self.supabase.table("trades").insert({
                    "user_id": self.user_id,
                    "trade_data": trade,
                    "created_at": datetime.now().isoformat()
                }).execute()
                return True
            except Exception as e:
                print(f"⚠️ Cloud sync failed: {e}")
        
        return True
    
    def load_trades(self) -> List[Dict]:
        """Load all trades from cloud (or local fallback)."""
        if self.supabase:
            try:
                result = self.supabase.table("trades") \
                    .select("trade_data") \
                    .eq("user_id", self.user_id) \
                    .order("created_at") \
                    .execute()
                
                if result.data:
                    trades = [r["trade_data"] for r in result.data]
                    self._save_local("trade_history", {"trades": trades})
                    return trades
            except Exception as e:
                print(f"⚠️ Cloud load failed: {e}")
        
        # Fallback to local
        local = self._load_local("trade_history")
        if isinstance(local, dict):
            return local.get("trades", [])
        elif isinstance(local, list):
            return local
        return []
    
    # =========================================================================
    # SNAPSHOTS
    # =========================================================================
    
    def save_snapshot(self, date: str, snapshot: Dict) -> bool:
        """Save daily snapshot."""
        # Load existing
        snapshots = self.load_snapshots()
        
        # Update or append
        existing = False
        for i, s in enumerate(snapshots):
            if s.get("date") == date:
                snapshots[i] = snapshot
                existing = True
                break
        
        if not existing:
            snapshots.append(snapshot)
        
        # Save locally
        self._save_local("daily_snapshots", {"snapshots": snapshots})
        
        # Sync to cloud
        if self.supabase:
            try:
                self.supabase.table("snapshots").upsert({
                    "user_id": self.user_id,
                    "date": date,
                    "snapshot_data": snapshot
                }, on_conflict="user_id,date").execute()
            except Exception as e:
                print(f"⚠️ Cloud sync failed: {e}")
        
        return True
    
    def load_snapshots(self) -> List[Dict]:
        """Load all snapshots."""
        if self.supabase:
            try:
                result = self.supabase.table("snapshots") \
                    .select("snapshot_data") \
                    .eq("user_id", self.user_id) \
                    .order("date") \
                    .execute()
                
                if result.data:
                    return [r["snapshot_data"] for r in result.data]
            except:
                pass
        
        # Fallback
        local = self._load_local("daily_snapshots")
        if isinstance(local, dict):
            return local.get("snapshots", [])
        elif isinstance(local, list):
            return local
        return []


# Singleton instance
_storage = None

def get_cloud_storage(user_id: str = "default") -> CloudStorage:
    """Get cloud storage instance."""
    global _storage
    if _storage is None:
        _storage = CloudStorage(user_id)
    return _storage
