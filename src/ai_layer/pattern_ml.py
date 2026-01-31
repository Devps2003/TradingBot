"""
Machine Learning Pattern Recognition Module.

Uses ML models for:
- Pattern classification
- Price direction prediction
- Feature importance analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pickle

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR
from src.utils.logger import LoggerMixin


class PatternML(LoggerMixin):
    """
    Machine learning based pattern recognition.
    """
    
    def __init__(self):
        """Initialize the pattern ML module."""
        self.model = None
        self.scaler = None
        self.model_path = DATA_DIR / "models" / "pattern_model.pkl"
        self.scaler_path = DATA_DIR / "models" / "pattern_scaler.pkl"
        
        # Create models directory
        (DATA_DIR / "models").mkdir(exist_ok=True)
        
        # Try to load existing model
        self._load_model()
    
    def _load_model(self) -> bool:
        """
        Load trained model from disk.
        
        Returns:
            True if model loaded successfully
        """
        try:
            if self.model_path.exists() and self.scaler_path.exists():
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                self.logger.info("Loaded trained pattern model")
                return True
        except Exception as e:
            self.logger.warning(f"Could not load model: {e}")
        
        return False
    
    def _save_model(self) -> bool:
        """
        Save trained model to disk.
        
        Returns:
            True if saved successfully
        """
        try:
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)
            with open(self.scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            self.logger.info("Saved pattern model")
            return True
        except Exception as e:
            self.logger.error(f"Could not save model: {e}")
            return False
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from OHLCV data.
        
        Args:
            df: OHLCV dataframe
        
        Returns:
            Feature dataframe
        """
        features = pd.DataFrame()
        
        # Price-based features
        features["returns_1d"] = df["close"].pct_change(1)
        features["returns_5d"] = df["close"].pct_change(5)
        features["returns_10d"] = df["close"].pct_change(10)
        features["returns_20d"] = df["close"].pct_change(20)
        
        # Range features
        features["daily_range"] = (df["high"] - df["low"]) / df["close"]
        features["body_range"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"])
        
        # Moving average features
        features["sma_20_ratio"] = df["close"] / df["close"].rolling(20).mean()
        features["sma_50_ratio"] = df["close"] / df["close"].rolling(50).mean()
        features["ema_9_ratio"] = df["close"] / df["close"].ewm(span=9).mean()
        
        # Momentum features
        features["rsi_14"] = self._calculate_rsi(df["close"], 14)
        
        # Volume features
        features["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        features["volume_change"] = df["volume"].pct_change()
        
        # Volatility features
        features["volatility_20"] = df["close"].pct_change().rolling(20).std()
        
        # Higher high / lower low features
        features["higher_high"] = (df["high"] > df["high"].shift(1)).astype(int)
        features["lower_low"] = (df["low"] < df["low"].shift(1)).astype(int)
        
        return features.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        forward_days: int = 5,
        threshold: float = 0.02,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data with labels.
        
        Args:
            df: OHLCV dataframe
            forward_days: Days to look ahead for label
            threshold: Minimum move for bullish/bearish label
        
        Returns:
            Features and labels
        """
        features = self.extract_features(df)
        
        # Create labels based on forward returns
        forward_returns = df["close"].pct_change(forward_days).shift(-forward_days)
        
        labels = pd.Series(index=forward_returns.index, data=0)  # 0 = neutral
        labels[forward_returns > threshold] = 1  # 1 = bullish
        labels[forward_returns < -threshold] = -1  # -1 = bearish
        
        # Align features and labels
        common_idx = features.index.intersection(labels.dropna().index)
        features = features.loc[common_idx]
        labels = labels.loc[common_idx]
        
        return features, labels
    
    def train_model(
        self,
        df: pd.DataFrame,
        forward_days: int = 5,
    ) -> Dict[str, Any]:
        """
        Train the pattern recognition model.
        
        Args:
            df: OHLCV dataframe with historical data
            forward_days: Days to look ahead for prediction
        
        Returns:
            Training results
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "sklearn not installed"}
        
        features, labels = self.prepare_training_data(df, forward_days)
        
        if len(features) < 100:
            return {"error": "Insufficient data for training"}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, shuffle=False
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_accuracy = self.model.score(X_train_scaled, y_train)
        test_accuracy = self.model.score(X_test_scaled, y_test)
        
        # Save model
        self._save_model()
        
        return {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_accuracy": round(train_accuracy, 4),
            "test_accuracy": round(test_accuracy, 4),
            "features": list(features.columns),
        }
    
    def predict_pattern(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict pattern/direction from current data.
        
        Args:
            df: Recent OHLCV data
        
        Returns:
            Prediction results
        """
        if self.model is None or self.scaler is None:
            return {
                "prediction": "NEUTRAL",
                "confidence": 50,
                "note": "Model not trained",
            }
        
        if not SKLEARN_AVAILABLE:
            return {"error": "sklearn not installed"}
        
        # Extract features
        features = self.extract_features(df)
        
        if features.empty:
            return {"prediction": "NEUTRAL", "confidence": 50}
        
        # Use last row (current state)
        latest = features.iloc[[-1]]
        
        # Scale
        latest_scaled = self.scaler.transform(latest)
        
        # Predict
        prediction = self.model.predict(latest_scaled)[0]
        probabilities = self.model.predict_proba(latest_scaled)[0]
        
        # Get confidence
        confidence = max(probabilities) * 100
        
        # Map prediction
        if prediction == 1:
            pred_label = "BULLISH"
        elif prediction == -1:
            pred_label = "BEARISH"
        else:
            pred_label = "NEUTRAL"
        
        return {
            "prediction": pred_label,
            "confidence": round(confidence, 1),
            "probabilities": {
                "bearish": round(probabilities[0] * 100, 1) if len(probabilities) > 0 else 0,
                "neutral": round(probabilities[1] * 100, 1) if len(probabilities) > 1 else 0,
                "bullish": round(probabilities[2] * 100, 1) if len(probabilities) > 2 else 0,
            },
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from trained model.
        
        Returns:
            Dictionary of feature importances
        """
        if self.model is None:
            return {}
        
        if not hasattr(self.model, "feature_importances_"):
            return {}
        
        # Get feature names from training
        # This assumes we stored them during training
        feature_names = [
            "returns_1d", "returns_5d", "returns_10d", "returns_20d",
            "daily_range", "body_range", "sma_20_ratio", "sma_50_ratio",
            "ema_9_ratio", "rsi_14", "volume_ratio", "volume_change",
            "volatility_20", "higher_high", "lower_low"
        ]
        
        importances = self.model.feature_importances_
        
        return {
            name: round(imp, 4)
            for name, imp in sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )
        }
    
    def backtest_model(
        self,
        df: pd.DataFrame,
        forward_days: int = 5,
    ) -> Dict[str, Any]:
        """
        Backtest the model on historical data.
        
        Args:
            df: OHLCV dataframe
            forward_days: Days to look ahead
        
        Returns:
            Backtest results
        """
        if self.model is None or self.scaler is None:
            return {"error": "Model not trained"}
        
        features, labels = self.prepare_training_data(df, forward_days)
        
        if features.empty:
            return {"error": "Could not prepare test data"}
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        predictions = self.model.predict(features_scaled)
        
        # Calculate metrics
        correct = sum(predictions == labels)
        total = len(labels)
        accuracy = correct / total if total > 0 else 0
        
        # Direction accuracy (ignoring neutral)
        directional_mask = (labels != 0) & (predictions != 0)
        directional_correct = sum((predictions == labels)[directional_mask])
        directional_total = sum(directional_mask)
        directional_accuracy = directional_correct / directional_total if directional_total > 0 else 0
        
        return {
            "total_predictions": total,
            "accuracy": round(accuracy, 4),
            "directional_accuracy": round(directional_accuracy, 4),
            "bullish_predictions": sum(predictions == 1),
            "bearish_predictions": sum(predictions == -1),
            "neutral_predictions": sum(predictions == 0),
        }


# Convenience function
def predict_direction(df: pd.DataFrame) -> Dict:
    """Predict price direction using ML."""
    ml = PatternML()
    return ml.predict_pattern(df)
