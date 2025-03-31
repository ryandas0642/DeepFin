import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.scaler = StandardScaler()
        
    def create_features(self, market_data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Create features from market data including VIX and options data."""
        try:
            # Get OHLCV and VIX data
            ohlcv = market_data['ohlcv']
            vix = market_data['vix']
            
            # Create price-based features
            price_features = self._create_price_features(ohlcv)
            
            # Create VIX-based features
            vix_features = self._create_vix_features(vix)
            
            # Create options-based features
            options_features = self._create_options_features(market_data['options'])
            
            # Combine all features
            features = np.concatenate([
                price_features,
                vix_features,
                options_features
            ], axis=1)
            
            # Create labels (next day's return)
            labels = ohlcv['Close'].pct_change().shift(-1).values[self.window_size:]
            
            return features, labels
            
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            raise
            
    def _create_price_features(self, ohlcv: pd.DataFrame) -> np.ndarray:
        """Create price-based technical indicators."""
        try:
            # Calculate returns
            returns = ohlcv['Close'].pct_change()
            
            # Calculate moving averages
            ma5 = ohlcv['Close'].rolling(window=5).mean()
            ma20 = ohlcv['Close'].rolling(window=20).mean()
            ma50 = ohlcv['Close'].rolling(window=50).mean()
            
            # Calculate RSI
            delta = ohlcv['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = ohlcv['Close'].ewm(span=12, adjust=False).mean()
            exp2 = ohlcv['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            # Calculate Bollinger Bands
            bb_middle = ohlcv['Close'].rolling(window=20).mean()
            bb_std = ohlcv['Close'].rolling(window=20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            # Combine features
            price_features = np.column_stack([
                returns.values,
                ma5.values,
                ma20.values,
                ma50.values,
                rsi.values,
                macd.values,
                signal.values,
                bb_upper.values,
                bb_middle.values,
                bb_lower.values
            ])
            
            # Remove NaN values
            price_features = price_features[self.window_size:]
            
            return price_features
            
        except Exception as e:
            logger.error(f"Error creating price features: {str(e)}")
            raise
            
    def _create_vix_features(self, vix: pd.DataFrame) -> np.ndarray:
        """Create VIX-based features."""
        try:
            # Calculate VIX returns
            vix_returns = vix['Close'].pct_change()
            
            # Calculate VIX moving averages
            vix_ma5 = vix['Close'].rolling(window=5).mean()
            vix_ma20 = vix['Close'].rolling(window=20).mean()
            
            # Calculate VIX percentile
            vix_percentile = vix['Close'].rolling(window=252).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )
            
            # Calculate VIX term structure (if available)
            vix_term = vix['Close'].rolling(window=5).apply(
                lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0]
            )
            
            # Combine VIX features
            vix_features = np.column_stack([
                vix_returns.values,
                vix_ma5.values,
                vix_ma20.values,
                vix_percentile.values,
                vix_term.values
            ])
            
            # Remove NaN values
            vix_features = vix_features[self.window_size:]
            
            return vix_features
            
        except Exception as e:
            logger.error(f"Error creating VIX features: {str(e)}")
            raise
            
    def _create_options_features(self, options_data: Dict) -> np.ndarray:
        """Create options-based features."""
        try:
            # Initialize features array
            num_dates = len(options_data)
            options_features = np.zeros((num_dates, 4))  # 4 features per date
            
            # Extract features for each date
            for i, (expiry, data) in enumerate(options_data.items()):
                # Calculate put-call ratio
                calls_volume = data['calls']['volume'].sum()
                puts_volume = data['puts']['volume'].sum()
                put_call_ratio = puts_volume / calls_volume if calls_volume > 0 else 0
                
                # Calculate average implied volatility
                calls_iv = data['calls']['impliedVolatility'].mean()
                puts_iv = data['puts']['impliedVolatility'].mean()
                avg_iv = (calls_iv + puts_iv) / 2
                
                # Calculate volume-weighted average strike
                calls_weighted = (data['calls']['volume'] * data['calls']['strike']).sum()
                puts_weighted = (data['puts']['volume'] * data['puts']['strike']).sum()
                total_volume = calls_volume + puts_volume
                weighted_strike = (calls_weighted + puts_weighted) / total_volume if total_volume > 0 else 0
                
                # Calculate open interest ratio
                calls_oi = data['calls']['openInterest'].sum()
                puts_oi = data['puts']['openInterest'].sum()
                oi_ratio = puts_oi / calls_oi if calls_oi > 0 else 0
                
                options_features[i] = [put_call_ratio, avg_iv, weighted_strike, oi_ratio]
            
            # Remove NaN values
            options_features = options_features[self.window_size:]
            
            return options_features
            
        except Exception as e:
            logger.error(f"Error creating options features: {str(e)}")
            raise 