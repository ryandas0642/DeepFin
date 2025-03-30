import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import os
import warnings

from src.models.arima_garch import ARIMAGARCHModel
from src.models.deep_learning import DeepLearningModel, register_torch_classes

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Register models at import
register_torch_classes()

class EnsembleModel:
    def __init__(self, window_size=60, deep_model_type='lstm', arima_weight=0.5, dl_weight=0.5):
        """
        Ensemble model combining ARIMA+GARCH and deep learning.
        
        Args:
            window_size (int): Size of the rolling window
            deep_model_type (str): Type of deep learning model ('lstm' or 'gru')
            arima_weight (float): Weight for ARIMA+GARCH predictions in ensemble
            dl_weight (float): Weight for deep learning predictions in ensemble
        """
        self.window_size = window_size
        self.deep_model_type = deep_model_type
        self.arima_weight = arima_weight
        self.dl_weight = dl_weight
        
        # Initialize models
        try:
            logger.info("Initializing ARIMA+GARCH model...")
            self.arima_garch = ARIMAGARCHModel()
            
            logger.info(f"Initializing {deep_model_type.upper()} model...")
            # Make sure PyTorch classes are registered
            register_torch_classes()
            self.dl_model = DeepLearningModel(model_type=deep_model_type)
            
            # Initialize scalers
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            
            logger.info("Ensemble model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            logger.info("Using fallback initialization...")
            # Try minimal initialization
            self.arima_garch = ARIMAGARCHModel()
            self.dl_model = DeepLearningModel(model_type='lstm')  # Default to LSTM
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
        
    def fit(self, data, target_col='log_return'):
        """
        Fit the ensemble model to data.
        
        Args:
            data (pd.DataFrame): Data containing target and features
            target_col (str): Name of the target column
            
        Returns:
            self
        """
        try:
            logger.info("Fitting ensemble model...")
            
            # Extract target variable
            y = data[target_col].values
            
            # Get feature columns for deep learning
            feature_cols = [
                'log_return', 'macd', 'macd_signal', 'macd_diff', 
                'rsi', 'bollinger_width', 'volatility',
                'momentum_5', 'momentum_10', 'momentum_20'
            ]
            
            # Check if all features exist
            missing_cols = [col for col in feature_cols if col not in data.columns]
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
                feature_cols = [col for col in feature_cols if col in data.columns]
                
            # Make sure we have at least some features
            if not feature_cols:
                logger.warning("No feature columns found. Using only 'log_return' for deep learning.")
                feature_cols = ['log_return']
                
                # Make sure log_return exists
                if 'log_return' not in data.columns:
                    logger.warning("'log_return' not found. Creating it.")
                    if 'Close' in data.columns:
                        data['log_return'] = np.log(data['Close'] / data['Close'].shift(1)).fillna(0)
                    elif 'Adj Close' in data.columns:
                        data['log_return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1)).fillna(0)
                    else:
                        logger.error("No price column found for creating log_return")
                        raise ValueError("No suitable columns for feature creation")
                
            # Extract features
            X = data[feature_cols].values
            
            # Ensure enough data for window
            if len(data) <= self.window_size:
                logger.warning(f"Not enough data for window size {self.window_size}. Reducing window size.")
                self.window_size = max(5, len(data) // 2)
                
            # Create sequences for deep learning
            X_seq = []
            y_seq = []
            
            for i in range(len(data) - self.window_size):
                X_seq.append(X[i:i+self.window_size])
                y_seq.append(y[i+self.window_size])
                
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            
            # Check if we have sequences to train on
            if len(X_seq) == 0 or len(y_seq) == 0:
                logger.error("No training sequences could be created")
                return self
                
            # Fit deep learning model
            try:
                logger.info(f"Fitting {self.deep_model_type.upper()} model...")
                # Register model classes before training
                register_torch_classes()
                self.dl_model.fit(X_seq, y_seq)
            except Exception as e:
                logger.error(f"Error fitting deep learning model: {str(e)}")
                logger.info("Continuing with only ARIMA+GARCH model")
                # Adjust weights if DL model fails
                self.arima_weight = 1.0
                self.dl_weight = 0.0
            
            # Fit ARIMA+GARCH model on the target variable
            try:
                logger.info("Fitting ARIMA+GARCH model...")
                self.arima_garch.fit(data[target_col])
            except Exception as e:
                logger.error(f"Error fitting ARIMA+GARCH model: {str(e)}")
                logger.info("Continuing with only deep learning model")
                # Adjust weights if ARIMA model fails
                self.arima_weight = 0.0
                self.dl_weight = 1.0
            
            logger.info("Ensemble model fitting completed")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting ensemble model: {str(e)}")
            # Return self to prevent cascade failures
            return self
            
    def predict_next(self, data, target_col='log_return'):
        """
        Predict the next value in the time series.
        
        Args:
            data (pd.DataFrame): Recent data
            target_col (str): Name of the target column
            
        Returns:
            float: Predicted next value
        """
        try:
            # Ensure we have enough data
            if len(data) < self.window_size:
                logger.warning(f"Not enough data. Need at least {self.window_size} points, got {len(data)}")
                # Reduce window size if needed
                self.window_size = max(5, len(data) - 1)
                logger.info(f"Adjusted window size to {self.window_size}")
                
            # Get feature columns for deep learning
            feature_cols = [
                'log_return', 'macd', 'macd_signal', 'macd_diff', 
                'rsi', 'bollinger_width', 'volatility',
                'momentum_5', 'momentum_10', 'momentum_20'
            ]
            
            # Check if all features exist
            missing_cols = [col for col in feature_cols if col not in data.columns]
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
                feature_cols = [col for col in feature_cols if col in data.columns]
                
            # Make sure we have at least some features
            if not feature_cols:
                logger.warning("No feature columns found. Using only 'log_return'.")
                if 'log_return' in data.columns:
                    feature_cols = ['log_return']
                else:
                    logger.error("No features available for prediction")
                    return 0.0
                
            # Get the last window_size points for deep learning
            X = data[feature_cols].values[-self.window_size:].reshape(1, self.window_size, len(feature_cols))
            
            # Make deep learning prediction
            dl_prediction = 0.0
            if self.dl_weight > 0:
                try:
                    register_torch_classes()
                    dl_pred = self.dl_model.predict(X)
                    
                    # Ensure dl_prediction is a scalar
                    if isinstance(dl_pred, np.ndarray):
                        if dl_pred.size > 0:
                            dl_prediction = float(dl_pred.flatten()[0])
                        else:
                            dl_prediction = 0.0
                    else:
                        dl_prediction = float(dl_pred)
                    
                    logger.info(f"DL prediction: {dl_prediction}, type: {type(dl_prediction)}")
                except Exception as e:
                    logger.warning(f"Error making DL prediction: {str(e)}. Using zero.")
                    dl_prediction = 0.0
                    self.dl_weight = 0.0
                    self.arima_weight = 1.0
            
            # Make ARIMA+GARCH prediction
            arima_prediction = 0.0
            if self.arima_weight > 0:
                try:
                    arima_result, _ = self.arima_garch.predict(steps=1)
                    
                    # Handle different return types from ARIMA
                    if isinstance(arima_result, pd.Series):
                        arima_prediction = float(arima_result.iloc[0])
                    elif isinstance(arima_result, np.ndarray):
                        if arima_result.size > 0:
                            arima_prediction = float(arima_result.flatten()[0])
                        else:
                            arima_prediction = 0.0
                    else:
                        arima_prediction = float(arima_result)
                        
                    logger.info(f"ARIMA prediction: {arima_prediction}, type: {type(arima_prediction)}")
                except Exception as e:
                    logger.warning(f"Error making ARIMA prediction: {str(e)}. Using zero.")
                    arima_prediction = 0.0
                    self.arima_weight = 0.0
                    self.dl_weight = 1.0
            
            # Validate weights sum to 1.0
            total_weight = self.arima_weight + self.dl_weight
            if abs(total_weight) < 1e-6:
                logger.warning("Both model weights are zero. Using equal weights.")
                self.arima_weight = self.dl_weight = 0.5
            elif total_weight != 1.0:
                logger.warning(f"Weights don't sum to 1.0: {total_weight}. Normalizing.")
                self.arima_weight /= total_weight
                self.dl_weight /= total_weight
            
            # Combine predictions
            try:
                ensemble_prediction = (self.arima_weight * arima_prediction) + (self.dl_weight * dl_prediction)
                # Ensure the final result is a scalar float
                ensemble_prediction = float(ensemble_prediction)
                logger.info(f"Ensemble prediction: {ensemble_prediction}, type: {type(ensemble_prediction)}")
            except Exception as e:
                logger.warning(f"Error combining predictions: {str(e)}. Using DL prediction only.")
                # If combining fails, use whichever prediction is available
                if self.dl_weight > 0:
                    ensemble_prediction = float(dl_prediction)
                else:
                    ensemble_prediction = float(arima_prediction)
            
            return ensemble_prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            # Return a default value in case of error to prevent app crash
            return 0.0
            
    def generate_signals(self, data, target_col='log_return'):
        """
        Generate trading signals based on rolling predictions.
        
        Args:
            data (pd.DataFrame): Historical data
            target_col (str): Name of the target column
            
        Returns:
            pd.DataFrame: Data with added signals
        """
        try:
            logger.info("Generating trading signals...")
            
            # Make a copy of the data
            result = data.copy()
            
            # Add a column for signals
            result['signal'] = 0
            
            # Add columns for predictions
            result['prediction'] = np.nan
            
            # Need at least window_size data points to start
            if len(result) <= self.window_size:
                logger.warning(f"Not enough data points. Need more than {self.window_size}, got {len(result)}.")
                return result
                
            start_idx = self.window_size
            
            # Generate signals using rolling window
            for i in range(start_idx, len(result)):
                try:
                    # Get the training data up to this point
                    train_data = result.iloc[:i]
                    
                    # Fit the model
                    self.fit(train_data, target_col=target_col)
                    
                    # Make prediction for the next day
                    prediction = self.predict_next(train_data, target_col=target_col)
                    
                    # Store the prediction
                    result.loc[result.index[i], 'prediction'] = prediction
                    
                    # Generate signal based on the prediction
                    if prediction > 0:
                        result.loc[result.index[i], 'signal'] = 1  # Long
                    else:
                        result.loc[result.index[i], 'signal'] = -1  # Short
                        
                    # Log progress
                    if i % 20 == 0:
                        logger.info(f"Processed {i}/{len(result)} data points")
                except Exception as e:
                    logger.error(f"Error at step {i}: {str(e)}")
                    # Continue with next iteration
                    continue
                    
            logger.info("Signal generation completed")
            return result
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            # Return original data to prevent cascade failures
            return data
            
    def evaluate(self, data, target_col='log_return'):
        """
        Evaluate the ensemble model on test data.
        
        Args:
            data (pd.DataFrame): Test data
            target_col (str): Name of the target column
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            # Generate signals and predictions
            results = self.generate_signals(data, target_col=target_col)
            
            # Check if predictions were generated
            if 'prediction' not in results.columns or results['prediction'].isna().all():
                logger.error("No predictions were generated")
                return {
                    'mse': 999.9,
                    'rmse': 999.9,
                    'mae': 999.9,
                    'direction_accuracy': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': -1.0,
                    'final_return': 0.0,
                    'buy_hold_return': 0.0
                }
            
            # Get actual and predicted values (skipping NaN values)
            mask = ~results['prediction'].isna()
            actual = results[target_col].iloc[self.window_size:][mask].values
            predicted = results['prediction'].dropna().values
            
            # Ensure arrays are of equal length
            min_len = min(len(actual), len(predicted))
            if min_len == 0:
                logger.error("No valid predictions to evaluate")
                return {
                    'mse': 999.9,
                    'rmse': 999.9,
                    'mae': 999.9,
                    'direction_accuracy': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': -1.0,
                    'final_return': 0.0,
                    'buy_hold_return': 0.0
                }
                
            actual = actual[:min_len]
            predicted = predicted[:min_len]
            
            # Calculate metrics
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual, predicted)
            
            # Calculate direction accuracy
            actual_direction = np.sign(actual)
            predicted_direction = np.sign(predicted)
            direction_accuracy = np.mean(actual_direction == predicted_direction)
            
            # Calculate trading performance
            # Shift signals to apply them to next day's returns
            results['shifted_signal'] = results['signal'].shift(1)
            
            # Calculate strategy returns (signal * next day's return)
            results['strategy_return'] = results['shifted_signal'] * results[target_col]
            
            # Calculate cumulative returns
            results['cum_return'] = (1 + results[target_col]).cumprod()
            results['cum_strategy_return'] = (1 + results['strategy_return']).cumprod()
            
            # Calculate Sharpe ratio (assuming 252 trading days)
            sharpe_ratio = 0.0
            try:
                strategy_returns = results['strategy_return'].dropna()
                if len(strategy_returns) > 0 and np.std(strategy_returns) > 0:
                    sharpe_ratio = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(252)
            except Exception as e:
                logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            
            # Calculate max drawdown
            max_drawdown = 0.0
            try:
                cum_returns = results['cum_strategy_return'].dropna()
                if len(cum_returns) > 0:
                    running_max = cum_returns.cummax()
                    drawdown = (cum_returns / running_max) - 1
                    max_drawdown = drawdown.min()
            except Exception as e:
                logger.error(f"Error calculating max drawdown: {str(e)}")
            
            # Get final returns
            final_return = 0.0
            buy_hold_return = 0.0
            try:
                if len(results['cum_strategy_return']) > 0 and not np.isnan(results['cum_strategy_return'].iloc[-1]):
                    final_return = results['cum_strategy_return'].iloc[-1] - 1
                
                if len(results['cum_return']) > 0 and not np.isnan(results['cum_return'].iloc[-1]):
                    buy_hold_return = results['cum_return'].iloc[-1] - 1
            except Exception as e:
                logger.error(f"Error calculating final returns: {str(e)}")
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'direction_accuracy': direction_accuracy,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'final_return': final_return,
                'buy_hold_return': buy_hold_return
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {
                'mse': 999.9,
                'rmse': 999.9,
                'mae': 999.9,
                'direction_accuracy': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': -1.0,
                'final_return': 0.0,
                'buy_hold_return': 0.0
            } 