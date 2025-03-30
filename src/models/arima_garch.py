import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import logging
import warnings
from itertools import product
from sklearn.metrics import mean_squared_error

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ARIMAGARCHModel:
    def __init__(self, max_p=3, max_d=1, max_q=3, max_garch_p=1, max_garch_q=1):
        """
        Initialize ARIMA+GARCH model.
        
        Args:
            max_p (int): Maximum value for ARIMA p parameter (AR order)
            max_d (int): Maximum value for ARIMA d parameter (differencing)
            max_q (int): Maximum value for ARIMA q parameter (MA order)
            max_garch_p (int): Maximum value for GARCH p parameter
            max_garch_q (int): Maximum value for GARCH q parameter
        """
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.max_garch_p = max_garch_p
        self.max_garch_q = max_garch_q
        self.arima_model = None
        self.garch_model = None
        self.arima_order = None
        self.garch_order = None
        self.arima_residuals = None
        
    def _find_best_arima(self, data):
        """
        Find the best ARIMA model using AIC.
        
        Args:
            data (pd.Series): Time series data
            
        Returns:
            tuple: (best_order, best_model)
        """
        best_aic = float('inf')
        best_order = None
        best_model = None
        
        # Try different ARIMA parameters
        for p, d, q in product(range(0, self.max_p + 1), 
                              range(0, self.max_d + 1), 
                              range(0, self.max_q + 1)):
            # Skip if all parameters are 0
            if p == 0 and d == 0 and q == 0:
                continue
                
            try:
                model = ARIMA(data, order=(p, d, q))
                results = model.fit()
                
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, d, q)
                    best_model = results
                    
            except Exception as e:
                # Skip models that don't converge
                continue
                
        if best_model is None:
            # If no model converges, use a simple model
            try:
                best_model = ARIMA(data, order=(1, 0, 0)).fit()
                best_order = (1, 0, 0)
            except:
                # If even the simple model fails, raise an error
                raise ValueError("Could not find a suitable ARIMA model")
                
        return best_order, best_model
    
    def _find_best_garch(self, residuals):
        """
        Find the best GARCH model using AIC.
        
        Args:
            residuals (pd.Series): Residuals from ARIMA model
            
        Returns:
            tuple: (best_order, best_model)
        """
        best_aic = float('inf')
        best_order = None
        best_model = None
        
        # Try different GARCH parameters
        for p, q in product(range(1, self.max_garch_p + 1), 
                           range(1, self.max_garch_q + 1)):
            try:
                model = arch_model(residuals, vol='GARCH', p=p, q=q)
                results = model.fit(disp='off')
                
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, q)
                    best_model = results
                    
            except Exception as e:
                # Skip models that don't converge
                continue
                
        if best_model is None:
            # If no model converges, use a simple model
            try:
                best_model = arch_model(residuals, vol='GARCH', p=1, q=1).fit(disp='off')
                best_order = (1, 1)
            except:
                # If even the simple model fails, use a simple model with low persistence
                try:
                    best_model = arch_model(residuals, vol='GARCH', p=1, q=1, rescale=True).fit(disp='off')
                    best_order = (1, 1)
                except:
                    # If everything fails, raise an error
                    raise ValueError("Could not find a suitable GARCH model")
                    
        return best_order, best_model
    
    def fit(self, data):
        """
        Fit the ARIMA+GARCH model to the data.
        
        Args:
            data (pd.Series): Time series data
            
        Returns:
            self
        """
        try:
            logger.info("Fitting ARIMA model...")
            self.arima_order, self.arima_model = self._find_best_arima(data)
            logger.info(f"Best ARIMA order: {self.arima_order}")
            
            # Get residuals from ARIMA model
            self.arima_residuals = pd.Series(self.arima_model.resid, index=data.index[self.arima_model.loglikelihood_burn:])
            
            logger.info("Fitting GARCH model on ARIMA residuals...")
            self.garch_order, self.garch_model = self._find_best_garch(self.arima_residuals)
            logger.info(f"Best GARCH order: GARCH{self.garch_order}")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA+GARCH model: {str(e)}")
            raise
    
    def predict(self, steps=1):
        """
        Forecast future values and volatility.
        
        Args:
            steps (int): Number of steps to forecast
            
        Returns:
            tuple: (forecasted_mean, forecasted_variance)
        """
        if self.arima_model is None or self.garch_model is None:
            raise ValueError("Model must be fit before making predictions")
            
        try:
            # Get ARIMA forecast
            arima_forecast = self.arima_model.forecast(steps=steps)
            
            # Ensure forecast is in the right format
            if isinstance(arima_forecast, pd.Series):
                # Series is already good, just extract first value if only one step
                if steps == 1:
                    arima_forecast_flat = arima_forecast.iloc[0]
                else:
                    arima_forecast_flat = arima_forecast.values.flatten()
            elif isinstance(arima_forecast, np.ndarray):
                # Flatten array
                arima_forecast_flat = arima_forecast.flatten()
                # If one step, extract single value
                if steps == 1 and len(arima_forecast_flat) > 0:
                    arima_forecast_flat = arima_forecast_flat[0]
            else:
                # Try to convert to a flat array
                try:
                    arima_forecast_flat = np.array(arima_forecast).flatten()
                    if steps == 1 and len(arima_forecast_flat) > 0:
                        arima_forecast_flat = arima_forecast_flat[0]
                except:
                    # If all else fails, return a default value
                    logger.warning("Could not process ARIMA forecast, returning default value")
                    arima_forecast_flat = 0.0
            
            # Get GARCH forecast
            garch_forecast = self.garch_model.forecast(horizon=steps)
            
            # Extract conditional variance
            if hasattr(garch_forecast, 'variance'):
                forecasted_variance = garch_forecast.variance.iloc[-1].values
            else:
                # For older versions of arch package
                try:
                    forecasted_variance = np.array([garch_forecast._forecasts['h'].values[-1][0]])
                except:
                    forecasted_variance = np.array([0.01])  # Default value
            
            # Ensure variance is the right shape
            forecasted_variance = forecasted_variance.flatten()
            
            return arima_forecast_flat, forecasted_variance
            
        except Exception as e:
            logger.error(f"Error making forecast: {str(e)}")
            # Return default values to prevent app crash
            return np.array([0.0]), np.array([0.01])
    
    def evaluate(self, test_data):
        """
        Evaluate the model on test data.
        
        Args:
            test_data (pd.Series): Test data
            
        Returns:
            dict: Evaluation metrics
        """
        if self.arima_model is None or self.garch_model is None:
            raise ValueError("Model must be fit before evaluation")
            
        try:
            # Make one-step forecasts
            forecasts = []
            
            for i in range(len(test_data)):
                # Combine training data with test data up to this point
                forecast, _ = self.predict(steps=1)
                if isinstance(forecast, (np.ndarray, list)):
                    forecasts.append(forecast[0] if len(forecast) > 0 else 0.0)
                else:
                    forecasts.append(forecast)
                
                # Refit the model with one more observation
                if i < len(test_data) - 1:
                    combined_data = pd.concat([self.arima_model.data.orig_y, test_data[:i+1]])
                    self.fit(combined_data)
            
            # Calculate metrics
            mse = mean_squared_error(test_data, forecasts)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(test_data - forecasts))
            
            # Calculate accuracy of direction prediction
            actual_direction = np.sign(test_data.diff().values[1:])
            predicted_direction = np.sign(np.diff(forecasts))
            direction_accuracy = np.mean(actual_direction == predicted_direction)
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'direction_accuracy': direction_accuracy
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise 