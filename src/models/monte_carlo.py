import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import logging
from scipy.stats import norm

logger = logging.getLogger(__name__)

class MonteCarloSimulator:
    def __init__(self, 
                 num_paths: int = 10000,
                 num_steps: int = 252,
                 risk_free_rate: float = 0.02):
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.risk_free_rate = risk_free_rate
        
    def simulate_price_paths(self,
                           current_price: float,
                           volatility: float,
                           days_to_expiry: int) -> np.ndarray:
        """Simulate price paths using geometric Brownian motion."""
        try:
            # Convert days to steps
            steps = int(days_to_expiry * self.num_steps / 252)
            
            # Calculate time step
            dt = days_to_expiry / (252 * steps)
            
            # Generate random paths
            paths = np.zeros((self.num_paths, steps + 1))
            paths[:, 0] = current_price
            
            # Generate random numbers
            z = np.random.standard_normal((self.num_paths, steps))
            
            # Simulate price paths
            for t in range(1, steps + 1):
                paths[:, t] = paths[:, t-1] * np.exp(
                    (self.risk_free_rate - 0.5 * volatility**2) * dt +
                    volatility * np.sqrt(dt) * z[:, t-1]
                )
            
            return paths
            
        except Exception as e:
            logger.error(f"Error simulating price paths: {str(e)}")
            raise
            
    def longstaff_schwartz(self,
                          paths: np.ndarray,
                          strike_price: float,
                          option_type: str = 'call') -> float:
        """Calculate option price using Longstaff-Schwartz algorithm."""
        try:
            num_paths, num_steps = paths.shape
            
            # Calculate payoffs at expiry
            if option_type == 'call':
                payoffs = np.maximum(paths[:, -1] - strike_price, 0)
            else:
                payoffs = np.maximum(strike_price - paths[:, -1], 0)
            
            # Initialize value function
            value = payoffs
            
            # Backward induction
            for t in range(num_steps-2, -1, -1):
                # Calculate discount factor
                discount = np.exp(-self.risk_free_rate * (1/252))
                
                # Find in-the-money paths
                if option_type == 'call':
                    itm = paths[:, t] > strike_price
                else:
                    itm = paths[:, t] < strike_price
                
                if np.sum(itm) > 0:
                    # Fit polynomial regression
                    x = paths[itm, t]
                    y = value[itm] * discount
                    
                    # Use quadratic regression
                    coeffs = np.polyfit(x, y, 2)
                    continuation_value = np.polyval(coeffs, x)
                    
                    # Update value function
                    value[itm] = np.where(
                        payoffs[itm] > continuation_value,
                        payoffs[itm],
                        continuation_value
                    )
                
                value *= discount
            
            return np.mean(value)
            
        except Exception as e:
            logger.error(f"Error in Longstaff-Schwartz algorithm: {str(e)}")
            raise
            
    def find_optimal_strike(self,
                          current_price: float,
                          volatility: float,
                          days_to_expiry: int,
                          option_type: str,
                          target_delta: float = 0.5) -> Tuple[float, float]:
        """Find optimal strike price based on target delta."""
        try:
            # Simulate price paths
            paths = self.simulate_price_paths(
                current_price=current_price,
                volatility=volatility,
                days_to_expiry=days_to_expiry
            )
            
            # Binary search for strike price
            left = current_price * 0.5
            right = current_price * 1.5
            tolerance = 0.01
            max_iterations = 100
            
            for _ in range(max_iterations):
                strike = (left + right) / 2
                price = self.longstaff_schwartz(paths, strike, option_type)
                
                # Calculate delta
                if option_type == 'call':
                    delta = self._calculate_delta(
                        current_price=current_price,
                        strike_price=strike,
                        volatility=volatility,
                        days_to_expiry=days_to_expiry
                    )
                else:
                    delta = -self._calculate_delta(
                        current_price=current_price,
                        strike_price=strike,
                        volatility=volatility,
                        days_to_expiry=days_to_expiry
                    )
                
                if abs(delta - target_delta) < tolerance:
                    return strike, price
                
                if delta > target_delta:
                    right = strike
                else:
                    left = strike
            
            # Return best approximation
            return (left + right) / 2, self.longstaff_schwartz(paths, (left + right) / 2, option_type)
            
        except Exception as e:
            logger.error(f"Error finding optimal strike: {str(e)}")
            raise
            
    def _calculate_delta(self,
                        current_price: float,
                        strike_price: float,
                        volatility: float,
                        days_to_expiry: int) -> float:
        """Calculate option delta using Black-Scholes formula."""
        try:
            t = days_to_expiry / 252
            d1 = (np.log(current_price / strike_price) + 
                  (self.risk_free_rate + 0.5 * volatility**2) * t) / (volatility * np.sqrt(t))
            
            if current_price > strike_price:
                return norm.cdf(d1)
            else:
                return -norm.cdf(-d1)
                
        except Exception as e:
            logger.error(f"Error calculating delta: {str(e)}")
            raise 