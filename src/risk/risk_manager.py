import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from scipy.optimize import minimize
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, float] = {}
        self.trade_history: List[Dict] = []
        
    def calculate_position_size(self, 
                              price: float,
                              volatility: float,
                              confidence: float,
                              max_position_size: float = 0.1) -> float:
        """Calculate optimal position size using Kelly Criterion and risk management rules."""
        try:
            # Kelly Criterion calculation
            win_rate = confidence
            win_loss_ratio = 2.0  # Assuming 2:1 reward-to-risk ratio
            kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            
            # Adjust Kelly fraction for risk management
            kelly_fraction = min(kelly_fraction, max_position_size)
            kelly_fraction = max(kelly_fraction, 0.0)
            
            # Calculate position size in dollars
            position_size = self.current_capital * kelly_fraction
            
            # Adjust for volatility
            volatility_adjustment = 1.0 / (1.0 + volatility)
            position_size *= volatility_adjustment
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            raise
            
    def optimize_portfolio(self, 
                         returns: pd.DataFrame,
                         risk_free_rate: float = 0.02) -> Dict[str, float]:
        """Optimize portfolio weights using Modern Portfolio Theory."""
        try:
            # Calculate mean returns and covariance matrix
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # Define objective function (Sharpe ratio)
            def objective(weights):
                portfolio_return = np.sum(mean_returns * weights)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
                return -sharpe_ratio  # Negative because we're minimizing
            
            # Define constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum of weights = 1
                {'type': 'ineq', 'fun': lambda x: x}  # All weights >= 0
            ]
            
            # Initial guess (equal weights)
            n_assets = len(returns.columns)
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                constraints=constraints
            )
            
            # Create portfolio weights dictionary
            portfolio_weights = dict(zip(returns.columns, result.x))
            
            return portfolio_weights
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {str(e)}")
            raise
            
    def calculate_risk_metrics(self, 
                             returns: pd.Series,
                             risk_free_rate: float = 0.02) -> Dict[str, float]:
        """Calculate various risk metrics."""
        try:
            # Calculate basic statistics
            mean_return = returns.mean()
            std_return = returns.std()
            annualized_return = (1 + mean_return) ** 252 - 1
            annualized_volatility = std_return * np.sqrt(252)
            
            # Calculate Sharpe ratio
            excess_returns = returns - risk_free_rate/252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            
            # Calculate Sortino ratio
            downside_returns = returns[returns < 0]
            sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            # Calculate Value at Risk (VaR)
            var_95 = np.percentile(returns, 5)
            
            # Calculate Expected Shortfall (ES)
            es_95 = returns[returns <= var_95].mean()
            
            # Calculate win rate
            win_rate = (returns > 0).mean()
            
            # Calculate profit factor
            gross_profit = returns[returns > 0].sum()
            gross_loss = abs(returns[returns < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
            
            return {
                'mean_return': mean_return,
                'std_return': std_return,
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'es_95': es_95,
                'win_rate': win_rate,
                'profit_factor': profit_factor
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            raise
            
    def update_position(self, 
                       symbol: str,
                       price: float,
                       quantity: float,
                       trade_type: str) -> None:
        """Update position and trade history."""
        try:
            # Update position
            if trade_type == 'buy':
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
            elif trade_type == 'sell':
                self.positions[symbol] = self.positions.get(symbol, 0) - quantity
                
            # Record trade
            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'price': price,
                'quantity': quantity,
                'type': trade_type,
                'position_size': self.positions.get(symbol, 0)
            }
            self.trade_history.append(trade)
            
            # Update capital
            trade_value = price * quantity
            if trade_type == 'buy':
                self.current_capital -= trade_value
            elif trade_type == 'sell':
                self.current_capital += trade_value
                
        except Exception as e:
            logger.error(f"Error updating position: {str(e)}")
            raise
            
    def get_position_summary(self) -> Dict[str, Dict]:
        """Get summary of current positions and performance."""
        try:
            summary = {
                'positions': self.positions,
                'current_capital': self.current_capital,
                'total_trades': len(self.trade_history),
                'trade_history': self.trade_history
            }
            return summary
            
        except Exception as e:
            logger.error(f"Error getting position summary: {str(e)}")
            raise 