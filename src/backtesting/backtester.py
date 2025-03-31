import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from src.risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 commission: float = 0.001,  # 0.1% commission per trade
                 slippage: float = 0.001,    # 0.1% slippage
                 position_sizing: str = 'kelly'):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_sizing = position_sizing
        self.positions: Dict[str, float] = {}
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.risk_manager = RiskManager(initial_capital=initial_capital)
        
    def run_backtest(self,
                    data: pd.DataFrame,
                    predictions: pd.Series,
                    confidence_threshold: float = 0.6,
                    stop_loss: float = 0.02,
                    take_profit: float = 0.04) -> Dict:
        """Run backtest on historical data."""
        try:
            # Initialize results tracking
            self.equity_curve = [self.initial_capital]
            self.trades = []
            self.positions = {}
            self.current_capital = self.initial_capital
            
            # Iterate through each day
            for i in range(1, len(data)):
                current_price = data['Close'].iloc[i]
                current_prediction = predictions.iloc[i]
                current_volatility = data['Close'].pct_change().rolling(window=20).std().iloc[i]
                
                # Check for exit signals
                if self._should_exit_position(current_price, stop_loss, take_profit):
                    self._close_position(current_price, data.index[i])
                    continue
                
                # Check for entry signals
                if self._should_enter_position(current_prediction, confidence_threshold):
                    position_size = self._calculate_position_size(
                        current_price,
                        current_volatility,
                        current_prediction
                    )
                    self._open_position(current_price, position_size, data.index[i])
                
                # Update equity curve
                self._update_equity_curve(current_price)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(data)
            
            return {
                'equity_curve': pd.Series(self.equity_curve, index=data.index),
                'trades': self.trades,
                'performance_metrics': performance_metrics,
                'final_capital': self.current_capital,
                'total_return': (self.current_capital - self.initial_capital) / self.initial_capital
            }
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise
            
    def _should_enter_position(self, prediction: float, confidence_threshold: float) -> bool:
        """Determine if we should enter a position based on prediction and confidence."""
        try:
            # Check if we already have a position
            if self.positions:
                return False
                
            # Check if prediction exceeds confidence threshold
            return abs(prediction) > confidence_threshold
            
        except Exception as e:
            logger.error(f"Error checking entry conditions: {str(e)}")
            raise
            
    def _should_exit_position(self, current_price: float, stop_loss: float, take_profit: float) -> bool:
        """Determine if we should exit the current position."""
        try:
            if not self.positions:
                return False
                
            # Get entry price and position size
            entry_price = self.trades[-1]['price']
            position_size = self.positions['SPY']
            
            # Calculate current return
            current_return = (current_price - entry_price) / entry_price
            
            # Check stop loss and take profit conditions
            if position_size > 0:  # Long position
                return current_return <= -stop_loss or current_return >= take_profit
            else:  # Short position
                return current_return >= stop_loss or current_return <= -take_profit
                
        except Exception as e:
            logger.error(f"Error checking exit conditions: {str(e)}")
            raise
            
    def _calculate_position_size(self, 
                               price: float,
                               volatility: float,
                               prediction: float) -> float:
        """Calculate position size based on risk management rules."""
        try:
            if self.position_sizing == 'kelly':
                # Use Kelly Criterion
                position_size = self.risk_manager.calculate_position_size(
                    price=price,
                    volatility=volatility,
                    confidence=abs(prediction)
                )
            else:
                # Use fixed position sizing (e.g., 2% of capital)
                position_size = self.current_capital * 0.02
                
            # Adjust for commission and slippage
            position_size *= (1 - self.commission - self.slippage)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            raise
            
    def _open_position(self, price: float, size: float, timestamp: datetime) -> None:
        """Open a new position."""
        try:
            # Calculate number of shares
            shares = size / price
            
            # Record trade
            trade = {
                'timestamp': timestamp,
                'type': 'buy' if size > 0 else 'sell',
                'price': price,
                'shares': abs(shares),
                'value': size,
                'commission': abs(size * self.commission),
                'slippage': abs(size * self.slippage)
            }
            self.trades.append(trade)
            
            # Update position
            self.positions['SPY'] = shares
            
            # Update capital
            self.current_capital -= (size + trade['commission'] + trade['slippage'])
            
        except Exception as e:
            logger.error(f"Error opening position: {str(e)}")
            raise
            
    def _close_position(self, price: float, timestamp: datetime) -> None:
        """Close the current position."""
        try:
            if not self.positions:
                return
                
            # Calculate position value
            position_size = self.positions['SPY']
            position_value = position_size * price
            
            # Record trade
            trade = {
                'timestamp': timestamp,
                'type': 'sell' if position_size > 0 else 'buy',
                'price': price,
                'shares': abs(position_size),
                'value': position_value,
                'commission': abs(position_value * self.commission),
                'slippage': abs(position_value * self.slippage)
            }
            self.trades.append(trade)
            
            # Update capital
            self.current_capital += (position_value - trade['commission'] - trade['slippage'])
            
            # Clear position
            self.positions = {}
            
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            raise
            
    def _update_equity_curve(self, current_price: float) -> None:
        """Update the equity curve with current portfolio value."""
        try:
            if not self.positions:
                portfolio_value = self.current_capital
            else:
                position_value = self.positions['SPY'] * current_price
                portfolio_value = self.current_capital + position_value
                
            self.equity_curve.append(portfolio_value)
            
        except Exception as e:
            logger.error(f"Error updating equity curve: {str(e)}")
            raise
            
    def _calculate_performance_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        try:
            # Calculate returns
            returns = pd.Series(self.equity_curve).pct_change()
            
            # Calculate risk metrics
            risk_metrics = self.risk_manager.calculate_risk_metrics(returns)
            
            # Calculate additional metrics
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t['type'] == 'sell' and t['price'] > t['price']])
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate average trade metrics
            avg_trade_return = np.mean([t['value'] for t in self.trades]) if self.trades else 0
            avg_win = np.mean([t['value'] for t in self.trades if t['type'] == 'sell' and t['price'] > t['price']]) if winning_trades > 0 else 0
            avg_loss = np.mean([t['value'] for t in self.trades if t['type'] == 'sell' and t['price'] <= t['price']]) if losing_trades > 0 else 0
            
            # Calculate profit factor
            gross_profit = sum([t['value'] for t in self.trades if t['type'] == 'sell' and t['price'] > t['price']])
            gross_loss = abs(sum([t['value'] for t in self.trades if t['type'] == 'sell' and t['price'] <= t['price']]))
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
            
            return {
                **risk_metrics,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_trade_return': avg_trade_return,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_commission': sum(t['commission'] for t in self.trades),
                'total_slippage': sum(t['slippage'] for t in self.trades)
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            raise 