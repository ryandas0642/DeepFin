import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Portfolio:
    def __init__(self, initial_capital=100000):
        """
        Portfolio simulator for trading strategy.
        
        Args:
            initial_capital (float): Initial investment capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = 0  # 0: no position, 1: long, -1: short
        self.position_history = []
        self.capital_history = []
        self.trades = []
        
    def apply_signals(self, data, signal_col='signal', price_col=None):
        """
        Apply trading signals to the portfolio.
        
        Args:
            data (pd.DataFrame): Data with signals
            signal_col (str): Name of the signal column
            price_col (str, optional): Name of the price column. If None, will use 'Adj Close' or 'Close'
            
        Returns:
            pd.DataFrame: Data with portfolio metrics
        """
        try:
            logger.info("Simulating portfolio with trading signals...")
            
            # Make a copy of the data
            result = data.copy()
            
            # Determine price column if not provided
            if price_col is None:
                price_col = 'Adj Close' if 'Adj Close' in result.columns else 'Close'
                logger.info(f"Using {price_col} for price data in portfolio simulation")
            
            # Add columns for portfolio metrics
            result['position'] = 0
            result['capital'] = self.initial_capital
            result['holdings'] = 0
            result['cash'] = self.initial_capital
            result['trade_pnl'] = 0
            
            # Reset portfolio
            self.current_capital = self.initial_capital
            self.position = 0
            self.position_history = []
            self.capital_history = []
            self.trades = []
            
            # Loop through each day
            for i in range(1, len(result)):
                prev_idx = result.index[i-1]
                curr_idx = result.index[i]
                
                # Get current signal
                signal = result.loc[curr_idx, signal_col]
                
                # Get prices
                prev_price = result.loc[prev_idx, price_col]
                curr_price = result.loc[curr_idx, price_col]
                
                # Update position based on signal
                if signal != 0 and signal != self.position:
                    # Close existing position if any
                    if self.position != 0:
                        # Calculate P&L
                        if self.position == 1:  # Long
                            pnl = (curr_price - prev_price) * (result.loc[prev_idx, 'holdings'])
                        else:  # Short
                            pnl = (prev_price - curr_price) * (result.loc[prev_idx, 'holdings'])
                            
                        # Update cash and capital
                        result.loc[curr_idx, 'cash'] = result.loc[prev_idx, 'cash'] + result.loc[prev_idx, 'holdings'] * curr_price + pnl
                        result.loc[curr_idx, 'trade_pnl'] = pnl
                        
                        # Record trade
                        self.trades.append({
                            'entry_date': result.index[i-1],
                            'exit_date': curr_idx,
                            'entry_price': prev_price,
                            'exit_price': curr_price,
                            'position': self.position,
                            'pnl': pnl,
                            'return': pnl / self.current_capital
                        })
                    else:
                        result.loc[curr_idx, 'cash'] = result.loc[prev_idx, 'cash']
                    
                    # Open new position
                    self.position = signal
                    
                    # Calculate number of shares to buy/sell
                    shares = result.loc[curr_idx, 'cash'] / curr_price
                    
                    if self.position == 1:  # Long
                        result.loc[curr_idx, 'holdings'] = shares
                        result.loc[curr_idx, 'cash'] = 0
                    else:  # Short
                        result.loc[curr_idx, 'holdings'] = -shares
                        result.loc[curr_idx, 'cash'] = result.loc[curr_idx, 'cash'] + (shares * curr_price)
                        
                # If no change in position
                else:
                    # Update holdings value
                    result.loc[curr_idx, 'holdings'] = result.loc[prev_idx, 'holdings']
                    result.loc[curr_idx, 'cash'] = result.loc[prev_idx, 'cash']
                    
                    # Calculate P&L if we have a position
                    if self.position != 0:
                        if self.position == 1:  # Long
                            pnl = (curr_price - prev_price) * result.loc[curr_idx, 'holdings']
                        else:  # Short
                            pnl = (prev_price - curr_price) * abs(result.loc[curr_idx, 'holdings'])
                            
                        result.loc[curr_idx, 'trade_pnl'] = pnl
                    
                # Update position
                result.loc[curr_idx, 'position'] = self.position
                
                # Update capital
                if self.position == 1:  # Long
                    result.loc[curr_idx, 'capital'] = result.loc[curr_idx, 'holdings'] * curr_price + result.loc[curr_idx, 'cash']
                elif self.position == -1:  # Short
                    result.loc[curr_idx, 'capital'] = result.loc[curr_idx, 'cash'] - result.loc[curr_idx, 'holdings'] * curr_price
                else:  # No position
                    result.loc[curr_idx, 'capital'] = result.loc[curr_idx, 'cash']
                
                # Update current capital and save history
                self.current_capital = result.loc[curr_idx, 'capital']
                self.position_history.append(self.position)
                self.capital_history.append(self.current_capital)
                
            # Calculate cumulative P&L
            result['cum_pnl'] = result['trade_pnl'].cumsum()
            
            # Calculate returns
            result['daily_return'] = result['capital'].pct_change()
            result['cum_return'] = result['capital'] / self.initial_capital - 1
            
            # Calculate metrics
            result['drawdown'] = 1 - result['capital'] / result['capital'].cummax()
            
            logger.info("Portfolio simulation completed")
            return result
            
        except Exception as e:
            logger.error(f"Error simulating portfolio: {str(e)}")
            raise
            
    def calculate_metrics(self, data):
        """
        Calculate performance metrics for the portfolio.
        
        Args:
            data (pd.DataFrame): Data with portfolio metrics
            
        Returns:
            dict: Performance metrics
        """
        try:
            # Calculate returns
            returns = data['daily_return'].dropna()
            
            # Trading days per year (approx)
            trading_days = 252
            
            # Calculate annualized return
            total_return = data['capital'].iloc[-1] / self.initial_capital - 1
            years = len(returns) / trading_days
            cagr = (1 + total_return) ** (1 / years) - 1
            
            # Calculate Sharpe ratio
            sharpe_ratio = np.sqrt(trading_days) * (returns.mean() / returns.std())
            
            # Calculate max drawdown
            max_drawdown = data['drawdown'].max()
            
            # Calculate win rate
            if len(self.trades) > 0:
                winning_trades = sum(1 for trade in self.trades if trade['pnl'] > 0)
                win_rate = winning_trades / len(self.trades)
            else:
                win_rate = 0
                
            # Calculate profit factor
            if len(self.trades) > 0:
                gross_profit = sum(trade['pnl'] for trade in self.trades if trade['pnl'] > 0)
                gross_loss = abs(sum(trade['pnl'] for trade in self.trades if trade['pnl'] < 0))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            else:
                profit_factor = 0
                
            return {
                'total_return': total_return,
                'cagr': cagr,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'trade_count': len(self.trades)
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise
    
    def plot_equity_curve(self, data, benchmark_data=None, price_col=None):
        """
        Plot equity curve and benchmark comparison.
        
        Args:
            data (pd.DataFrame): Data with portfolio metrics
            benchmark_data (pd.DataFrame, optional): Benchmark data
            price_col (str, optional): Name of the price column. If None, will use 'Adj Close' or 'Close'
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        try:
            # Determine price column if not provided
            if price_col is None and benchmark_data is not None:
                price_col = 'Adj Close' if 'Adj Close' in benchmark_data.columns else 'Close'
                logger.info(f"Using {price_col} for benchmark data")
                
            # Create subplots
            fig = make_subplots(rows=2, cols=1, 
                               shared_xaxes=True, 
                               vertical_spacing=0.03, 
                               subplot_titles=('Portfolio Value', 'Drawdown'),
                               row_heights=[0.7, 0.3])
            
            # Add portfolio equity curve
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['capital'],
                    mode='lines',
                    name='Portfolio',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Add benchmark if provided
            if benchmark_data is not None and price_col in benchmark_data.columns:
                # Calculate benchmark value
                benchmark = benchmark_data[price_col] / benchmark_data[price_col].iloc[0] * self.initial_capital
                
                fig.add_trace(
                    go.Scatter(
                        x=benchmark_data.index,
                        y=benchmark,
                        mode='lines',
                        name='Buy & Hold',
                        line=dict(color='green')
                    ),
                    row=1, col=1
                )
            
            # Add drawdown
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['drawdown'] * 100,
                    mode='lines',
                    name='Drawdown %',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
            
            # Add buy/sell markers
            buy_signals = data[data['position'] == 1].index
            sell_signals = data[data['position'] == -1].index
            
            fig.add_trace(
                go.Scatter(
                    x=buy_signals,
                    y=data.loc[buy_signals, 'capital'],
                    mode='markers',
                    name='Buy',
                    marker=dict(symbol='triangle-up', size=10, color='green')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=sell_signals,
                    y=data.loc[sell_signals, 'capital'],
                    mode='markers',
                    name='Sell',
                    marker=dict(symbol='triangle-down', size=10, color='red')
                ),
                row=1, col=1
            )
            
            # Update layout
            fig.update_layout(
                title='Portfolio Performance',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                yaxis2_title='Drawdown (%)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=800,
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting equity curve: {str(e)}")
            raise
    
    def plot_returns_distribution(self, data):
        """
        Plot distribution of returns.
        
        Args:
            data (pd.DataFrame): Data with portfolio metrics
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        try:
            # Create figure
            fig = go.Figure()
            
            # Add daily returns histogram
            fig.add_trace(
                go.Histogram(
                    x=data['daily_return'].dropna() * 100,
                    nbinsx=50,
                    name='Daily Returns',
                    marker_color='blue',
                    opacity=0.75
                )
            )
            
            # Add normal distribution curve for comparison
            returns = data['daily_return'].dropna() * 100
            x = np.linspace(returns.min(), returns.max(), 100)
            mean, std = returns.mean(), returns.std()
            y = np.exp(-(x - mean)**2 / (2 * std**2)) / (std * np.sqrt(2 * np.pi))
            y = y * len(returns) * (returns.max() - returns.min()) / 50  # Scale to match histogram
            
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    name='Normal Distribution',
                    line=dict(color='red')
                )
            )
            
            # Update layout
            fig.update_layout(
                title='Distribution of Daily Returns',
                xaxis_title='Daily Return (%)',
                yaxis_title='Frequency',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting returns distribution: {str(e)}")
            raise 