import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_price_chart(data, title='SPY Price Chart'):
    """
    Create an OHLCV price chart.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLCV data
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    try:
        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True, 
                           vertical_spacing=0.03, 
                           row_heights=[0.7, 0.3],
                           subplot_titles=(title, 'Volume'))
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add volume bar chart
        colors = ['red' if row['Open'] > row['Close'] else 'green' for _, row in data.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                marker_color=colors,
                name='Volume'
            ),
            row=2, col=1
        )
        
        # Add moving averages
        for period, color in [(20, 'blue'), (50, 'orange'), (200, 'purple')]:
            ma_col = f'MA{period}'
            data[ma_col] = data['Close'].rolling(window=period).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[ma_col],
                    line=dict(color=color, width=1),
                    name=f'{period}-day MA'
                ),
                row=1, col=1
            )
        
        # Update layout
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=800,
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text='Price ($)', row=1, col=1)
        fig.update_yaxes(title_text='Volume', row=2, col=1)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting price chart: {str(e)}")
        raise

def plot_returns_comparison(predictions, actuals, title='Predicted vs Actual Returns'):
    """
    Plot comparison of predicted vs actual returns.
    
    Args:
        predictions (pd.Series): Series of predicted returns
        actuals (pd.Series): Series of actual returns
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    try:
        # Create figure
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.1, 
                           subplot_titles=(title, 'Prediction Error'))
        
        # Add actual returns
        fig.add_trace(
            go.Scatter(
                x=actuals.index,
                y=actuals.values,
                mode='lines',
                name='Actual Returns',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Add predicted returns
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions.values,
                mode='lines',
                name='Predicted Returns',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        # Calculate and plot prediction error
        error = actuals - predictions
        
        fig.add_trace(
            go.Scatter(
                x=error.index,
                y=error.values,
                mode='lines',
                name='Prediction Error',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        
        # Add horizontal line at zero for error plot
        fig.add_trace(
            go.Scatter(
                x=[error.index.min(), error.index.max()],
                y=[0, 0],
                mode='lines',
                name='Zero Line',
                line=dict(color='black', dash='dash', width=1)
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            yaxis_tickformat='.2%',
            yaxis2_tickformat='.2%'
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text='Returns', row=1, col=1)
        fig.update_yaxes(title_text='Error', row=2, col=1)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting returns comparison: {str(e)}")
        raise

def plot_signals_on_price(data, signal_col='signal', price_col=None, title='Trading Signals'):
    """
    Plot trading signals on price chart.
    
    Args:
        data (pd.DataFrame): DataFrame with price and signal data
        signal_col (str): Name of the signal column
        price_col (str, optional): Name of the price column. If None, will use 'Adj Close' or 'Close'
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    try:
        # Determine price column if not provided
        if price_col is None:
            price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
            logger.info(f"Using {price_col} for price data in signal plot")
            
        # Create figure
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[price_col],
                mode='lines',
                name='Price',
                line=dict(color='blue')
            )
        )
        
        # Add buy signals
        buy_signals = data[data[signal_col] == 1].index
        
        if len(buy_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals,
                    y=data.loc[buy_signals, price_col],
                    mode='markers',
                    name='Buy',
                    marker=dict(symbol='triangle-up', size=15, color='green')
                )
            )
        
        # Add sell signals
        sell_signals = data[data[signal_col] == -1].index
        
        if len(sell_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals,
                    y=data.loc[sell_signals, price_col],
                    mode='markers',
                    name='Sell',
                    marker=dict(symbol='triangle-down', size=15, color='red')
                )
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=600
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting signals on price chart: {str(e)}")
        raise

def plot_correlation_matrix(data, title='Feature Correlation Matrix'):
    """
    Plot correlation matrix of features.
    
    Args:
        data (pd.DataFrame): DataFrame with features
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    try:
        # Calculate correlation matrix
        corr = data.corr()
        
        # Create figure
        fig = go.Figure()
        
        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale='RdBu',
                zmid=0,
                text=corr.values.round(2),
                texttemplate='%{text:.2f}',
                colorbar=dict(title='Correlation')
            )
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=800,
            width=800,
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting correlation matrix: {str(e)}")
        raise

def plot_model_metrics(metrics_dict, title='Model Performance Metrics'):
    """
    Plot model performance metrics.
    
    Args:
        metrics_dict (dict): Dictionary of metrics
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    try:
        # Create figure
        fig = go.Figure()
        
        # Add a bar for each metric
        metrics = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=values,
                marker_color='blue',
                text=values,
                texttemplate='%{text:.4f}',
                textposition='outside'
            )
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Metric',
            yaxis_title='Value',
            template='plotly_white',
            height=500
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting model metrics: {str(e)}")
        raise 