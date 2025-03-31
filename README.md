# DeepFin: Advanced Trading Strategy Analyzer

DeepFin is a sophisticated trading strategy analyzer that combines quantitative analysis, sentiment analysis, and market data to generate trading signals and recommendations. The system integrates VIX data and options trading capabilities to provide comprehensive market analysis and trading opportunities.

## Features

### Data Collection and Analysis
- **Market Data**: Fetches OHLCV data, VIX data, and options chain data using yfinance
- **Sentiment Analysis**: Analyzes news sentiment using FinBERT
- **Technical Indicators**: Calculates various technical indicators including:
  - Moving averages (SMA, EMA)
  - RSI, MACD, Bollinger Bands
  - Volume-based indicators
  - VIX-based indicators
  - Options-based indicators

### Model Architecture
- **Advanced RNN Models**: 
  - LSTM and GRU architectures
  - Multi-layer design with dropout
  - Quantile regression for uncertainty estimation
  - Sentiment integration
  - VIX-based volatility adjustment

### Risk Management
- **Position Sizing**: 
  - Kelly Criterion
  - Fixed position sizing
  - Risk-adjusted position sizing based on VIX
- **Portfolio Optimization**:
  - Mean-variance optimization
  - Risk parity
  - Maximum diversification
- **Risk Metrics**:
  - Value at Risk (VaR)
  - Expected Shortfall
  - Sharpe Ratio
  - Sortino Ratio
  - Maximum Drawdown

### Options Trading
- **Options Chain Analysis**:
  - Put-Call ratio analysis
  - Implied volatility analysis
  - Volume and open interest analysis
  - Strike price selection using Monte Carlo simulation
- **Options Greeks**:
  - Delta, Gamma, Theta, Vega calculations
  - Risk exposure analysis
  - Time decay consideration
- **VIX Integration**:
  - VIX percentile analysis
  - Volatility regime detection
  - Term structure analysis
  - Volatility trading signals
- **Monte Carlo Simulation**:
  - Longstaff-Schwartz algorithm for American options
  - Geometric Brownian motion price paths
  - Optimal strike selection based on target delta
  - Confidence-adjusted position sizing
  - Early exercise boundary estimation

### Backtesting Capabilities
- **Realistic Simulation**:
  - Transaction costs
  - Market impact
  - Commission handling
  - Slippage modeling
  - Position sizing rules
- **Performance Metrics**:
  - Equity curve tracking
  - Trade-by-trade analysis
  - Risk-adjusted returns
  - Transaction cost analysis
- **Risk Management Rules**:
  - Stop-loss implementation
  - Take-profit targets
  - Position sizing limits
  - Portfolio exposure controls

### Visualization and Reporting
- **Interactive Dashboard**:
  - Real-time price charts
  - Technical indicators
  - Sentiment analysis
  - VIX analysis
  - Options chain visualization
  - Monte Carlo simulation paths
- **Performance Reports**:
  - Trade history
  - Risk metrics
  - Portfolio analytics
  - Options trading analysis
  - Monte Carlo simulation results

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DeepFin.git
cd DeepFin
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main script with default parameters:
```bash
python main.py --ticker SPY
```

### Advanced Usage

Run with custom parameters:
```bash
python main.py \
    --ticker SPY \
    --days 1825 \
    --window-size 60 \
    --hidden-size 128 \
    --num-layers 2 \
    --dropout 0.2 \
    --model-type lstm \
    --quant-weight 0.5 \
    --initial-capital 100000 \
    --commission 0.001 \
    --slippage 0.001 \
    --position-sizing kelly \
    --confidence-threshold 0.6 \
    --stop-loss 0.02 \
    --take-profit 0.04 \
    --days-to-expiry 30 \
    --dashboard
```

### Command Line Arguments

- `--ticker`: Stock ticker symbol (default: SPY)
- `--days`: Number of days of historical data (default: 1825 [5 years])
- `--window-size`: Window size for the model (default: 60)
- `--hidden-size`: Hidden size for the model (default: 128)
- `--num-layers`: Number of layers in the model (default: 2)
- `--dropout`: Dropout rate (default: 0.2)
- `--model-type`: Model type (lstm or gru) (default: lstm)
- `--quant-weight`: Weight for quantitative data (default: 0.5)
- `--initial-capital`: Initial capital for trading (default: 100000)
- `--commission`: Commission per trade (default: 0.1%)
- `--slippage`: Slippage per trade (default: 0.1%)
- `--position-sizing`: Position sizing method (kelly or fixed) (default: kelly)
- `--confidence-threshold`: Confidence threshold for trade entry (default: 0.6)
- `--stop-loss`: Stop loss percentage (default: 2%)
- `--take-profit`: Take profit percentage (default: 4%)
- `--days-to-expiry`: Days to expiry for options trades (default: 30)
- `--debug`: Enable debug mode
- `--dashboard`: Launch interactive dashboard

## Output

The script generates a detailed results file containing:
- Backtest parameters
- Performance metrics
- Trade summary
- Current market analysis
- Options trading recommendations
- Greeks calculations
- Monte Carlo simulation results
- Early exercise boundaries

## Project Structure

```
DeepFin/
├── src/
│   ├── data/
│   │   ├── data_collector.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── advanced_rnn.py
│   │   └── monte_carlo.py
│   ├── risk/
│   │   └── risk_manager.py
│   ├── backtesting/
│   │   └── backtester.py
│   └── visualization/
│       └── dashboard.py
├── main.py
├── requirements.txt
└── README.md
```

## Performance Metrics

The backtesting system provides comprehensive performance metrics:

### Returns
- Total Return
- Annualized Return
- Risk-Adjusted Returns (Sharpe, Sortino)

### Risk Metrics
- Maximum Drawdown
- Value at Risk (VaR)
- Expected Shortfall (ES)
- Beta
- Alpha

### Trading Statistics
- Total Trades
- Win Rate
- Profit Factor
- Average Trade Return
- Average Win/Loss
- Largest Win/Loss

### Cost Analysis
- Total Commission
- Total Slippage
- Net Profit After Costs

### Monte Carlo Analysis
- Price Path Simulations
- Early Exercise Probabilities
- Optimal Exercise Boundaries
- Delta-Neutral Positions
- Risk-Adjusted Returns

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Do not use it for actual trading without proper testing and validation. The authors are not responsible for any financial losses incurred through the use of this software. 