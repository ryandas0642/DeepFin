# Advanced Trading Strategy Analyzer

A sophisticated trading strategy analysis tool that combines ARIMA+GARCH models with deep learning to provide comprehensive market analysis and trading recommendations. This tool is particularly focused on options trading strategies for major ETFs like SPY. The tool analyzes market data from the post-COVID era (2020-2024) to generate options trade recommendations (buy a call or a put) for the SPY ETF, trained on both quantitative and sentiment data for S&P500.

## System Requirements and Hardware Constraints

### Current System Requirements
- CPU: AMD Ryzen 5 7600X or equivalent
- GPU: NVIDIA RTX 4080 or equivalent
- RAM: 32GB recommended
- Storage: 1TB SSD recommended

### Data Limitations
The current implementation is constrained by:
- Hardware limitations of consumer-grade GPUs
- Data availability from free sources (Yahoo Finance)
- Memory constraints for large-scale model training
- Processing power limitations for extensive historical data

### Future Hardware Optimizations

#### GPU Acceleration
- **CUDA Development**
  - Parallel processing implementation
  - GPU-accelerated model training
  - Batch processing optimization
  - Memory management improvements
  - Multi-GPU support

#### Advanced GPU Support
- NVIDIA H100/H200 integration
- NVIDIA A100 support
- Multi-GPU training capabilities
- Distributed training framework
- GPU memory optimization

#### Performance Improvements
- Multi-threading implementation
- Parallel data processing
- Distributed computing support
- Memory-efficient algorithms
- Batch size optimization

### Future Data Enhancements
- Integration with premium data sources
- Extended historical data analysis
- Real-time data processing
- Alternative data sources
- Market microstructure data

## Features

- **Hybrid Model Architecture**
  - ARIMA+GARCH for volatility modeling and risk assessment
  - Advanced RNN (LSTM/GRU) for price prediction
  - Multi-task learning for simultaneous price and volatility prediction
  - Attention mechanism for better feature extraction

- **Comprehensive Analysis**
  - Price prediction with confidence intervals
  - Volatility forecasting
  - Options trading recommendations
  - Risk metrics calculation (Sharpe ratio, Sortino ratio, etc.)
  - Maximum drawdown analysis
  - Win rate and profit factor calculations

- **Data Integration**
  - Quantitative data analysis (price, volume, technical indicators)
  - Sentiment analysis integration (news and social media)
  - Configurable data weights for quantitative vs. sentiment analysis

## Data Sources and Processing

### Quantitative Data (Yahoo Finance)

The tool fetches comprehensive OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance for the post-COVID era (2020-2024), including:

- **Price Data**
  - Daily open, high, low, and closing prices
  - Adjusted closing prices for accurate historical analysis
  - Volume data for liquidity analysis
  - Intraday price movements
  - Price gaps and overnight changes
  - COVID-19 market impact analysis
  - Post-pandemic recovery patterns

- **Technical Indicators**
  - Moving averages (SMA, EMA)
    - Multiple timeframes (5, 10, 20, 50, 200 days)
    - Exponential weighting schemes
    - Adaptive moving averages
  - Relative Strength Index (RSI)
    - Multiple periods (6, 12, 24)
    - Divergence analysis
    - Overbought/oversold levels
  - MACD (Moving Average Convergence Divergence)
    - Custom fast/slow periods
    - Signal line analysis
    - Histogram patterns
  - Bollinger Bands
    - Multiple standard deviations
    - Band width analysis
    - Breakout detection
  - Volume-based indicators
    - On-balance volume (OBV)
    - Volume-weighted average price (VWAP)
    - Money flow index
    - Volume profile analysis
  - Momentum indicators
    - Rate of change (ROC)
    - Stochastic oscillator
    - Williams %R
    - True strength index

- **Derived Features**
  - Log returns
    - Daily returns
    - Rolling returns
    - Annualized returns
  - Price momentum
    - Multiple timeframes
    - Cross-asset momentum
    - Mean reversion signals
  - Volatility measures
    - Historical volatility
    - Implied volatility
    - Volatility term structure
    - Volatility skew analysis
  - Trading volume analysis
    - Volume profile
    - Volume-weighted metrics
    - Liquidity analysis
  - Price patterns and formations
    - Candlestick patterns
    - Chart patterns
    - Support/resistance levels
    - Fibonacci retracements

### Sentiment Analysis

The tool integrates FinBERT, a pre-trained BERT model specifically fine-tuned for financial sentiment analysis, to process market sentiment data:

- **FinBERT Implementation**
  - Pre-trained financial sentiment model
    - Fine-tuned on financial news and reports
    - Domain-specific vocabulary
    - Financial context understanding
  - Sentiment classification
    - Positive/Negative/Neutral classification
    - Confidence scores
    - Aspect-based sentiment
  - Multi-source integration
    - Financial news articles
    - Earnings reports
    - SEC filings
    - Social media discussions
  - Temporal alignment
    - Daily sentiment aggregation
    - Rolling sentiment windows
    - Sentiment momentum

- **Sentiment Feature Engineering**
  - Daily sentiment scores
    - Aggregated sentiment metrics
    - Sentiment volatility
    - Sentiment trends
  - Sentiment-based indicators
    - Sentiment momentum
    - Sentiment divergence
    - Sentiment regime detection
  - Cross-asset sentiment
    - Sector sentiment correlation
    - Market breadth sentiment
    - Global sentiment indicators

- **RNN Integration**
  - Sentiment embedding layer
    - FinBERT output processing
    - Feature dimension reduction
    - Temporal alignment
  - Multi-modal attention
    - Price-sentiment attention
    - Cross-feature attention
    - Dynamic weighting
  - Sentiment impact modeling
    - Lagged sentiment effects
    - Sentiment persistence
    - Threshold effects
  - Training process
    - Joint sentiment-price training
    - Sentiment-specific loss terms
    - Adaptive weighting

## Model Architecture

The application uses a sophisticated hybrid model architecture:

### 1. ARIMA+GARCH Component

- **ARIMA Model**
  - Automatic parameter selection (p,d,q)
    - Grid search optimization
    - AIC/BIC criteria
    - Cross-validation
  - AIC-based model optimization
    - Information criteria comparison
    - Model complexity trade-off
    - Overfitting prevention
  - Seasonal decomposition
    - STL decomposition
    - Seasonal adjustment
    - Trend extraction
  - Trend analysis
    - Linear trend detection
    - Nonlinear trend modeling
    - Breakpoint detection
  - Residual diagnostics
    - ACF/PACF analysis
    - Ljung-Box test
    - Jarque-Bera test
    - QQ plots

- **GARCH Model**
  - Volatility clustering detection
    - ARCH effects testing
    - Clustering analysis
    - Regime detection
  - Conditional variance modeling
    - GARCH(1,1) specification
    - Asymmetric GARCH
    - Long-memory GARCH
  - Risk assessment
    - Value at Risk (VaR)
    - Expected Shortfall
    - Tail risk analysis
  - Tail risk analysis
    - Extreme value theory
    - Tail index estimation
    - Tail dependence
  - Volatility forecasting
    - Multi-step ahead forecasts
    - Confidence intervals
    - Volatility term structure

### 2. Advanced RNN Architecture

The custom RNN implementation features:

- **Core Architecture**
  - LSTM/GRU layers with configurable depth
    - Stacked architecture
    - Residual connections
    - Layer-wise dropout
  - Bidirectional processing
    - Forward/backward passes
    - Context integration
    - Feature fusion
  - Skip connections for gradient flow
    - Highway connections
    - Dense connections
    - Attention shortcuts
  - Layer normalization for stable training
    - Batch normalization
    - Instance normalization
    - Adaptive normalization
  - Dropout for regularization
    - Spatial dropout
    - Recurrent dropout
    - Variational dropout

- **Attention Mechanism**
  - Multi-head self-attention
    - Query-key-value projections
    - Scaled dot-product attention
    - Positional encoding
  - Temporal attention for time series
    - Time-aware attention
    - Seasonal attention
    - Trend attention
  - Feature importance scoring
    - Feature-wise attention
    - Cross-feature attention
    - Dynamic weighting
  - Cross-attention between price and sentiment
    - Multi-modal attention
    - Feature alignment
    - Information fusion
  - Dynamic weight adjustment
    - Adaptive attention
    - Context-dependent weights
    - Market regime adaptation

- **Multi-task Learning**
  - Joint price and volatility prediction
    - Shared feature extraction
    - Task-specific heads
    - Loss weighting
  - Sentiment impact analysis
    - Sentiment feature extraction
    - Impact quantification
    - Temporal alignment
  - Risk assessment integration
    - Risk factor modeling
    - Uncertainty estimation
    - Confidence scoring
  - Options pricing components
    - Implied volatility prediction
    - Greeks calculation
    - Term structure modeling
  - Market regime classification
    - Regime detection
    - Transition modeling
    - State prediction

- **Training Process**
  - Adaptive learning rates
    - Learning rate scheduling
    - Warm-up periods
    - Cyclical learning rates
  - Gradient clipping
    - Norm-based clipping
    - Value-based clipping
    - Adaptive clipping
  - Early stopping
    - Validation monitoring
    - Patience scheduling
    - Model checkpointing
  - Model checkpointing
    - Best model saving
    - State preservation
    - Recovery mechanisms
  - Cross-validation
    - Time series splits
    - Rolling window validation
    - Performance metrics

### 3. Ensemble Approach

- **Model Integration**
  - Weighted combination of predictions
    - Dynamic weighting
    - Performance-based weights
    - Uncertainty-based weights
  - Dynamic weight adjustment
    - Online weight updates
    - Adaptive weighting
    - Market regime adaptation
  - Confidence scoring
    - Model confidence
    - Prediction uncertainty
    - Ensemble agreement
  - Uncertainty quantification
    - Prediction intervals
    - Confidence bounds
    - Risk assessment
  - Ensemble diversity metrics
    - Model correlation
    - Prediction variance
    - Information gain

- **Feature Integration**
  - Quantitative data weighting
    - Feature importance
    - Information content
    - Stability metrics
  - Sentiment data weighting
    - Sentiment reliability
    - Temporal relevance
    - Source credibility
  - Technical indicator importance
    - Indicator effectiveness
    - Redundancy analysis
    - Feature selection
  - Market regime consideration
    - Regime detection
    - Feature adaptation
    - Model switching
  - Risk factor integration
    - Risk decomposition
    - Factor modeling
    - Risk attribution

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd trading-strategy-analyzer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional, for sentiment analysis):
Create a `.env` file with your API keys:
```
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret
```

## Usage

Run the application from the command line:

```bash
python main.py [options]
```

### Command Line Options

- `--ticker`: Stock ticker symbol (default: SPY)
- `--days`: Number of days of historical data (default: 1825 [5 years])
- `--window-size`: Window size for the model (default: 60)
- `--hidden-size`: Hidden size for the model (default: 128)
- `--num-layers`: Number of layers in the model (default: 2)
- `--dropout`: Dropout rate (default: 0.2)
- `--model-type`: Model type (lstm or gru, default: lstm)
- `--quant-weight`: Weight for quantitative data (default: 0.5)
- `--debug`: Enable debug mode for detailed logging

### Example Commands

Basic usage:
```bash
python main.py
```

Custom analysis for AAPL:
```bash
python main.py --ticker AAPL --days 1825 --window-size 30 --model-type gru --quant-weight 0.7
```

With debug mode:
```bash
python main.py --debug
```

## Output

The application generates two types of output:

1. **Console Output**
   - Trading recommendations
   - Risk analysis metrics
   - Progress updates during model training

2. **Results File**
   - Detailed analysis saved to `results_[TICKER]_[TIMESTAMP].txt`
   - Includes all predictions and metrics
   - Timestamped for easy tracking

## Risk Management

The tool includes comprehensive risk management features:

- Volatility forecasting
  - Historical volatility
  - Implied volatility
  - Volatility term structure
- Maximum drawdown calculation
  - Rolling drawdowns
  - Peak-to-trough analysis
  - Recovery periods
- Sharpe and Sortino ratios
  - Risk-adjusted returns
  - Downside risk metrics
  - Performance attribution
- Win rate analysis
  - Trade success rate
  - Profit factor
  - Risk-reward ratio
- Profit factor calculation
  - Gross profit/loss
  - Net profit/loss
  - Transaction costs
- Confidence scoring for recommendations
  - Model confidence
  - Market regime
  - Risk factors

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for educational and research purposes only. Trading involves significant risk, and past performance does not guarantee future results. Always do your own research and consider consulting with financial advisors before making investment decisions. 