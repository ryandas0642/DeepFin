import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any
from src.data.data_collector import DataCollector
from src.data.feature_engineering import FeatureEngineer
from src.models.advanced_rnn import AdvancedDeepLearningModel
from src.risk.risk_manager import RiskManager
from src.visualization.dashboard import Dashboard
from src.backtesting.backtester import Backtester

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(ticker: str, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
    """Load and preprocess data."""
    try:
        collector = DataCollector(ticker)
        market_data = collector.get_market_data(start_date, end_date)
        return market_data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def create_features(market_data: Dict[str, pd.DataFrame], window_size: int = 60) -> tuple:
    """Create features from market data."""
    try:
        engineer = FeatureEngineer(window_size=window_size)
        features, labels = engineer.create_features(market_data)
        return features, labels
    except Exception as e:
        logger.error(f"Error creating features: {str(e)}")
        raise

def train_model(features: np.ndarray, 
                labels: np.ndarray,
                window_size: int,
                hidden_size: int,
                num_layers: int,
                dropout: float,
                model_type: str,
                quant_weight: float) -> AdvancedDeepLearningModel:
    """Train the model."""
    try:
        model = AdvancedDeepLearningModel(
            window_size=window_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            model_type=model_type,
            quant_weight=quant_weight,
            sentiment_weight=1.0 - quant_weight
        )
        
        model.fit(features, labels)
        return model
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def run_backtest(model: AdvancedDeepLearningModel,
                data: pd.DataFrame,
                features: np.ndarray,
                initial_capital: float,
                commission: float,
                slippage: float,
                position_sizing: str,
                confidence_threshold: float,
                stop_loss: float,
                take_profit: float) -> Dict:
    """Run backtest on historical data."""
    try:
        # Generate predictions
        predictions = model.predict(features)
        
        # Initialize backtester
        backtester = Backtester(
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            position_sizing=position_sizing
        )
        
        # Run backtest
        results = backtester.run_backtest(
            data=data,
            predictions=predictions,
            confidence_threshold=confidence_threshold,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        return results
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        raise

def generate_option_recommendation(collector: DataCollector,
                                current_price: float,
                                vix: float,
                                prediction: float,
                                confidence: float,
                                days_to_expiry: int = 30) -> Dict:
    """Generate options trading recommendation."""
    try:
        recommendation = collector.get_option_recommendation(
            current_price=current_price,
            vix=vix,
            prediction=prediction,
            confidence=confidence,
            days_to_expiry=days_to_expiry
        )
        return recommendation
    except Exception as e:
        logger.error(f"Error generating option recommendation: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Advanced Trading Strategy Console App')
    parser.add_argument('--ticker', type=str, default='SPY', help='Stock ticker symbol')
    parser.add_argument('--days', type=int, default=1825, help='Number of days of historical data (default: 1825 [5 years])')
    parser.add_argument('--window-size', type=int, default=60, help='Window size for the model')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden size for the model')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of layers in the model')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--model-type', type=str, default='lstm', choices=['lstm', 'gru'], help='Model type')
    parser.add_argument('--quant-weight', type=float, default=0.5, help='Weight for quantitative data')
    parser.add_argument('--initial-capital', type=float, default=100000.0, help='Initial capital for trading')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission per trade (default: 0.1%)')
    parser.add_argument('--slippage', type=float, default=0.001, help='Slippage per trade (default: 0.1%)')
    parser.add_argument('--position-sizing', type=str, default='kelly', choices=['kelly', 'fixed'], help='Position sizing method')
    parser.add_argument('--confidence-threshold', type=float, default=0.6, help='Confidence threshold for trade entry')
    parser.add_argument('--stop-loss', type=float, default=0.02, help='Stop loss percentage (default: 2%)')
    parser.add_argument('--take-profit', type=float, default=0.04, help='Take profit percentage (default: 4%)')
    parser.add_argument('--days-to-expiry', type=int, default=30, help='Days to expiry for options trades')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--dashboard', action='store_true', help='Launch interactive dashboard')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Calculate date range (default to 5 years from today)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        # Ensure we're in the post-COVID era (2020 onwards)
        covid_start = datetime(2020, 1, 1)
        if start_date < covid_start:
            logger.info("Adjusting start date to post-COVID era (2020-01-01)")
            start_date = covid_start
            args.days = (end_date - start_date).days
        
        print(f"\nLoading data for {args.ticker} from {start_date.date()} to {end_date.date()}")
        print(f"Total days of data: {args.days}")
        
        # Load data
        collector = DataCollector(args.ticker)
        market_data = load_data(args.ticker, start_date, end_date)
        
        print(f"\nLoaded {len(market_data['ohlcv'])} days of data")
        
        # Create features
        print("\nCreating features...")
        features, labels = create_features(
            market_data,
            window_size=args.window_size
        )
        
        # Train model
        print("\nTraining model...")
        model = train_model(
            features,
            labels,
            window_size=args.window_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            model_type=args.model_type,
            quant_weight=args.quant_weight
        )
        
        # Run backtest
        print("\nRunning backtest...")
        backtest_results = run_backtest(
            model=model,
            data=market_data['ohlcv'],
            features=features,
            initial_capital=args.initial_capital,
            commission=args.commission,
            slippage=args.slippage,
            position_sizing=args.position_sizing,
            confidence_threshold=args.confidence_threshold,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit
        )
        
        # Generate current prediction and option recommendation
        print("\nGenerating current prediction and option recommendation...")
        current_price = market_data['ohlcv']['Close'].iloc[-1]
        current_vix = market_data['vix']['Close'].iloc[-1]
        current_prediction = model.predict(features[-1:])[0]
        current_confidence = abs(current_prediction)
        
        option_recommendation = generate_option_recommendation(
            collector=collector,
            current_price=current_price,
            vix=current_vix,
            prediction=current_prediction,
            confidence=current_confidence,
            days_to_expiry=args.days_to_expiry
        )
        
        # Save results to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"results_{args.ticker}_{timestamp}.txt"
        
        with open(results_file, 'w') as f:
            f.write(f"Trading Strategy Results for {args.ticker}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data period: {start_date.date()} to {end_date.date()}\n")
            f.write(f"Total days: {args.days}\n\n")
            
            f.write("=== Backtest Parameters ===\n")
            f.write(f"Initial Capital: ${args.initial_capital:,.2f}\n")
            f.write(f"Commission: {args.commission*100:.2f}%\n")
            f.write(f"Slippage: {args.slippage*100:.2f}%\n")
            f.write(f"Position Sizing: {args.position_sizing}\n")
            f.write(f"Confidence Threshold: {args.confidence_threshold:.2f}\n")
            f.write(f"Stop Loss: {args.stop_loss*100:.2f}%\n")
            f.write(f"Take Profit: {args.take_profit*100:.2f}%\n\n")
            
            f.write("=== Performance Metrics ===\n")
            for metric, value in backtest_results['performance_metrics'].items():
                if isinstance(value, float):
                    f.write(f"{metric}: {value:.4f}\n")
                else:
                    f.write(f"{metric}: {value}\n")
                    
            f.write(f"\nFinal Capital: ${backtest_results['final_capital']:,.2f}\n")
            f.write(f"Total Return: {backtest_results['total_return']*100:.2f}%\n")
            
            f.write("\n=== Trade Summary ===\n")
            f.write(f"Total Trades: {len(backtest_results['trades'])}\n")
            f.write(f"Winning Trades: {backtest_results['performance_metrics']['winning_trades']}\n")
            f.write(f"Losing Trades: {backtest_results['performance_metrics']['losing_trades']}\n")
            f.write(f"Win Rate: {backtest_results['performance_metrics']['win_rate']*100:.2f}%\n")
            f.write(f"Average Trade Return: ${backtest_results['performance_metrics']['avg_trade_return']:,.2f}\n")
            f.write(f"Average Win: ${backtest_results['performance_metrics']['avg_win']:,.2f}\n")
            f.write(f"Average Loss: ${backtest_results['performance_metrics']['avg_loss']:,.2f}\n")
            f.write(f"Profit Factor: {backtest_results['performance_metrics']['profit_factor']:.2f}\n")
            f.write(f"Total Commission: ${backtest_results['performance_metrics']['total_commission']:,.2f}\n")
            f.write(f"Total Slippage: ${backtest_results['performance_metrics']['total_slippage']:,.2f}\n")
            
            f.write("\n=== Current Market Analysis ===\n")
            f.write(f"Current Price: ${current_price:,.2f}\n")
            f.write(f"Current VIX: {current_vix:.2f}\n")
            f.write(f"Model Prediction: {current_prediction:.4f}\n")
            f.write(f"Prediction Confidence: {current_confidence:.4f}\n")
            
            f.write("\n=== Options Trading Recommendation ===\n")
            f.write(f"Option Type: {option_recommendation['option_type'].upper()}\n")
            f.write(f"Strike Price: ${option_recommendation['strike_price']:,.2f}\n")
            f.write(f"Estimated Price: ${option_recommendation['estimated_price']:,.2f}\n")
            f.write(f"Days to Expiry: {option_recommendation['days_to_expiry']}\n")
            f.write(f"VIX Percentile: {option_recommendation['vix_percentile']:.2f}\n")
            f.write("\nGreeks:\n")
            for greek, value in option_recommendation['greeks'].items():
                f.write(f"{greek.upper()}: {value:.4f}\n")
        
        print(f"\nResults saved to {results_file}")
        
        # Launch dashboard if requested
        if args.dashboard:
            print("\nLaunching interactive dashboard...")
            dashboard = Dashboard()
            dashboard.run_server(debug=args.debug)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 