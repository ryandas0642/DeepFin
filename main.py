import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any
from src.data.data_loader import DataLoader
from src.models.advanced_rnn import AdvancedDeepLearningModel
from src.data.sentiment_analyzer import SentimentAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Load and preprocess data."""
    try:
        loader = DataLoader(ticker)
        data = loader.load_data(start_date, end_date)
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def print_predictions(predictions: Dict[str, Any]) -> None:
    """Print predictions in a formatted way."""
    print("\n=== Trading Recommendations ===")
    print(f"Recommended Action: {predictions['action']}")
    print(f"Confidence: {predictions['confidence']*100:.1f}%")
    print(f"Strike Price: ${predictions['strike_price']:.2f}")
    print(f"Expiry Date: {predictions['expiry_date']}")
    print(f"Predicted Return: {predictions['predicted_return']*100:.2f}%")
    print(f"Predicted Volatility: {predictions['predicted_volatility']*100:.2f}%")

def print_risk_metrics(data: pd.DataFrame) -> None:
    """Print risk metrics in a formatted way."""
    print("\n=== Risk Analysis ===")
    print(f"Current Volatility: {data['volatility'].iloc[-1]*100:.2f}%")
    print(f"Max Drawdown: {data['max_drawdown'].iloc[-1]*100:.2f}%")
    print(f"Sharpe Ratio: {data['sharpe_ratio'].iloc[-1]:.2f}")
    print(f"Sortino Ratio: {data['sortino_ratio'].iloc[-1]:.2f}")
    print(f"Win Rate: {data['win_rate'].iloc[-1]*100:.1f}%")
    print(f"Profit Factor: {data['profit_factor'].iloc[-1]:.2f}")

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
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
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
        
        data = load_data(args.ticker, start_date, end_date)
        
        print(f"\nLoaded {len(data)} days of data")
        
        # Initialize and train model
        print("\nInitializing model...")
        model = AdvancedDeepLearningModel(
            window_size=args.window_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            model_type=args.model_type,
            quant_weight=args.quant_weight,
            sentiment_weight=1.0 - args.quant_weight
        )
        
        print("Training model...")
        model.fit(data)
        
        # Make predictions
        print("Generating predictions...")
        predictions = model.predict(data)
        
        # Print results
        print_predictions(predictions)
        print_risk_metrics(data)
        
        # Save results to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"results_{args.ticker}_{timestamp}.txt"
        
        with open(results_file, 'w') as f:
            f.write(f"Trading Strategy Results for {args.ticker}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data period: {start_date.date()} to {end_date.date()}\n")
            f.write(f"Total days: {args.days}\n\n")
            f.write("=== Trading Recommendations ===\n")
            for key, value in predictions.items():
                f.write(f"{key}: {value}\n")
            f.write("\n=== Risk Analysis ===\n")
            f.write(f"Current Volatility: {data['volatility'].iloc[-1]*100:.2f}%\n")
            f.write(f"Max Drawdown: {data['max_drawdown'].iloc[-1]*100:.2f}%\n")
            f.write(f"Sharpe Ratio: {data['sharpe_ratio'].iloc[-1]:.2f}\n")
            f.write(f"Sortino Ratio: {data['sortino_ratio'].iloc[-1]:.2f}\n")
            f.write(f"Win Rate: {data['win_rate'].iloc[-1]*100:.1f}%\n")
            f.write(f"Profit Factor: {data['profit_factor'].iloc[-1]:.2f}\n")
        
        print(f"\nResults saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 