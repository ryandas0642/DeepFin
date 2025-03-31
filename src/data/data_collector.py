import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import time
from concurrent.futures import ThreadPoolExecutor
import ta
from ..models.monte_carlo import MonteCarloSimulator

logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.vix_ticker = "^VIX"  # VIX index ticker
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.mc_simulator = MonteCarloSimulator()
        
    def get_ohlcv_data(self, start_date: datetime, end_date: datetime, interval: str = "1d") -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(self.ticker)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {str(e)}")
            raise
            
    def get_option_chain(self, expiry_date: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch option chain data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(self.ticker)
            if expiry_date is None:
                # Get the next monthly expiry
                expirations = ticker.options
                expiry_date = datetime.strptime(expirations[0], "%Y-%m-%d")
            
            options = ticker.option_chain(expiry_date.strftime("%Y-%m-%d"))
            calls = options.calls
            puts = options.puts
            
            # Combine calls and puts
            calls['type'] = 'call'
            puts['type'] = 'put'
            options_df = pd.concat([calls, puts])
            
            return options_df
        except Exception as e:
            logger.error(f"Error fetching option chain: {str(e)}")
            raise
            
    def get_sentiment_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Collect and analyze sentiment data from various sources."""
        try:
            # Get news articles
            news_df = self._get_news_articles(start_date, end_date)
            
            # Analyze sentiment using FinBERT
            sentiments = self._analyze_sentiment(news_df['title'].tolist())
            
            # Create sentiment DataFrame
            sentiment_df = pd.DataFrame({
                'date': news_df['date'],
                'title': news_df['title'],
                'sentiment': sentiments,
                'source': news_df['source']
            })
            
            # Aggregate daily sentiment
            daily_sentiment = sentiment_df.groupby('date')['sentiment'].agg(['mean', 'std', 'count'])
            
            return daily_sentiment
        except Exception as e:
            logger.error(f"Error collecting sentiment data: {str(e)}")
            raise
            
    def _get_news_articles(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch news articles from various sources."""
        # Implementation for news scraping
        # This is a placeholder - you would need to implement actual news scraping
        # from sources like Yahoo Finance, Reuters, Bloomberg, etc.
        pass
        
    def _analyze_sentiment(self, texts: List[str]) -> List[float]:
        """Analyze sentiment using FinBERT model."""
        try:
            sentiments = []
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                outputs = self.model(**inputs)
                sentiment_score = torch.softmax(outputs.logits, dim=1)[0][1].item()  # Positive sentiment score
                sentiments.append(sentiment_score)
            return sentiments
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            raise
            
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the DataFrame."""
        try:
            # Moving averages
            df['sma_5'] = ta.trend.sma_indicator(df['Close'], window=5)
            df['sma_10'] = ta.trend.sma_indicator(df['Close'], window=10)
            df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['sma_200'] = ta.trend.sma_indicator(df['Close'], window=200)
            
            # RSI
            df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
            
            # MACD
            macd = ta.trend.macd_diff(df['Close'])
            df['macd'] = macd
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['bb_high'] = bollinger.bollinger_hband()
            df['bb_low'] = bollinger.bollinger_lband()
            df['bb_mid'] = bollinger.bollinger_mavg()
            
            # Volume indicators
            df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            df['vwap'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
            
            # Momentum indicators
            df['roc'] = ta.momentum.roc(df['Close'])
            df['stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
            
            return df
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            raise
            
    def get_market_data(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch comprehensive market data including OHLCV, VIX, and options data."""
        try:
            # Fetch OHLCV data
            ticker = yf.Ticker(self.ticker)
            ohlcv = ticker.history(start=start_date, end=end_date)
            
            # Fetch VIX data
            vix = yf.Ticker(self.vix_ticker)
            vix_data = vix.history(start=start_date, end=end_date)
            
            # Fetch options chain data
            options = ticker.options
            options_data = {}
            
            # Get options data for the next 4 expiration dates
            for expiry in options[:4]:
                try:
                    opt = ticker.option_chain(expiry)
                    options_data[expiry] = {
                        'calls': opt.calls,
                        'puts': opt.puts
                    }
                except Exception as e:
                    logger.warning(f"Error fetching options data for expiry {expiry}: {str(e)}")
                    continue
            
            return {
                'ohlcv': ohlcv,
                'vix': vix_data,
                'options': options_data
            }
            
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            raise
            
    def get_option_recommendation(self, 
                                current_price: float,
                                vix: float,
                                prediction: float,
                                confidence: float,
                                days_to_expiry: int = 30) -> Dict:
        """Generate options trading recommendation based on model prediction and VIX."""
        try:
            # Calculate implied volatility percentile
            vix_percentile = self._calculate_vix_percentile(vix)
            
            # Determine option type based on prediction
            option_type = 'call' if prediction > 0 else 'put'
            
            # Convert VIX to daily volatility
            daily_vol = vix / np.sqrt(252)
            
            # Find optimal strike price using Monte Carlo simulation
            target_delta = 0.5 * confidence  # Adjust target delta based on confidence
            strike_price, estimated_price = self.mc_simulator.find_optimal_strike(
                current_price=current_price,
                volatility=daily_vol,
                days_to_expiry=days_to_expiry,
                option_type=option_type,
                target_delta=target_delta
            )
            
            # Round strike price to nearest standard option strike
            strike_price = round(strike_price / 5) * 5
            
            # Calculate Greeks
            greeks = self._calculate_greeks(
                current_price=current_price,
                strike_price=strike_price,
                vix=vix,
                days_to_expiry=days_to_expiry,
                option_type=option_type
            )
            
            return {
                'option_type': option_type,
                'strike_price': strike_price,
                'estimated_price': estimated_price,
                'days_to_expiry': days_to_expiry,
                'greeks': greeks,
                'vix_percentile': vix_percentile,
                'confidence': confidence,
                'prediction': prediction,
                'target_delta': target_delta
            }
            
        except Exception as e:
            logger.error(f"Error generating option recommendation: {str(e)}")
            raise
            
    def _calculate_vix_percentile(self, current_vix: float) -> float:
        """Calculate the percentile of current VIX value."""
        try:
            # Fetch historical VIX data for the last year
            vix = yf.Ticker(self.vix_ticker)
            hist_vix = vix.history(period='1y')['Close']
            
            # Calculate percentile
            percentile = np.percentile(hist_vix, current_vix)
            return percentile
            
        except Exception as e:
            logger.error(f"Error calculating VIX percentile: {str(e)}")
            raise
            
    def _calculate_greeks(self,
                         current_price: float,
                         strike_price: float,
                         vix: float,
                         days_to_expiry: int,
                         option_type: str) -> Dict[str, float]:
        """Calculate option Greeks."""
        try:
            # Convert VIX to daily volatility
            daily_vol = vix / np.sqrt(252)
            
            # Calculate time to expiry in years
            t = days_to_expiry / 365
            
            # Calculate d1 and d2
            d1 = (np.log(current_price / strike_price) + 
                  (0.5 * daily_vol**2) * t) / (daily_vol * np.sqrt(t))
            d2 = d1 - daily_vol * np.sqrt(t)
            
            # Calculate Greeks
            if option_type == 'call':
                delta = self._normal_cdf(d1)
                gamma = self._normal_pdf(d1) / (current_price * daily_vol * np.sqrt(t))
                theta = (-current_price * self._normal_pdf(d1) * daily_vol / 
                        (2 * np.sqrt(t)) - 
                        0.02 * strike_price * np.exp(-0.02 * t) * self._normal_cdf(d2))
                vega = current_price * np.sqrt(t) * self._normal_pdf(d1) / 100
            else:
                delta = -self._normal_cdf(-d1)
                gamma = self._normal_pdf(d1) / (current_price * daily_vol * np.sqrt(t))
                theta = (-current_price * self._normal_pdf(d1) * daily_vol / 
                        (2 * np.sqrt(t)) + 
                        0.02 * strike_price * np.exp(-0.02 * t) * self._normal_cdf(-d2))
                vega = current_price * np.sqrt(t) * self._normal_pdf(d1) / 100
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega
            }
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {str(e)}")
            raise
            
    def _normal_cdf(self, x: float) -> float:
        """Calculate cumulative distribution function of standard normal distribution."""
        return 0.5 * (1 + np.erf(x / np.sqrt(2)))
        
    def _normal_pdf(self, x: float) -> float:
        """Calculate probability density function of standard normal distribution."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi) 