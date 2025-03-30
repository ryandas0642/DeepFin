"""
ARIMA + GARCH Trading Strategy with Deep Learning Integration

This package implements a predictive trading strategy on the S&P 500 index using
a hybrid ARIMA + GARCH model combined with a mandatory LSTM/GRU deep learning model.
"""

__version__ = '0.1.0'

import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info(f"ARIMA+GARCH+DL Trading Strategy v{__version__}") 