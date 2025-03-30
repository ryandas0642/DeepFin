import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime, timedelta
import yfinance as yf
from typing import Tuple, Dict, Optional, Union
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, vol: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.vol = torch.FloatTensor(vol)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.vol[idx]

class AdvancedRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        model_type: str = 'lstm',
        quant_weight: float = 0.5,
        sentiment_weight: float = 0.5
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.model_type = model_type
        self.quant_weight = quant_weight
        self.sentiment_weight = sentiment_weight
        
        # Main RNN layer
        if model_type.lower() == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:  # GRU
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Volatility prediction head
        self.vol_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Price prediction head
        self.price_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Options recommendation head
        self.options_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3)  # [action_prob, strike_prob, expiry_prob]
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # RNN forward pass
        rnn_out, _ = self.rnn(x)
        
        # Apply attention
        attention_weights = self.attention(rnn_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended = torch.sum(attention_weights * rnn_out, dim=1)
        
        # Get predictions
        vol_pred = self.vol_head(attended)
        price_pred = self.price_head(attended)
        options_pred = self.options_head(attended)
        
        return price_pred, vol_pred, options_pred

class AdvancedDeepLearningModel:
    def __init__(
        self,
        window_size: int = 60,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        model_type: str = 'lstm',
        quant_weight: float = 0.5,
        sentiment_weight: float = 0.5,
        learning_rate: float = 0.001
    ):
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.model_type = model_type
        self.quant_weight = quant_weight
        self.sentiment_weight = sentiment_weight
        self.learning_rate = learning_rate
        
        # Initialize scalers
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.scaler_vol = StandardScaler()
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def _prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features for the model."""
        # Quantitative features
        quant_features = [
            'log_return', 'macd', 'macd_signal', 'macd_diff',
            'rsi', 'bollinger_width', 'volatility',
            'momentum_5', 'momentum_10', 'momentum_20'
        ]
        
        # Sentiment features (example - would need to be implemented)
        sentiment_features = [
            'sentiment_score', 'news_volume', 'social_volume'
        ]
        
        # Combine features with weights
        X_quant = data[quant_features].values
        X_sent = data[sentiment_features].values if all(f in data.columns for f in sentiment_features) else np.zeros((len(data), len(sentiment_features)))
        
        X = self.quant_weight * X_quant + self.sentiment_weight * X_sent
        
        # Prepare target variables
        y = data['log_return'].values
        vol = data['volatility'].values
        
        return X, y, vol
        
    def fit(self, data: pd.DataFrame, epochs: int = 100, batch_size: int = 32):
        """Fit the model to the data."""
        try:
            # Prepare features
            X, y, vol = self._prepare_features(data)
            
            # Scale features
            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
            vol_scaled = self.scaler_vol.fit_transform(vol.reshape(-1, 1))
            
            # Create sequences
            X_seq = []
            y_seq = []
            vol_seq = []
            
            for i in range(len(data) - self.window_size):
                X_seq.append(X_scaled[i:i+self.window_size])
                y_seq.append(y_scaled[i+self.window_size])
                vol_seq.append(vol_scaled[i+self.window_size])
                
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            vol_seq = np.array(vol_seq)
            
            # Create dataset and dataloader
            dataset = TimeSeriesDataset(X_seq, y_seq, vol_seq)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Initialize model
            self.model = AdvancedRNN(
                input_size=X_seq.shape[2],
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                model_type=self.model_type,
                quant_weight=self.quant_weight,
                sentiment_weight=self.sentiment_weight
            ).to(self.device)
            
            # Define loss functions
            mse_loss = nn.MSELoss()
            bce_loss = nn.BCEWithLogitsLoss()
            
            # Define optimizer
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Training loop
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch_X, batch_y, batch_vol in dataloader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    batch_vol = batch_vol.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    price_pred, vol_pred, options_pred = self.model(batch_X)
                    
                    # Calculate losses
                    price_loss = mse_loss(price_pred, batch_y)
                    vol_loss = mse_loss(vol_pred, batch_vol)
                    options_loss = bce_loss(options_pred, torch.zeros_like(options_pred))  # Placeholder
                    
                    # Combined loss
                    loss = price_loss + vol_loss + options_loss
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
                    
            logger.info("Model training completed")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            raise
            
    def predict(self, data: pd.DataFrame) -> Dict[str, Union[float, str, datetime]]:
        """Make predictions and generate options trading recommendations."""
        try:
            if self.model is None:
                raise ValueError("Model must be fit before making predictions")
                
            # Prepare features
            X, _, _ = self._prepare_features(data)
            
            # Scale features
            X_scaled = self.scaler_X.transform(X)
            
            # Get last window
            X_seq = X_scaled[-self.window_size:].reshape(1, self.window_size, -1)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_seq).to(self.device)
                price_pred, vol_pred, options_pred = self.model(X_tensor)
                
                # Unscale predictions
                price_pred = self.scaler_y.inverse_transform(price_pred.cpu().numpy())[0][0]
                vol_pred = self.scaler_vol.inverse_transform(vol_pred.cpu().numpy())[0][0]
                
                # Process options prediction
                options_probs = torch.softmax(torch.FloatTensor(options_pred.cpu().numpy()[0]), dim=0)
                action_idx = torch.argmax(options_probs[:2]).item()  # 0 for call, 1 for put
                action = "BUY CALL" if action_idx == 0 else "BUY PUT"
                
                # Calculate strike price (simplified)
                current_price = data['Close'].iloc[-1]
                strike_offset = options_probs[1].item() * 10  # 0-10% offset
                strike_price = current_price * (1 + strike_offset if action_idx == 0 else 1 - strike_offset)
                
                # Calculate expiry date (1 month from now)
                expiry_date = datetime.now() + timedelta(days=30)
                
                # Calculate confidence score
                confidence = options_probs[action_idx].item()
                
                return {
                    'action': action,
                    'strike_price': round(strike_price, 2),
                    'expiry_date': expiry_date.strftime('%Y-%m-%d'),
                    'predicted_return': round(price_pred, 4),
                    'predicted_volatility': round(vol_pred, 4),
                    'confidence': round(confidence, 4)
                }
                
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise 