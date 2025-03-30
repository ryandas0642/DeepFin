import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Register custom models with torch
def register_torch_classes():
    try:
        # Register models with torch.serialization
        torch.serialization._class_registration_names.update({
            'LSTMModel': 'src.models.deep_learning.LSTMModel',
            'GRUModel': 'src.models.deep_learning.GRUModel'
        })
        logger.info("PyTorch classes registered successfully")
    except Exception as e:
        logger.warning(f"Failed to register PyTorch classes: {str(e)}")

# Register classes at module import
register_torch_classes()

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        """
        LSTM model for time series prediction.
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Number of hidden dimensions
            num_layers (int): Number of LSTM layers
            output_dim (int): Number of output dimensions
            dropout (float): Dropout rate
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Pass through LSTM layers
        out, _ = self.lstm(x, (h0, c0))
        
        # Get output from the last time step
        out = self.fc(out[:, -1, :])
        
        return out

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        """
        GRU model for time series prediction.
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Number of hidden dimensions
            num_layers (int): Number of GRU layers
            output_dim (int): Number of output dimensions
            dropout (float): Dropout rate
        """
        super(GRUModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        
        # GRU layers
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Pass through GRU layers
        out, _ = self.gru(x, h0)
        
        # Get output from the last time step
        out = self.fc(out[:, -1, :])
        
        return out

class DeepLearningModel:
    def __init__(self, model_type='lstm', input_dim=10, hidden_dim=64, num_layers=2, output_dim=1, 
                 learning_rate=0.001, batch_size=32, epochs=100, patience=10):
        """
        Deep learning model for time series prediction.
        
        Args:
            model_type (str): Type of model to use ('lstm' or 'gru')
            input_dim (int): Number of input features
            hidden_dim (int): Number of hidden dimensions
            num_layers (int): Number of recurrent layers
            output_dim (int): Number of output dimensions
            learning_rate (float): Learning rate for optimizer
            batch_size (int): Batch size for training
            epochs (int): Maximum number of epochs
            patience (int): Number of epochs to wait for improvement before early stopping
        """
        self.model_type = model_type.lower()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        
        # Use CPU for better compatibility
        self.device = torch.device('cpu')
        
        # Initialize scalers
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Create model
        self.model = self._create_model()
        
        logger.info(f"Using device: {self.device}")
        
    def _create_model(self):
        """
        Create the specified model.
        
        Returns:
            nn.Module: PyTorch model
        """
        try:
            if self.model_type == 'lstm':
                model = LSTMModel(
                    input_dim=self.input_dim,
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                    output_dim=self.output_dim
                )
            elif self.model_type == 'gru':
                model = GRUModel(
                    input_dim=self.input_dim,
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                    output_dim=self.output_dim
                )
            else:
                logger.error(f"Unknown model type: {self.model_type}")
                raise ValueError(f"Unknown model type: {self.model_type}")
                
            # Move model to device
            model = model.to(self.device)
            logger.info(f"Created {self.model_type.upper()} model with {self.input_dim} input features")
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            # Create a simple fallback model
            logger.info("Creating fallback model")
            fallback = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim)
            ).to(self.device)
            return fallback
    
    def _prepare_data(self, X, y=None):
        """
        Prepare data for training or prediction.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray, optional): Target values
            
        Returns:
            tuple or tensor: Processed data
        """
        try:
            # Reshape X if necessary
            if len(X.shape) == 2:
                # If X is (n_samples, n_features), reshape to (n_samples, 1, n_features)
                X = X.reshape((X.shape[0], 1, X.shape[1]))
                
            # Get dimensions
            n_samples, seq_len, n_features = X.shape
            
            # Ensure input_dim is correct
            if n_features != self.input_dim:
                logger.warning(f"Input dimension mismatch: expected {self.input_dim}, got {n_features}")
                self.input_dim = n_features
                self.model = self._create_model()
            
            # Standardize the input features
            X_reshaped = X.reshape(-1, n_features)
            X_scaled = self.scaler_X.transform(X_reshaped)
            X_scaled = X_scaled.reshape(n_samples, seq_len, n_features)
            
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            if y is not None:
                # Standardize the target
                y_scaled = self.scaler_y.transform(y.reshape(-1, 1)).flatten()
                y_tensor = torch.FloatTensor(y_scaled).to(self.device)
                return X_tensor, y_tensor
            else:
                return X_tensor
                
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
        
    def fit(self, X, y, validation_split=0.2, verbose=1):
        """
        Train the model on the given data.
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, seq_len, n_features)
            y (np.ndarray): Target values of shape (n_samples,)
            validation_split (float): Fraction of data to use for validation
            verbose (int): Verbosity level
            
        Returns:
            self
        """
        try:
            logger.info(f"Training {self.model_type.upper()} model...")
            
            # Update input dimensions if necessary
            if len(X.shape) == 3:
                self.input_dim = X.shape[2]
                self.model = self._create_model()
            
            # Scale the data
            self.scaler_X.fit(X.reshape(-1, X.shape[-1]))
            self.scaler_y.fit(y.reshape(-1, 1))
            
            # Prepare data
            X_tensor, y_tensor = self._prepare_data(X, y)
            
            # Split into train and validation sets
            if validation_split > 0:
                split_idx = int(len(X) * (1 - validation_split))
                X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
                y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
            else:
                X_train, y_train = X_tensor, y_tensor
                X_val, y_val = None, None
            
            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=min(self.batch_size, len(train_dataset)),
                shuffle=True
            )
            
            if X_val is not None:
                val_dataset = TensorDataset(X_val, y_val)
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=min(self.batch_size, len(val_dataset)),
                    shuffle=False
                )
            
            # Define loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Early stopping variables
            best_val_loss = float('inf')
            best_model_dict = None
            patience_counter = 0
            
            # Training loop
            train_losses = []
            val_losses = []
            
            for epoch in range(self.epochs):
                self.model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    # Forward pass
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y.unsqueeze(1))
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * batch_X.size(0)
                    
                train_loss /= len(train_loader.dataset)
                train_losses.append(train_loss)
                
                # Validation
                if X_val is not None:
                    self.model.eval()
                    val_loss = 0.0
                    
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            outputs = self.model(batch_X)
                            loss = criterion(outputs, batch_y.unsqueeze(1))
                            val_loss += loss.item() * batch_X.size(0)
                            
                    val_loss /= len(val_loader.dataset)
                    val_losses.append(val_loss)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_dict = self.model.state_dict().copy()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
                        
                    if verbose >= 1 and (epoch + 1) % (max(1, self.epochs // 10)) == 0:
                        logger.info(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
                else:
                    if verbose >= 1 and (epoch + 1) % (max(1, self.epochs // 10)) == 0:
                        logger.info(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.6f}")
            
            # Load the best model if early stopping was used
            if best_model_dict is not None:
                self.model.load_state_dict(best_model_dict)
                
            logger.info("Training completed")
            return self
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray or float: Predicted values
        """
        if not hasattr(self, 'model') or self.model is None:
            logger.error("Model not trained yet")
            return 0.0
            
        try:
            self.model.eval()
            
            # Prepare data
            X_tensor = self._prepare_data(X, y=None)
            
            # Make predictions
            with torch.no_grad():
                predictions = self.model(X_tensor)
                
            # Convert to numpy and inverse transform
            predictions_np = predictions.cpu().numpy()
            
            # Flatten the predictions array
            if predictions_np.ndim > 1:
                predictions_np = predictions_np.flatten()
                
            # Apply inverse transform if needed
            if hasattr(self, 'scaler_y') and self.scaler_y is not None:
                try:
                    # Reshape for inverse transform if needed
                    if predictions_np.ndim == 1:
                        predictions_reshaped = predictions_np.reshape(-1, 1)
                    else:
                        predictions_reshaped = predictions_np
                        
                    # Apply inverse transform
                    predictions_np = self.scaler_y.inverse_transform(predictions_reshaped).flatten()
                except Exception as e:
                    logger.warning(f"Error in inverse transform: {str(e)}. Using raw predictions.")
            
            # Ensure we return a 1D array or scalar
            if predictions_np.size == 1:
                # If there's only one prediction, return it as a scalar float
                return float(predictions_np.item())
            else:
                # Otherwise return the flattened array
                return predictions_np.flatten()
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            # Return a default prediction to prevent app crash
            return 0.0
    
    def evaluate(self, X, y):
        """
        Evaluate the model on test data.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            
        Returns:
            dict: Evaluation metrics
        """
        if not hasattr(self, 'model') or self.model is None:
            logger.error("Model not trained yet")
            return {
                'mse': 999.9,
                'rmse': 999.9,
                'mae': 999.9,
                'direction_accuracy': 0.0
            }
            
        try:
            # Make predictions
            y_pred = self.predict(X)
            
            # Ensure y_pred is an array
            if not isinstance(y_pred, np.ndarray):
                y_pred = np.array([y_pred])
                
            # Ensure y is an array
            if not isinstance(y, np.ndarray):
                y = np.array([y])
                
            # Ensure dimensions match
            if len(y_pred) != len(y):
                logger.warning(f"Prediction length ({len(y_pred)}) doesn't match target length ({len(y)})")
                # Truncate to the smaller size
                min_len = min(len(y_pred), len(y))
                y_pred = y_pred[:min_len]
                y = y[:min_len]
            
            # Calculate metrics
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            
            # Calculate direction accuracy
            actual_direction = np.sign(np.diff(np.append([0], y)))
            predicted_direction = np.sign(np.diff(np.append([0], y_pred)))
            
            # Ensure lengths match
            min_dir_len = min(len(actual_direction), len(predicted_direction))
            direction_accuracy = np.mean(actual_direction[:min_dir_len] == predicted_direction[:min_dir_len])
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'direction_accuracy': direction_accuracy
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {
                'mse': 999.9,
                'rmse': 999.9,
                'mae': 999.9,
                'direction_accuracy': 0.0
            }
    
    def save(self, path):
        """
        Save the model to disk.
        
        Args:
            path (str): Path to save the model
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
            # Save the model parameters and hyperparameters
            state = {
                'model_state_dict': self.model.state_dict(),
                'model_type': self.model_type,
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'output_dim': self.output_dim,
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y
            }
            torch.save(state, path)
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load(self, path):
        """
        Load a model from disk.
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            self
        """
        try:
            # Register the classes again to ensure they are available
            register_torch_classes()
            
            # Load the saved state
            state = torch.load(path, map_location=self.device)
            
            # Update model parameters
            self.model_type = state['model_type']
            self.input_dim = state['input_dim']
            self.hidden_dim = state['hidden_dim']
            self.num_layers = state['num_layers']
            self.output_dim = state['output_dim']
            
            # Recreate the model
            self.model = self._create_model()
            
            # Load the state dict
            self.model.load_state_dict(state['model_state_dict'])
            
            # Load the scalers
            self.scaler_X = state['scaler_X']
            self.scaler_y = state['scaler_y']
            
            logger.info(f"Model loaded from {path}")
            return self
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error("Creating a new model instead")
            self.model = self._create_model()
            return self 