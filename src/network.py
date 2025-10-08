# Network.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

class NeuralNetworkClassifier(nn.Module):
    """
    PyTorch Neural Network for binary classification
    """
    def __init__(self, input_size):
        super(NeuralNetworkClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 16)
        self.layer4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        x = self.sigmoid(self.layer4(x))
        return x


class PyTorchNNWrapper(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for PyTorch Neural Network
    """
    def __init__(self, input_size, epochs=100, lr=0.001, batch_size=32):
        self.input_size = input_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def fit(self, X, y):
        """Train the neural network"""
        # Initialize model
        self.model = NeuralNetworkClassifier(self.input_size).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_tensor, y_tensor, test_size=0.15, random_state=42
        )      
        # Training loop
        self.model.train()
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Mini-batch training
            indices = torch.randperm(len(X_train))
            epoch_loss = 0
            
            for i in range(0, len(X_train), self.batch_size):
                batch_idx = indices[i:i+self.batch_size]
                batch_X = X_train[batch_idx]
                batch_y = y_train[batch_idx]
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Validation loss (for early stopping)
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
            self.model.train()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            predictions = (outputs.cpu().numpy() > 0.5).astype(int).flatten()
        return predictions
    
    def predict_proba(self, X):
        """Return probability estimates (for sklearn compatibility)"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor).cpu().numpy()
        # Return probabilities for both classes
        return np.hstack([1 - outputs, outputs])


def get_neural_network(input_size):
    """
    Get configured PyTorch neural network classifier
    """
    return PyTorchNNWrapper(
        input_size=input_size,
        epochs=250,
        lr=0.001,
        batch_size=16
    )