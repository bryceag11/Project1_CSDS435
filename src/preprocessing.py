from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd

def preprocess_data(X, y):
    """
    Preprocess data with mixed types (numeric + categorical).
    """
    X_processed = X.copy()
    
    # Find categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        
        for col in categorical_cols:
            
            # Label encode:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X[col])
    
    # Convert all to float
    X_numeric = X_processed.astype(float)
    
    # Normalize using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    
    # Convert labels to int array
    y = y.astype(int).values if hasattr(y, 'values') else y.astype(int)
    
    # print(f"\nPreprocessed: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    # print(f"Features normalized: mean={X_scaled.mean():.6f}, std={X_scaled.std():.6f}")
    # print(f"Class distribution: {np.bincount(y)}")
    
    return X_scaled, y