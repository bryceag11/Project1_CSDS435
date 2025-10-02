# preprocessing.py
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(X, y):
    """
    Handle nominal (string) and continuous (numeric) features
    """
    # Identify column types
    numeric_features = []
    nominal_features = []
    
    for col in X.columns:
        if X[col].dtype == 'object':  # String/nominal
            nominal_features.append(col)
        else:
            numeric_features.append(col)
    
    # Encode nominal features
    label_encoders = {}
    X_processed = X.copy()
    
    for col in nominal_features:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Convert all to numeric
    X_processed = X_processed.astype(float)
    
    # Normalize continuous features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)
    
    return X_scaled, y, scaler, label_encoders