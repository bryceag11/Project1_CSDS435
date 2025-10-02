# data_loader.py
import numpy as np
import pandas as pd

def load_data(filepath):
    """
    Load each text file, with columns being features and last being a label
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            # Split by whitespace
            values = line.strip().split()
            data.append(values)
    
    df = pd.DataFrame(data)
    
    # Separate features and labels
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].astype(int)
    
    return X, y