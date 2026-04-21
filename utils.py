"""Utility functions for loading and preparing wine quality data."""
import pandas as pd
 
 
def load_data(path):
    """Load CSV and ensure 'quality' column exists."""
    df = pd.read_csv(path)
    if 'quality' not in df.columns:
        raise ValueError("CSV must contain a 'quality' column.")
    return df
 
 
def features_and_target(df, target='quality'):
    """Return X (features) and y (target)."""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y