import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
import logging

def create_holdout_split(
    features_df: pd.DataFrame, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the candidate pair feature matrix into a training set and a holdout set
    for local validation, ensuring stratifiction on the label.
    """
    logging.info(f"Splitting data with holdout test_size: {test_size}")
    
    if 'label' not in features_df.columns:
        raise ValueError("Cannot create holdout split: 'label' column is missing from features.")
        
    X = features_df.drop(columns=['label'])
    y = features_df['label']
    
    # Stratified split to maintain class balance (matches vs non-matches)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    logging.info(f"Train size: {len(X_train)} pairs, Holdout size: {len(X_valid)} pairs.")
    return X_train, X_valid, y_train, y_valid
