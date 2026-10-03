import pandas as pd
import numpy as np

class BaselineFuzzyModel:
    """
    A simple unsupervised baseline that averages key string similarity features.
    Used for sanity checking the ML model (ML-001).
    """
    def __init__(self):
        self.feature_cols = ['name_jaccard', 'addr_jaccard', 'name_seq_matcher']
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Unsupervised model, fit does nothing."""
        pass
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns the row-wise mean of available similarity features as a score."""
        cols = [c for c in self.feature_cols if c in X.columns]
        if not cols:
            return np.zeros(len(X))
            
        return X[cols].mean(axis=1).fillna(0.0).values
