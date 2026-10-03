import pandas as pd
import numpy as np
import logging
from typing import Tuple

def f_beta_score(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.5) -> float:
    """Calculates the F-beta score."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision == 0 and recall == 0:
        return 0.0
        
    beta_sq = beta ** 2
    return (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)

def find_optimal_threshold(
    y_true: pd.Series, 
    y_scores: pd.Series, 
    beta: float = 0.5,
    num_steps: int = 100
) -> Tuple[float, float]:
    """
    Sweeps through possible probability thresholds to find the one that
    maximizes the F-beta score (default F0.5 per EVAL-001).
    """
    logging.info(f"Searching for optimal threshold (F{beta})...")
    
    thresholds = np.linspace(0.01, 0.99, num_steps)
    best_threshold = 0.5
    best_score = -1.0
    
    y_true_np = y_true.values
    y_scores_np = y_scores.values
    
    for t in thresholds:
        y_pred = (y_scores_np >= t).astype(int)
        score = f_beta_score(y_true_np, y_pred, beta)
        
        if score > best_score:
            best_score = score
            best_threshold = t
            
    logging.info(f"Optimal threshold found: {best_threshold:.4f} (Score: {best_score:.4f})")
    return best_threshold, best_score
