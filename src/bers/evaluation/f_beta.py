import pandas as pd
import logging
from typing import Set, Tuple, Dict
from bers.constants import OUT_COL_RECORD_ID_1, OUT_COL_RECORD_ID_2

def compute_competition_metrics(
    predicted_matches: pd.DataFrame, 
    ground_truth: pd.DataFrame, 
    beta: float = 0.5
) -> Dict[str, float]:
    """
    Computes global precision, recall, and F-beta (F0.5) 
    by treating the evaluation as a set-matching problem over the output pairs.
    """
    logging.info("Computing final evaluation metrics...")
    
    # Convert DataFrames to sets of sorted tuples for exact undirected comparison
    pred_set: Set[Tuple[str, str]] = set()
    if not predicted_matches.empty:
        for _, row in predicted_matches.iterrows():
            id1, id2 = sorted([str(row[OUT_COL_RECORD_ID_1]), str(row[OUT_COL_RECORD_ID_2])])
            pred_set.add((id1, id2))
            
    gt_set: Set[Tuple[str, str]] = set()
    if not ground_truth.empty:
        for _, row in ground_truth.iterrows():
            # Ground truth is typically just two columns of IDs
            id1, id2 = sorted([str(row.iloc[0]), str(row.iloc[1])])
            gt_set.add((id1, id2))
            
    tp = len(pred_set.intersection(gt_set))
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    f_beta = 0.0
    if precision > 0 or recall > 0:
        beta_sq = beta ** 2
        f_beta = (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)
        
    metrics = {
        'precision': precision,
        'recall': recall,
        f'f{beta}': f_beta,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }
    
    logging.info(f"Evaluation Results: Precision={precision:.4f}, Recall={recall:.4f}, F{beta}={f_beta:.4f}")
    return metrics
