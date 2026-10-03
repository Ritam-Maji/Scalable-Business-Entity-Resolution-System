import pandas as pd
import numpy as np
from bers.evaluation.f_beta import compute_competition_metrics
from bers.decision.threshold_search import f_beta_score
from bers.constants import OUT_COL_RECORD_ID_1, OUT_COL_RECORD_ID_2

def test_f_beta_score_math():
    # Perfect match
    assert f_beta_score(np.array([1, 1, 0]), np.array([1, 1, 0]), beta=0.5) == 1.0
    # No matches
    assert f_beta_score(np.array([1, 1]), np.array([0, 0]), beta=0.5) == 0.0

def test_compute_competition_metrics():
    # Predicted matches
    pred = pd.DataFrame({
        OUT_COL_RECORD_ID_1: ['1', '3'],
        OUT_COL_RECORD_ID_2: ['2', '4']
    })
    
    # Ground truth (notice ordering is swapped for second pair to test undirected edge logic)
    gt = pd.DataFrame({
        'id1': ['1', '4'], 
        'id2': ['2', '3']
    })
    
    metrics = compute_competition_metrics(pred, gt)
    
    assert metrics['true_positives'] == 2
    assert metrics['false_positives'] == 0
    assert metrics['false_negatives'] == 0
    assert metrics['f0.5'] == 1.0
