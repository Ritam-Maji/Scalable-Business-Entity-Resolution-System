from typing import Set, Tuple, Dict
import logging

def compute_blocking_metrics(
    candidate_pairs: Set[Tuple[str, str]], 
    ground_truth_pairs: Set[Tuple[str, str]],
    total_records: int
) -> Dict[str, float]:
    """
    Computes standard blocking metrics (BLK-005, BLK-006):
    - Pair Completeness (Recall): Proportion of true matches that survived blocking.
    - Reduction Ratio (RR): How much the search space was reduced compared to Cartesian product.
    """
    if not ground_truth_pairs:
        logging.warning("Ground truth is empty, cannot compute Pair Completeness.")
        return {}
        
    # True positives: candidate pairs that are in ground truth
    # Note: Ensure both sets use sorted tuples (id1 < id2) to match properly
    true_positives = candidate_pairs.intersection(ground_truth_pairs)
    
    pair_completeness = len(true_positives) / len(ground_truth_pairs)
    
    # Total possible pairs in a Cartesian product (n * (n-1) / 2)
    total_possible_pairs = (total_records * (total_records - 1)) / 2
    
    reduction_ratio = 1.0 - (len(candidate_pairs) / total_possible_pairs) if total_possible_pairs > 0 else 0.0
    
    return {
        "pair_completeness": pair_completeness,
        "reduction_ratio": reduction_ratio,
        "candidates_count": float(len(candidate_pairs)),
        "true_matches_captured": float(len(true_positives))
    }
