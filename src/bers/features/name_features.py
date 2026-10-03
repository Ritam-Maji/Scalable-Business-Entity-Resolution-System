import pandas as pd
import difflib
from typing import Dict

def extract_name_features(name1: str, name2: str) -> Dict[str, float]:
    """Computes similarity features between two normalized business names."""
    if pd.isna(name1) or pd.isna(name2) or not name1 or not name2:
        return {'name_exact_match': 0.0, 'name_jaccard': 0.0, 'name_seq_matcher': 0.0}
        
    # Exact match
    exact = 1.0 if name1 == name2 else 0.0
    
    # Token Jaccard Similarity
    tokens1 = set(name1.split())
    tokens2 = set(name2.split())
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    jaccard = intersection / union if union > 0 else 0.0
    
    # Sequence Matcher (difflib ratio gives a fast character-level similarity)
    seq = difflib.SequenceMatcher(None, name1, name2).ratio()
    
    return {'name_exact_match': exact, 'name_jaccard': jaccard, 'name_seq_matcher': seq}
