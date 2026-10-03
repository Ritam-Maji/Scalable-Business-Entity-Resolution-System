import pandas as pd
import difflib
from typing import Dict

def extract_address_features(addr1: str, addr2: str) -> Dict[str, float]:
    """Computes similarity features between two normalized addresses."""
    if pd.isna(addr1) or pd.isna(addr2) or not addr1 or not addr2:
        return {'addr_exact_match': 0.0, 'addr_jaccard': 0.0, 'addr_seq_matcher': 0.0}
        
    exact = 1.0 if addr1 == addr2 else 0.0
    
    tokens1 = set(addr1.split())
    tokens2 = set(addr2.split())
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    jaccard = intersection / union if union > 0 else 0.0
    
    seq = difflib.SequenceMatcher(None, addr1, addr2).ratio()
    
    return {'addr_exact_match': exact, 'addr_jaccard': jaccard, 'addr_seq_matcher': seq}
