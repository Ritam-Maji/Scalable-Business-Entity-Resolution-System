import pandas as pd
from typing import Dict

def extract_agreement_features(
    city1: str, city2: str,
    state1: str, state2: str,
    country1: str, country2: str,
    zip1: str, zip2: str
) -> Dict[str, float]:
    """
    Computes boolean/categorical agreement features for simple fields.
    Uses -1.0 to indicate that one or both fields are missing, 
    allowing tree-based models (like LightGBM) to handle missingness directly.
    """
    
    def exact_match_score(val1: str, val2: str) -> float:
        if pd.isna(val1) or pd.isna(val2) or not val1 or not val2:
            return -1.0  # Missing data indicator
        return 1.0 if val1 == val2 else 0.0
        
    return {
        'city_match': exact_match_score(city1, city2),
        'state_match': exact_match_score(state1, state2),
        'country_match': exact_match_score(country1, country2),
        'zip_match': exact_match_score(zip1, zip2)
    }
