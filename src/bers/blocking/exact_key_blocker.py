import pandas as pd
from typing import Set, Tuple, List
from bers.blocking.base import Blocker
from bers.constants import COL_RECORD_ID

class ExactKeyBlocker(Blocker):
    """
    Blocks records that have exact matches on a specified set of columns.
    For example, blocking by ['country', 'city'] means only businesses in the same
    city and country will form candidate pairs.
    """
    
    def __init__(self, blocking_keys: List[str]):
        self.blocking_keys = blocking_keys
        
    def block(self, df: pd.DataFrame) -> Set[Tuple[str, str]]:
        if not self.blocking_keys or df.empty:
            return set()
            
        pairs = set()
        
        # Drop rows where any of the blocking keys are missing to avoid false groupings
        valid_df = df.dropna(subset=self.blocking_keys).copy()
        
        # Group by the specified blocking keys
        groups = valid_df.groupby(self.blocking_keys)
        
        for _, group in groups:
            # We need at least 2 records to form a pair
            if len(group) > 1:
                ids = group[COL_RECORD_ID].tolist()
                # Generate all unique combinations of length 2
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        # Sort IDs to ensure undirected edge consistency (id_1 < id_2)
                        id1, id2 = sorted([ids[i], ids[j]])
                        pairs.add((id1, id2))
                        
        return pairs
