import pandas as pd
from typing import Set
from collections import defaultdict
from bers.blocking.base import RetrievalBlocker
from bers.constants import COL_RECORD_ID

class RareTokenBlocker(RetrievalBlocker):
    """
    Builds an inverted index on tokens in a specified column.
    Only queries tokens that are 'rare' (frequency <= max_token_freq) 
    to prevent combinatorial explosion while retaining informative signals.
    """
    def __init__(self, target_column: str = 'name', max_token_freq: int = 100):
        self.target_column = target_column
        self.max_token_freq = max_token_freq
        self.index = defaultdict(set)
        
    def build_index(self, df_target: pd.DataFrame) -> None:
        self.index.clear()
        if self.target_column not in df_target.columns:
            return
            
        col_idx = df_target.columns.get_loc(self.target_column) + 1
        id_idx = df_target.columns.get_loc(COL_RECORD_ID) + 1
        
        for row in df_target.itertuples():
            record_id = str(row[id_idx])
            if record_id.startswith('S1-'):
                continue
                
            val_raw = row[col_idx]
            if pd.isna(val_raw):
                continue
                
            val = str(val_raw).strip()
            if val:
                tokens = set(val.split())
                for token in tokens:
                    if len(token) > 2: # Ignore 1-2 char noise words
                        self.index[token].add(record_id)
                        
    def query(self, anchor_record: pd.Series) -> Set[str]:
        val = str(anchor_record.get(self.target_column, '')).strip()
        if not val:
            return set()
            
        tokens = set(val.split())
        candidates = set()
        
        for token in tokens:
            if len(token) > 2:
                posting_list = self.index.get(token, set())
                # Only use the token if it's informative (rare enough)
                if len(posting_list) <= self.max_token_freq:
                    candidates.update(posting_list)
                    
        return candidates
