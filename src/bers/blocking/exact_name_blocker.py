import pandas as pd
from typing import Set
from collections import defaultdict
from bers.blocking.base import RetrievalBlocker
from bers.constants import COL_RECORD_ID

class ExactNameBlocker(RetrievalBlocker):
    """
    Retrieves targets that have the exact same normalized name as the anchor.
    Highly precise, low recall.
    """
    def __init__(self):
        self.index = defaultdict(set)
        
    def build_index(self, df_target: pd.DataFrame) -> None:
        self.index.clear()
        if 'name' not in df_target.columns:
            return
            
        name_idx = df_target.columns.get_loc('name') + 1
        id_idx = df_target.columns.get_loc(COL_RECORD_ID) + 1
        
        for row in df_target.itertuples():
            record_id = str(row[id_idx])
            if record_id.startswith('S1-'):
                continue
                
            name_val = row[name_idx]
            if pd.isna(name_val):
                continue
                
            name = str(name_val).strip()
            if name:
                self.index[name].add(record_id)
                
    def query(self, anchor_record: pd.Series) -> Set[str]:
        name = str(anchor_record.get('name', '')).strip()
        if not name:
            return set()
        return self.index.get(name, set())
