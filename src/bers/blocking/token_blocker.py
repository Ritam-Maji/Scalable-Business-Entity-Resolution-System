import pandas as pd
from collections import defaultdict
from typing import Set, Tuple, Optional
from bers.blocking.base import Blocker
from bers.constants import COL_RECORD_ID

class TokenBlocker(Blocker):
    """
    Blocks records by building an inverted index on tokens (e.g., words in a name).
    If two records share at least one valid token, they form a candidate pair.
    """
    
    def __init__(self, target_column: str, stop_words: Optional[Set[str]] = None, max_token_freq: int = 1000):
        self.target_column = target_column
        self.stop_words = stop_words or set()
        self.max_token_freq = max_token_freq
        
    def block(self, df: pd.DataFrame) -> Set[Tuple[str, str]]:
        if df.empty or self.target_column not in df.columns:
            return set()
            
        inverted_index = defaultdict(list)
        
        # Build the inverted index
        for _, row in df.dropna(subset=[self.target_column]).iterrows():
            record_id = row[COL_RECORD_ID]
            text = str(row[self.target_column])
            
            # Simple tokenization by whitespace (assumes prior normalization)
            tokens = set(text.split())
            
            for token in tokens:
                if len(token) > 2 and token not in self.stop_words:
                    inverted_index[token].append(record_id)
                    
        pairs = set()
        # Generate pairs from the inverted index
        for token, record_ids in inverted_index.items():
            # Avoid massively common tokens creating explosions (e.g. O(N^2) pairs for a single token)
            if 1 < len(record_ids) <= self.max_token_freq:
                for i in range(len(record_ids)):
                    for j in range(i + 1, len(record_ids)):
                        id1, id2 = sorted([record_ids[i], record_ids[j]])
                        pairs.add((id1, id2))
                        
        return pairs
