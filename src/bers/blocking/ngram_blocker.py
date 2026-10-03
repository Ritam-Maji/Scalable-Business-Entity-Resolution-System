import pandas as pd
from typing import Set
from collections import defaultdict
from bers.blocking.base import RetrievalBlocker
from bers.constants import COL_RECORD_ID

class NGramBlocker(RetrievalBlocker):
    """
    Builds an inverted index on character n-grams (e.g. 3-grams) of the name.
    Useful for catching spelling errors or slight variations that evade exact tokens.
    """
    def __init__(self, n: int = 3, max_ngram_freq: int = 200, min_shared_ngrams: int = 2):
        self.n = n
        self.max_ngram_freq = max_ngram_freq
        self.min_shared_ngrams = min_shared_ngrams
        self.index = defaultdict(set)
        
    def _get_ngrams(self, text: str) -> Set[str]:
        text = text.replace(" ", "")
        if len(text) < self.n:
            return set()
        return set(text[i:i+self.n] for i in range(len(text) - self.n + 1))
        
    def build_index(self, df_target: pd.DataFrame) -> None:
        self.index.clear()
        if 'name' not in df_target.columns:
            return
            
        for _, row in df_target.dropna(subset=['name']).iterrows():
            name = str(row['name']).strip()
            if name:
                ngrams = self._get_ngrams(name)
                rec_id = str(row[COL_RECORD_ID])
                for ng in ngrams:
                    self.index[ng].add(rec_id)
                    
    def query(self, anchor_record: pd.Series) -> Set[str]:
        name = str(anchor_record.get('name', '')).strip()
        if not name:
            return set()
            
        ngrams = self._get_ngrams(name)
        
        # Count how many ngrams are shared per candidate
        candidate_counts = defaultdict(int)
        
        for ng in ngrams:
            posting_list = self.index.get(ng, set())
            if len(posting_list) <= self.max_ngram_freq:
                for cand_id in posting_list:
                    candidate_counts[cand_id] += 1
                    
        # Only return candidates that share at least min_shared_ngrams
        candidates = {
            cand_id for cand_id, count in candidate_counts.items() 
            if count >= self.min_shared_ngrams
        }
        return candidates
