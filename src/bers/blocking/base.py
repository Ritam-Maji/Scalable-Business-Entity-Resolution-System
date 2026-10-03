from abc import ABC, abstractmethod
import pandas as pd
from typing import Set

class RetrievalBlocker(ABC):
    """
    Abstract base class for anchor-based retrieval blockers.
    Instead of generating all pairs, it builds an index on the target
    dataset (S2+S3) and allows querying via anchors (S1).
    """
    
    @abstractmethod
    def build_index(self, df_target: pd.DataFrame) -> None:
        """
        Builds the internal inverted index using the target records (S2 + S3).
        """
        pass
        
    @abstractmethod
    def query(self, anchor_record: pd.Series) -> Set[str]:
        """
        Given a single anchor record (from S1), returns a set of target record IDs
        that match the blocking criteria.
        """
        pass
