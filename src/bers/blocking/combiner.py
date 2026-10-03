import pandas as pd
from typing import List, Generator, Tuple
import logging
from bers.blocking.base import RetrievalBlocker
from bers.constants import COL_RECORD_ID

class MultiStageRetrieval:
    """
    Combines multiple RetrievalBlockers.
    Given df_target (S2+S3), it builds indices.
    Given df_anchor (S1), it retrieves candidates for each anchor
    by unioning the results of all blockers, yielding them chunk by chunk.
    """
    def __init__(self, blockers: List[RetrievalBlocker]):
        self.blockers = blockers
        
    def index_targets(self, df_target: pd.DataFrame) -> None:
        """Builds all indices on the target dataframe."""
        for i, blocker in enumerate(self.blockers):
            logging.info(f"Building index for blocker {i+1}/{len(self.blockers)}: {blocker.__class__.__name__}")
            blocker.build_index(df_target)
            
    def retrieve_candidates(self, df_anchor: pd.DataFrame) -> Generator[Tuple[str, str], None, None]:
        """
        Iterates through the anchor dataframe and yields (anchor_id, target_id) pairs.
        """
        for idx, row in df_anchor.iterrows():
            anchor_id = str(row[COL_RECORD_ID])
            
            # Union candidates from all blockers
            final_candidates = set()
            for blocker in self.blockers:
                final_candidates.update(blocker.query(row))
                
            # Yield pairs
            for target_id in final_candidates:
                # We sort them to maintain consistent (id1, id2) ordering for feature building
                id1, id2 = sorted([anchor_id, target_id])
                yield (id1, id2)
