import pandas as pd
import logging
from bers.constants import COL_ID_1, COL_ID_2, OUT_COL_RECORD_ID_1, OUT_COL_RECORD_ID_2, OUT_COL_SCORE

def assemble_final_matches(
    scored_pairs: pd.DataFrame, 
    threshold: float,
    s1_ids: list
) -> pd.DataFrame:
    """
    Applies the decision threshold and aggregates matches per S1 anchor.
    Outputs strict format: source1_entity_id, matched_entity_ids
    """
    logging.info(f"Assembling final matches using threshold {threshold:.4f}")
    
    # Filter pairs that meet or exceed the threshold
    matches = scored_pairs[scored_pairs['score'] >= threshold].copy()
    
    if matches.empty:
        return pd.DataFrame({'source1_entity_id': s1_ids, 'matched_entity_ids': [''] * len(s1_ids)})
        
    grouped = matches.groupby(COL_ID_1)[COL_ID_2].apply(lambda x: ','.join(x)).reset_index()
    grouped.columns = ['source1_entity_id', 'matched_entity_ids']
    
    # Left join to ensure all singletons are represented as blank strings
    base_df = pd.DataFrame({'source1_entity_id': s1_ids})
    final_matches = base_df.merge(grouped, on='source1_entity_id', how='left').fillna('')
    
    return final_matches

def resolve_singletons(all_record_ids: list, matched_record_ids: set) -> list:
    """
    Identifies records that did not match with any other record (singletons),
    as required by ML-002/003.
    """
    singletons = [rid for rid in all_record_ids if rid not in matched_record_ids]
    logging.info(f"Resolved {len(singletons)} singleton records out of {len(all_record_ids)} total records.")
    return singletons
