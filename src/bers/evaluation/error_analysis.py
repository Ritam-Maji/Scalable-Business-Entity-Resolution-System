import pandas as pd
from typing import Tuple
from pathlib import Path
from bers.constants import COL_RECORD_ID, OUT_COL_RECORD_ID_1, OUT_COL_RECORD_ID_2

def generate_error_reports(
    predicted_matches: pd.DataFrame,
    ground_truth: pd.DataFrame,
    original_data: pd.DataFrame,
    output_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identifies False Merges (FP) and Missed Matches (FN) and joins 
    original entity attributes for human review (satisfies Section 12.3 audit logic).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pred_pairs = set()
    if not predicted_matches.empty:
        for _, row in predicted_matches.iterrows():
            id1, id2 = sorted([str(row[OUT_COL_RECORD_ID_1]), str(row[OUT_COL_RECORD_ID_2])])
            pred_pairs.add((id1, id2))
            
    gt_pairs = set()
    if not ground_truth.empty:
        for _, row in ground_truth.iterrows():
            id1, id2 = sorted([str(row.iloc[0]), str(row.iloc[1])])
            gt_pairs.add((id1, id2))
            
    false_positives = pred_pairs - gt_pairs
    false_negatives = gt_pairs - pred_pairs
    
    def build_report_df(pair_set) -> pd.DataFrame:
        if not pair_set:
            return pd.DataFrame()
            
        rows = []
        df_indexed = original_data.set_index(COL_RECORD_ID)
        
        for id1, id2 in pair_set:
            row = {'id_1': id1, 'id_2': id2}
            
            if id1 in df_indexed.index:
                r1 = df_indexed.loc[id1]
                row['name_1'] = r1.get('name', '')
                row['country_1'] = r1.get('country', '')
            
            if id2 in df_indexed.index:
                r2 = df_indexed.loc[id2]
                row['name_2'] = r2.get('name', '')
                row['country_2'] = r2.get('country', '')
                
            rows.append(row)
        return pd.DataFrame(rows)
        
    fp_df = build_report_df(false_positives)
    fn_df = build_report_df(false_negatives)
    
    if not fp_df.empty:
        fp_df.to_csv(output_dir / "false_merges.csv", index=False)
    if not fn_df.empty:
        fn_df.to_csv(output_dir / "missed_matches.csv", index=False)
        
    return fp_df, fn_df
