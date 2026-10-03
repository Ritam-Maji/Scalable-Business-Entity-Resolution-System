import pandas as pd
from pathlib import Path
import logging
from bers.constants import OUT_COL_RECORD_ID_1, OUT_COL_RECORD_ID_2, OUT_COL_SCORE, MATCHING_RESULTS_FILENAME

def validate_submission_files(output_dir: Path) -> bool:
    """
    Validates that the generated TSV files conform exactly to the competition schema.
    """
    matches_file = output_dir / MATCHING_RESULTS_FILENAME
    
    if not matches_file.exists():
        logging.error(f"Missing required file: {matches_file.name}")
        return False
        
    try:
        df = pd.read_csv(matches_file, sep='\t')
        
        expected_cols = [OUT_COL_RECORD_ID_1, OUT_COL_RECORD_ID_2, OUT_COL_SCORE]
        if list(df.columns) != expected_cols:
            logging.error(f"Schema mismatch in {matches_file.name}. Expected {expected_cols}, got {list(df.columns)}")
            return False
            
        if df.isnull().any().any():
            logging.error(f"NaN values found in {matches_file.name}. This is not allowed.")
            return False
            
        logging.info(f"Submission file {matches_file.name} successfully passed validation checks.")
        return True
        
    except Exception as e:
        logging.error(f"Validation failed with error: {e}")
        return False
