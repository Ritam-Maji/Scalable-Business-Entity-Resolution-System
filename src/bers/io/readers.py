import pandas as pd
from pathlib import Path
import logging
from bers.io.schema import validate_dataframe
from bers.constants import COL_RECORD_ID

def read_source_file(file_path: Path, source_prefix: str) -> pd.DataFrame:
    """
    Reads a source TSV file and prefixes its record IDs.
    Prefixing ensures global uniqueness across S1, S2, S3.
    """
    logging.info(f"Reading source file: {file_path}")
    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found: {file_path}")
        
    # Use PyArrow string backend for massive memory footprint reduction
    df = pd.read_csv(file_path, sep='\t', dtype='string[pyarrow]', na_filter=False)
    
    # Map dataset-specific columns to our standard internal schema
    rename_map = {
        'entity_id': COL_RECORD_ID,
        'business_name': 'name',
        'business_address': 'address'
    }
    df = df.rename(columns=rename_map)
    
    # Prefixing is not needed, dataset already has S1-, S2-, S3-
    # if COL_RECORD_ID in df.columns:
    #     df[COL_RECORD_ID] = source_prefix + df[COL_RECORD_ID].astype(str)
        
    return df

def read_ground_truth(file_path: Path) -> pd.DataFrame:
    """Reads the ground truth TSV file."""
    logging.info(f"Reading ground truth file: {file_path}")
    if not file_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")
    
    return pd.read_csv(file_path, sep='\t')
