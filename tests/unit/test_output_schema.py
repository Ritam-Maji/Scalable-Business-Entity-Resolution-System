import pandas as pd
from bers.validation.output_checks import validate_submission_files
from bers.constants import MATCHING_RESULTS_FILENAME
from pathlib import Path

def test_validate_submission_files_success(tmp_path: Path):
    # Create valid mock file
    df = pd.DataFrame({
        'record_id_1': ['1'],
        'record_id_2': ['2'],
        'score': [0.9]
    })
    
    file_path = tmp_path / MATCHING_RESULTS_FILENAME
    df.to_csv(file_path, sep='\t', index=False)
    
    assert validate_submission_files(tmp_path) is True

def test_validate_submission_files_missing_col(tmp_path: Path):
    # Create invalid mock file
    df = pd.DataFrame({
        'record_id_1': ['1'],
        # missing record_id_2
        'score': [0.9]
    })
    
    file_path = tmp_path / MATCHING_RESULTS_FILENAME
    df.to_csv(file_path, sep='\t', index=False)
    
    assert validate_submission_files(tmp_path) is False
    
def test_validate_submission_files_nan(tmp_path: Path):
    # Create invalid mock file (contains NaN)
    df = pd.DataFrame({
        'record_id_1': ['1'],
        'record_id_2': ['2'],
        'score': [None]
    })
    
    file_path = tmp_path / MATCHING_RESULTS_FILENAME
    df.to_csv(file_path, sep='\t', index=False)
    
    assert validate_submission_files(tmp_path) is False
