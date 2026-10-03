import pandas as pd
from pathlib import Path
import csv
from typing import List, Dict

def write_tsv(data: List[Dict], file_path: Path, fieldnames: List[str]) -> None:
    """
    Writes a list of dictionaries to a TSV file.
    Enforces newline='' to prevent Windows \\r\\n corruption (CON-001).
    """
    with open(file_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(data)

def write_dataframe_to_tsv(df: pd.DataFrame, file_path: Path, mode: str = 'w', header: bool = True) -> None:
    """
    Writes a pandas DataFrame to a TSV file safely on Windows.
    Enforces newline='' by passing the explicitly opened file handle.
    """
    with open(file_path, mode=mode, encoding='utf-8', newline='') as f:
        df.to_csv(f, sep='\t', index=False, header=header)
