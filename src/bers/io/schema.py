import pandas as pd
from bers.constants import (
    COL_RECORD_ID, COL_NAME, COL_ADDRESS, COL_CITY, 
    COL_STATE, COL_ZIP, COL_COUNTRY
)

EXPECTED_COLUMNS = [
    COL_RECORD_ID, COL_NAME, COL_ADDRESS, COL_CITY, 
    COL_STATE, COL_ZIP, COL_COUNTRY
]

def validate_dataframe(df: pd.DataFrame, source_name: str) -> None:
    """Validates that a dataframe has the expected columns."""
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataframe from {source_name} is missing expected columns: {missing}")
