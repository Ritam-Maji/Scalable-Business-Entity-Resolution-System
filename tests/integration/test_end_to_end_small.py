import pytest
import pandas as pd
from pathlib import Path
from bers.pipeline import BERSPipeline
from bers.constants import MATCHING_RESULTS_FILENAME

def test_pipeline_dry_run_no_data(tmp_path: Path):
    """
    Tests that the pipeline fails gracefully and doesn't crash 
    if the data directories are empty (which they are until user provides them).
    """
    # Create a mock pipeline pointing to tmp dirs
    pipeline = BERSPipeline()
    pipeline.dataset_dir = tmp_path / "dataset"
    pipeline.output_dir = tmp_path / "output"
    pipeline.artifacts_dir = tmp_path / "artifacts"
    
    pipeline.dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Should run gracefully and log missing files without unhandled exceptions
    pipeline.run_training_pipeline()
    pipeline.run_inference_pipeline()
    
    # The output file shouldn't be created successfully since no data was processed
    assert not (pipeline.output_dir / MATCHING_RESULTS_FILENAME).exists()
