import zipfile
from pathlib import Path
import logging
from bers.config import get_config_path, load_yaml_config, resolve_path
from bers.constants import MATCHING_RESULTS_FILENAME, CANDIDATE_PAIRS_FILENAME

def build_submission_zip():
    """
    Packages the repository into the exact ZIP structure required by the competition SRS (Section 13).
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.info("Starting submission packaging...")
    
    config = load_yaml_config(get_config_path("default"))
    project_root = resolve_path('.')
    output_dir = resolve_path(config.get('output_dir', 'output'))
    zip_path = project_root / "submission.zip"
    
    # Exclude bulky/secret files not allowed in submission
    excludes = ['dataset', 'logs', '.venv', '.git', '__pycache__', '.pytest_cache', 'submission.zip']
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        
        # 1. Copy project root into code/business_entity_resolution/
        for p in project_root.rglob('*'):
            # Check against excludes
            if any(excl in p.parts for excl in excludes):
                continue
                
            if p.is_file():
                rel_path = p.relative_to(project_root)
                # Ensure it goes into code/business_entity_resolution/ per Section 13
                zip_target = Path("code") / "business_entity_resolution" / rel_path
                zf.write(p, arcname=str(zip_target).replace('\\', '/')) # ensure zip standard paths
                
        # 2. Copy the required files from output/ into the zip-root output/ folder
        matches_file = output_dir / MATCHING_RESULTS_FILENAME
        if matches_file.exists():
            zf.write(matches_file, arcname=f"output/{MATCHING_RESULTS_FILENAME}")
        else:
            logging.warning(f"Missing {MATCHING_RESULTS_FILENAME} - run inference first!")
            
        candidates_file = output_dir / CANDIDATE_PAIRS_FILENAME
        if candidates_file.exists():
            zf.write(candidates_file, arcname=f"output/{CANDIDATE_PAIRS_FILENAME}")
            
        # 3. Copy methodology document to the zip root
        doc_file = project_root / "docs" / "Documentation_template.md"
        if doc_file.exists():
            zf.write(doc_file, arcname="Documentation_template.md")
        else:
            logging.warning("Missing methodology doc (Documentation_template.md)!")
            
    logging.info(f"Submission ZIP built successfully at: {zip_path}")

if __name__ == "__main__":
    build_submission_zip()
