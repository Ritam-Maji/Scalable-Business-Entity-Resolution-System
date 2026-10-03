import argparse
import logging
from bers.packaging.build_zip import build_submission_zip

def main():
    parser = argparse.ArgumentParser(description="Build the final competition ZIP file.")
    args = parser.parse_args()
    
    try:
        build_submission_zip()
    except Exception as e:
        logging.error(f"Packaging failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
