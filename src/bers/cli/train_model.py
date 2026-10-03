import argparse
import logging
from bers.pipeline import BERSPipeline

def main():
    parser = argparse.ArgumentParser(description="Train the BERS ML model.")
    args = parser.parse_args()
    
    pipeline = BERSPipeline()
    try:
        pipeline.run_training_pipeline()
    except Exception as e:
        logging.error(f"Training failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
