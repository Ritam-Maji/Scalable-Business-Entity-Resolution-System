import argparse
import logging
from bers.pipeline import BERSPipeline

def main():
    parser = argparse.ArgumentParser(description="Run the BERS inference pipeline to generate submission TSVs.")
    args = parser.parse_args()
    
    pipeline = BERSPipeline()
    try:
        pipeline.run_inference_pipeline()
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
