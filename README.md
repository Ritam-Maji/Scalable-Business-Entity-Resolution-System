# Scalable Business Entity Resolution System

A scalable, Python-based entity resolution pipeline designed for accurately matching noisy and unstructured business records across multiple disconnected data sources (S1, S2, S3). It achieves high-precision and high-recall outputs via deeply normalized candidate generation and ML-based threshold classification.

## Core Features
* **PyArrow String Normalization:** Bypasses Pandas' memory-heavy object strings to process 12.5M rows rapidly with C++ backing.
* **Multi-Stage Blocking Retrieval:** Employs Exact Name and Rare Token blocking (with frequency pruning) to drop (N^2)$ candidate comparisons down to targeted (N)$ subsets.
* **Aggressive F0.5 Optimization:** A dedicated decision module capable of sweeping threshold confidence scores to perfectly tune Macro F0.5 metrics.
* **Streaming OOM-Safe Inference:** Employs explicit chunking and garbage collection to process test targets in batches of 300,000 without crashing memory ceilings.
* **Emergency Matchers:** Includes secondary heuristic matchers utilizing deep regex normalization for standalone exact-match submission.

## Setup & Installation

**Prerequisites:**
- Python 3.9+
- Windows PowerShell

1. Allow script execution in PowerShell:
   ``powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ``
2. Run the environment setup script:
   ``powershell
   .\scripts\setup_env.ps1
   ``
3. Or manually:
   ``cmd
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ``

## Directory Structure
- dataset/: Place your raw unzipped TSV files here (	rain_source1.tsv, 	est_source1.tsv, etc).
- output/: The pipeline dumps the final candidate_pairs.tsv and matching_results.tsv here.
- configs/: YAML configuration parameters for blocking rules, threshold tuning, and feature generation.

## CLI Execution

Once the package is installed, you can use the thin CLI wrappers to execute the pipeline directly:

**1. Train the LightGBM Model**
``bash
bers-train
``

**2. Generate Predictions (Inference)**
``bash
bers-infer
``

**3. Package the Submission**
``bash
bers-package
``
This automatically strips local logs and virtual environments, packaging the exact src/ code for submission.
