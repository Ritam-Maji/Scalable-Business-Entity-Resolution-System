# Scalable Business Entity Resolution System

A scalable, Python-based entity resolution pipeline for matching noisy, unstructured business records across multiple disconnected data sources. It combines deep text normalization, multi-stage candidate retrieval, similarity-based feature engineering, and LightGBM classification to produce precision-focused matches efficiently, even at the scale of **12.5M+ records**.

## Core Features

- **PyArrow-Backed Normalization:** Avoids memory-heavy Pandas object strings by using PyArrow's C++-backed string kernels, so 12.5M rows can be cleaned quickly and with controlled memory.
- **Multi-Stage Candidate Retrieval:** Combines Exact Name, Rare Token, and Address Token blocking, with frequency pruning to discard overly common tokens. This reduces the comparison space from O(N²) all-pairs to a targeted, near-linear set of candidates.
- **LightGBM Matching Model:** Classifies candidate pairs using engineered name, address, token-overlap, and categorical-agreement features.
- **F₀.₅ Threshold Optimization:** Sweeps classification thresholds to maximize the precision-weighted F₀.₅ metric.
- **Streaming, OOM-Safe Inference:** Processes candidate pairs and features in configurable chunks (default 300,000 rows) with explicit garbage collection to stay within memory limits.
- **Singleton-Aware Matching:** Handles entities with zero, one, or multiple corresponding records.
- **Fallback Heuristic Matchers:** Includes secondary exact-match matchers built on deep regex normalization, useful as baselines or when the model is unavailable.
- **Validated Outputs:** Generates candidate and final matching results with schema validation.

## Tech Stack

Python, Pandas, NumPy, PyArrow, Parquet, LightGBM, Scikit-learn, RapidFuzz, pytest, Jupyter, Git

## Pipeline Architecture

```
Business Records (S1, S2, S3)
      │
      ▼
Text Normalization (PyArrow)
      │
      ▼
Multi-Stage Candidate Retrieval
 ┌────┼───────────────┐
 ▼    ▼               ▼
Exact  Rare Tokens   Address
Name   (freq-pruned)  Tokens
 └────┼───────────────┘
      ▼
Candidate Deduplication
      │
      ▼
Feature Engineering
 ├── Name Similarity
 ├── Address Similarity
 ├── Token Overlap
 └── Categorical Agreement
      │
      ▼
LightGBM Classification
      │
      ▼
F₀.₅ Threshold Optimization
      │
      ▼
Final Entity Matches
```

## Project Structure

```
├── src/
│   └── bers/
│       ├── blocking/
│       ├── features/
│       ├── model/
│       ├── evaluation/
│       └── ...
├── tests/
├── notebooks/
├── docs/
├── configs/        # YAML config for blocking, thresholds, features
├── scripts/
├── dataset/        # Input TSV files (not tracked)
├── output/         # Generated results (not tracked)
├── README.md
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Dataset

The system works with large tab-separated business datasets containing identifiers, business names, addresses, and country information.

For privacy, size, and repository-management reasons, datasets are **not included** in this repository. To run the pipeline, place the unzipped TSV files in the local `dataset/` directory (for example, `train_source1.tsv` and `test_source1.tsv`, along with the corresponding files for the other sources).

## Setup

### Prerequisites

- Python 3.9+
- Windows PowerShell
- Git

### Installation

1. Allow script execution for the current PowerShell session:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```

2. Run the setup script:

   ```powershell
   .\scripts\setup_env.ps1
   ```

   Or set up manually:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   pip install -e .
   ```

## Running the Pipeline

After installation, use the CLI commands:

| Step | Command | Description |
|------|---------|-------------|
| Train | `bers-train` | Trains the LightGBM model and tunes the F₀.₅ threshold |
| Infer | `bers-infer` | Generates candidate pairs and predictions in streaming chunks |
| Package | `bers-package` | Bundles the `src/` code for submission, excluding logs and virtual environments |

Blocking rules, threshold tuning, and feature generation are controlled through the YAML files in `configs/`.

## Outputs

The pipeline writes the following files locally:

```
output/
├── candidate_pairs.tsv
└── matching_results.tsv
```

These generated files are intentionally excluded from version control.

## Results

> Add your evaluation metrics here once available.

| Metric | Value |
|--------|-------|
| Precision | _TBD_ |
| Recall | _TBD_ |
| F₀.₅ | _TBD_ |
| Candidate recall (blocking) | _TBD_ |
| Reduction ratio vs. all-pairs | _TBD_ |

## Testing

Run the full test suite with:

```powershell
pytest
```

The tests cover normalization, candidate retrieval, feature generation, evaluation, output validation, and end-to-end pipeline behavior.

## License

See [`LICENSE`](LICENSE) for licensing information.
