# Business Entity Resolution System (BERS)

This repository contains an end-to-end entity resolution pipeline to match business records across multiple sources (S1, S2, S3), satisfying the requirements laid out in the project SRS.

## 🚀 Setup & Installation

### Windows Prerequisites
- Python 3.9+
- PowerShell (for the setup script)

### Installation
1. Open PowerShell and ensure execution policies allow scripts:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```
2. Run the environment setup script:
   ```powershell
   .\scripts\setup_env.ps1
   ```
   Or manually:
   ```cmd
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ```

## 📁 Directory Structure
- `dataset/`: Place your `train_source1.tsv`, `test_source1.tsv` etc. here.
- `output/`: Where the pipeline drops `candidate_pairs.tsv` and `matching_results.tsv`.
- `configs/`: YAML files governing random seeds and pipeline parameters.

## 🏃 Running the Pipeline

Once the package is installed (`pip install -e .`), you can use the thin CLI wrappers from anywhere in the environment:

**1. Train the ML Model**
```bash
bers-train
```

**2. Generate Predictions (Inference)**
```bash
bers-infer
```

**3. Package the Final Submission ZIP**
```bash
bers-package
```
This automatically strips local logs/venvs and places exactly what the SRS requires into a `submission.zip`.

## 🧪 Running Tests
```bash
pytest tests/
```
