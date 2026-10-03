# Methodology Document

## 1. Normalization Engine
- **Text Cleaning**: All strings are lowercased, unicode NFKD normalized to remove accents, and punctuation is stripped.
- **Business Names**: Common corporate suffixes (LLC, Inc, GmbH) are stripped from the end. `DBA` (doing business as) substrings are isolated to capture the most distinct token.
- **Addresses & Countries**: Standardized using a mapping dictionary (e.g. "Street" -> "st", "United States" -> "us").

## 2. Layered Blocking Engine
To satisfy architectural constraints, we employ a set-union across multiple independent blocking signals:
- **Exact Key Blocker**: Groups records on deterministic composite keys (e.g., `country` + `city`).
- **Token Inverted Index**: Identifies records sharing at least one rare string token, bypassing expensive Cartesian comparisons.

## 3. Feature Generation
For candidate pairs surviving the blocking phase, we generate:
- Token Jaccard similarities
- Exact Boolean matches (-1.0 imputed for missing categorical data)
- difflib Sequence Matcher ratios for fuzzy string alignment

## 4. Matching Engine & Thresholding
- **Model**: A LightGBM Classifier (`lgbm_model.py`) trains on the engineered features. The model state, hyperparameters, and feature importances are saved alongside to satisfy `ML-003` auditability.
- **Decision Engine**: Instead of defaulting to 0.5, `threshold_search.py` sweeps probabilities to find the threshold that empirically maximizes the **F0.5** score against the local validation holdout.

## 5. Output Consistency (Windows)
A dedicated `write_tsv` utility explicitly enforces `newline=''` via Python's standard IO context to prevent Windows `\r\n` double-encoding, fully satisfying `CON-001`.
