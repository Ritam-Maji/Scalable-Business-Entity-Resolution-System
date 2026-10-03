from pathlib import Path

# Data Source Prefixes (used to ensure globally unique record IDs across sources)
SOURCE_1_PREFIX = "S1-"
SOURCE_2_PREFIX = "S2-"
SOURCE_3_PREFIX = "S3-"

# Column Names (Standardized internal schema)
COL_RECORD_ID = "record_id"
COL_NAME = "name"
COL_ADDRESS = "address"
COL_CITY = "city"
COL_STATE = "state"
COL_ZIP = "zip_code"
COL_COUNTRY = "country"
COL_PHONE = "phone"
COL_WEBSITE = "website"

# Pair Schema (for blocking/candidates)
COL_ID_1 = "id_1"
COL_ID_2 = "id_2"
COL_SCORE = "score"
COL_LABEL = "label"

# Output Column Names (Required by competition validator)
OUT_COL_RECORD_ID_1 = "id_1"
OUT_COL_RECORD_ID_2 = "id_2"
OUT_COL_SCORE = "score"

# File Names
CANDIDATE_PAIRS_FILENAME = "candidate_pairs.tsv"
MATCHING_RESULTS_FILENAME = "matching_results.tsv"
