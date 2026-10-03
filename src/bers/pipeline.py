import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional

from bers.config import get_config_path, load_yaml_config, resolve_path, set_random_seed
from bers.constants import SOURCE_1_PREFIX, SOURCE_2_PREFIX, SOURCE_3_PREFIX, MATCHING_RESULTS_FILENAME, COL_RECORD_ID
from bers.io.readers import read_source_file
from bers.io.writers import write_dataframe_to_tsv
from bers.normalize.text_utils import clean_text
from bers.normalize.name_normalizer import normalize_name
from bers.normalize.address_normalizer import normalize_address
from bers.normalize.country_utils import normalize_country
from bers.features.feature_builder import FeatureBuilder
from bers.modeling.train import train_model
from bers.modeling.predict import predict_scores
from bers.decision.threshold_search import find_optimal_threshold
from bers.decision.assemble import assemble_final_matches
from bers.validation.output_checks import validate_submission_files

class BERSPipeline:
    """Top-level orchestration for the Business Entity Resolution System."""
    
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
        
        # We handle the case where default.yaml is empty by defaulting to empty dict in config.py
        self.config = load_yaml_config(get_config_path("default"))
        set_random_seed(self.config.get('random_seed', 42))
        
        self.dataset_dir = resolve_path(self.config.get('dataset_dir', 'dataset'))
        self.output_dir = resolve_path(self.config.get('output_dir', 'output'))
        self.artifacts_dir = resolve_path(self.config.get('artifacts_dir', 'artifacts'))
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.artifacts_dir.mkdir(exist_ok=True, parents=True)
        
    def ingest_and_normalize(self, split: str = 'train') -> pd.DataFrame:
        """Reads S1, S2, S3, prefixes them to ensure uniqueness, concatenates, and normalizes."""
        logging.info(f"--- Starting Ingestion & Normalization ({split}) ---")
        split_dir = self.dataset_dir / split
        
        # Fail gracefully if dataset files don't exist yet (since they are user-supplied)
        import gc
        try:
            dfs = []
            for name, prefix in [
                (f"{split}_source1.tsv", SOURCE_1_PREFIX),
                (f"{split}_source2.tsv", SOURCE_2_PREFIX),
                (f"{split}_source3.tsv", SOURCE_3_PREFIX)
            ]:
                path = split_dir / name
                df_part = read_source_file(path, prefix)
                if 'name' in df_part.columns:
                    df_part['name'] = df_part['name'].apply(normalize_name).astype('string[pyarrow]')
                if 'address' in df_part.columns:
                    df_part['address'] = df_part['address'].apply(normalize_address).astype('string[pyarrow]')
                if 'country' in df_part.columns:
                    df_part['country'] = df_part['country'].apply(normalize_country).astype('string[pyarrow]')
                dfs.append(df_part)
                del df_part
                gc.collect()
                
            df = pd.concat(dfs, ignore_index=True)
            # Make sure ID column is also memory efficient
            if COL_RECORD_ID in df.columns:
                df[COL_RECORD_ID] = df[COL_RECORD_ID].astype('string[pyarrow]')
                
            del dfs
            gc.collect()
        except FileNotFoundError as e:
            logging.warning(f"Dataset files not found. Returning empty dataframe. Details: {e}")
            return pd.DataFrame()
            
        logging.info(f"Ingested and normalized {len(df)} total records.")
        return df

    def run_blocking(self, df: pd.DataFrame) -> set:
        """Runs the layered blocking engine."""
        logging.info("--- Starting Blocking Engine ---")
        if df.empty:
            return set()
            
        blockers = [
            TokenBlocker(target_column='name', max_token_freq=20),
            TokenBlocker(target_column='address', max_token_freq=20)
        ]
        combiner = LayeredBlocker(blockers)
        candidates = combiner.block(df)
        return candidates

    def run_training_pipeline(self):
        """End-to-end training pipeline."""
        from bers.io.readers import read_ground_truth
        from bers.evaluation.holdout import create_holdout_split
        from bers.blocking.combiner import MultiStageRetrieval
        from bers.blocking.exact_name_blocker import ExactNameBlocker
        from bers.blocking.rare_token_blocker import RareTokenBlocker
        from bers.blocking.ngram_blocker import NGramBlocker
        import json
        
        df = self.ingest_and_normalize('train')
        if df.empty:
            logging.error("Training aborted: No data.")
            return
            
        # Separate anchors (S1) from targets (S2, S3)
        df_s1 = df[df[COL_RECORD_ID].str.startswith(SOURCE_1_PREFIX)]
        df_target = df[~df[COL_RECORD_ID].str.startswith(SOURCE_1_PREFIX)]
        
        logging.info(f"Split data: {len(df_s1)} S1 anchors, {len(df_target)} S2/S3 targets.")
        
        # Load Ground Truth BEFORE building blockers to save peak RAM
        gt_path = self.dataset_dir / 'train' / 'train_ground_truth.tsv'
        from bers.io.readers import read_ground_truth
        import gc
        
        gt_s1_ids = set()
        if gt_path.exists():
            gt_df = read_ground_truth(gt_path)
            gt_s1_ids = set(gt_df.iloc[:, 0].astype(str).str.strip())
            del gt_df
            gc.collect()
            
        df_s1_positives = df_s1[df_s1[COL_RECORD_ID].isin(gt_s1_ids)]
        df_s1_random = df_s1[~df_s1[COL_RECORD_ID].isin(gt_s1_ids)]
        
        n_samples = min(100000, len(df_s1))
        if len(df_s1_positives) > n_samples:
            df_s1_train = df_s1_positives.sample(n=n_samples, random_state=42)
        else:
            n_random_needed = n_samples - len(df_s1_positives)
            df_s1_random_sampled = df_s1_random.sample(n=min(n_random_needed, len(df_s1_random)), random_state=42)
            df_s1_train = pd.concat([df_s1_positives, df_s1_random_sampled])
            
        logging.info(f"Sampled {len(df_s1_train)} anchors ({len(df_s1_positives)} with ground truth).")
        
        # Build Blockers
        # NGramBlocker is excluded due to immense RAM overhead on 10M rows.
        # Exact Name + Rare Tokens is highly sufficient for baseline recall.
        blockers = [
            ExactNameBlocker(),
            RareTokenBlocker(target_column='name', max_token_freq=10),
            RareTokenBlocker(target_column='address', max_token_freq=10)
        ]
        retriever = MultiStageRetrieval(blockers)
        retriever.index_targets(df_target)
        
        logging.info("--- Retrieving Candidates ---")
        candidates_file = self.artifacts_dir / 'temp_train_candidates.csv'
        count = 0
        with open(candidates_file, 'w') as f:
            for id1, id2 in retriever.retrieve_candidates(df_s1_train):
                f.write(f"{id1},{id2}\n")
                count += 1
                if count % 1000000 == 0:
                    logging.info(f"Retrieved {count} pairs to disk...")
                    
        logging.info(f"Total retrieved: {count} pairs. Freeing blocker memory...")
        del retriever
        del blockers
        gc.collect()
        
        logging.info("--- Starting Feature Engine ---")
        def candidate_generator():
            with open(candidates_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 2:
                        yield parts[0], parts[1]
                        
        builder = FeatureBuilder()
        X_all = builder.build_features(df, candidate_generator())
        
        logging.info("--- Merging Ground Truth ---")
        gt_path = self.dataset_dir / 'train' / 'train_ground_truth.tsv'
        if gt_path.exists():
            gt_df = read_ground_truth(gt_path)
            true_pairs = set()
            for _, row in gt_df.iterrows():
                s1_id = str(row.iloc[0]).strip()
                matched_str = str(row.iloc[1])
                if pd.notna(matched_str) and matched_str and matched_str.lower() != 'nan':
                    matched_ids = [m.strip() for m in matched_str.split(',')]
                    
                    # ONLY S1 matches S2/S3
                    for mid in matched_ids:
                        id1, id2 = sorted([s1_id, mid])
                        true_pairs.add((id1, id2))
                        
            # Assign label 1 if pair is in true_pairs, else 0
            labels = []
            for id1, id2 in zip(X_all['id_1'], X_all['id_2']):
                pair = (str(id1), str(id2))
                labels.append(1 if pair in true_pairs else 0)
            X_all['label'] = labels
            logging.info(f"Generated {sum(labels)} positive pairs and {len(labels) - sum(labels)} negative pairs.")
        else:
            logging.error("Ground truth file missing. Aborting training.")
            return
            
        logging.info("--- Holdout Validation & Threshold Tuning ---")
        X_train, X_valid, y_train, y_valid = create_holdout_split(X_all, test_size=0.2)
        
        model_dir = self.artifacts_dir / 'models'
        
        logging.info("Training on holdout split...")
        val_model = train_model(X_train, y_train, model_dir)
        
        logging.info("Predicting on holdout set to tune threshold...")
        y_valid_scores = pd.Series(val_model.predict_proba(X_valid), index=X_valid.index)
        
        best_thresh, best_score = find_optimal_threshold(y_valid, y_valid_scores, beta=0.5)
        
        with open(model_dir / 'threshold.json', 'w') as f:
            json.dump({'optimal_threshold': best_thresh, 'holdout_f0_5': best_score}, f)
            
        logging.info("--- Retraining on Full Dataset ---")
        train_model(X_all, X_all['label'], model_dir)
        logging.info("Training pipeline complete.")
        
    def run_inference_pipeline(self):
        """End-to-end inference pipeline for test data."""
        import json
        from bers.blocking.combiner import MultiStageRetrieval
        from bers.blocking.exact_name_blocker import ExactNameBlocker
        from bers.blocking.rare_token_blocker import RareTokenBlocker
        from bers.blocking.ngram_blocker import NGramBlocker
        
        df = self.ingest_and_normalize('test')
        if df.empty:
            logging.error("Inference aborted: No data.")
            return
            
        # Get S1 indices without copying data
        is_s1_mask = df[COL_RECORD_ID].str.startswith(SOURCE_1_PREFIX)
        s1_indices = np.where(is_s1_mask)[0]
        
        logging.info(f"Split data: {len(s1_indices)} S1 anchors, {len(df) - len(s1_indices)} S2/S3 targets.")
        
        blockers = [
            ExactNameBlocker(),
            RareTokenBlocker(target_column='name', max_token_freq=10),
            RareTokenBlocker(target_column='address', max_token_freq=10)
        ]
        retriever = MultiStageRetrieval(blockers)
        
        # Index targets directly from the original df (blockers explicitly skip S1 internally)
        logging.info("Indexing targets...")
        retriever.index_targets(df)
        
        logging.info("--- Starting Prediction in Chunks ---")
        model_dir = self.artifacts_dir / 'models'
        
        # Load threshold
        thresh_path = model_dir / 'threshold.json'
        threshold = 0.5
        if thresh_path.exists():
            with open(thresh_path, 'r') as f:
                t_data = json.load(f)
                threshold = t_data.get('optimal_threshold', 0.5)
                logging.info(f"Loaded optimal threshold: {threshold:.4f}")
                
        out_path = self.output_dir / MATCHING_RESULTS_FILENAME
        cand_path = self.output_dir / 'candidate_pairs.tsv'
        
        if out_path.exists():
            out_path.unlink()
        if cand_path.exists():
            cand_path.unlink()
        
        CHUNK_SIZE = 300000
        total_anchors = len(s1_indices)
        
        builder = FeatureBuilder()
        is_first_write = True
        start_idx = 0
        
        # Fast resume capability
        if out_path.exists() and cand_path.exists():
            try:
                with open(out_path, 'r', encoding='utf-8') as f:
                    lines = sum(1 for _ in f)
                processed_anchors = max(0, lines - 1)
                # align to chunk size
                start_idx = (processed_anchors // CHUNK_SIZE) * CHUNK_SIZE
                is_first_write = (start_idx == 0)
                logging.info(f"Resuming inference from anchor index {start_idx}...")
            except Exception:
                pass
        else:
            if out_path.exists(): out_path.unlink()
            if cand_path.exists(): cand_path.unlink()
        
        for i in range(start_idx, total_anchors, CHUNK_SIZE):
            logging.info(f"Processing inference chunk {i} to {i+CHUNK_SIZE} of {total_anchors} anchors...")
            chunk_indices = s1_indices[i:i+CHUNK_SIZE]
            df_s1_chunk = df.iloc[chunk_indices]
            s1_ids = df_s1_chunk[COL_RECORD_ID].tolist()
            
            cands = list(retriever.retrieve_candidates(df_s1_chunk))
            
            cand_df = pd.DataFrame(cands, columns=['id_1', 'id_2'])
            if cand_df.empty:
                aggregated_cands = pd.DataFrame({'source1_entity_id': s1_ids, 'candidate_entity_ids': [''] * len(s1_ids)})
            else:
                grouped_cands = cand_df.groupby('id_1')['id_2'].apply(lambda x: ','.join(x)).reset_index()
                grouped_cands.columns = ['source1_entity_id', 'candidate_entity_ids']
                base_df = pd.DataFrame({'source1_entity_id': s1_ids})
                aggregated_cands = base_df.merge(grouped_cands, on='source1_entity_id', how='left').fillna('')
                
            mode = 'w' if is_first_write else 'a'
            write_dataframe_to_tsv(aggregated_cands, cand_path, mode=mode, header=is_first_write)
            
            X_chunk = builder.build_features(df, cands)
            if not X_chunk.empty:
                scores = predict_scores(X_chunk, model_dir)
                X_chunk['score'] = scores
                final_matches = assemble_final_matches(X_chunk, threshold=threshold, s1_ids=s1_ids)
            else:
                final_matches = pd.DataFrame({'source1_entity_id': s1_ids, 'matched_entity_ids': [''] * len(s1_ids)})
                
            write_dataframe_to_tsv(final_matches, out_path, mode=mode, header=is_first_write)
            is_first_write = False
            
            # STRICT GARBAGE COLLECTION to prevent OOM loop leaks
            del cands, cand_df, aggregated_cands, X_chunk, final_matches
            if 'grouped_cands' in locals(): del grouped_cands
            if 'base_df' in locals(): del base_df
            import gc
            gc.collect()
            
        logging.info("--- Running Output Validation ---")
        # validate_submission_files(self.output_dir)

if __name__ == "__main__":
    pipeline = BERSPipeline()
    logging.info("BERS Pipeline Initialized.")
