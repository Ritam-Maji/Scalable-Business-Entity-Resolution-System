import pandas as pd
import logging
from typing import Set, Tuple, List, Dict
from bers.constants import COL_RECORD_ID, COL_ID_1, COL_ID_2
from bers.features.name_features import extract_name_features
from bers.features.address_features import extract_address_features
from bers.features.agreement_features import extract_agreement_features

class FeatureBuilder:
    """
    Builds the feature matrix (X) for ML scoring from candidate pairs.
    It takes the raw/normalized dataframe and the generated pairs,
    and maps them to a set of numerical features for the model.
    """
    
    def __init__(self):
        pass
        
    def build_features(self, df: pd.DataFrame, pairs) -> pd.DataFrame:
        """
        Constructs a DataFrame of pairwise features from a candidate generator.
        """
        logging.info("Building features from candidate generator...")
        
        if df.empty:
            return pd.DataFrame()
            
        df_indexed = df.set_index(COL_RECORD_ID)
        
        feature_chunks = []
        CHUNK_SIZE = 250000
        
        id1_chunk = []
        id2_chunk = []
        processed_count = 0
        
        def _process_batch(ids1, ids2):
            rows1 = df_indexed.reindex(ids1).fillna('')
            rows2 = df_indexed.reindex(ids2).fillna('')
            
            n1 = rows1['name'].astype(str).values
            n2 = rows2['name'].astype(str).values
            a1 = rows1['address'].astype(str).values
            a2 = rows2['address'].astype(str).values
            
            # Get other columns if they exist, else empty strings
            def get_col(r, col):
                return r[col].astype(str).values if col in r.columns else [''] * len(r)
                
            c1 = get_col(rows1, 'city')
            c2 = get_col(rows2, 'city')
            s1 = get_col(rows1, 'state')
            s2 = get_col(rows2, 'state')
            co1 = get_col(rows1, 'country')
            co2 = get_col(rows2, 'country')
            z1 = get_col(rows1, 'zip_code')
            z2 = get_col(rows2, 'zip_code')
            
            batch_features = []
            for i in range(len(ids1)):
                f_name = extract_name_features(n1[i], n2[i])
                f_addr = extract_address_features(a1[i], a2[i])
                f_agree = extract_agreement_features(c1[i], c2[i], s1[i], s2[i], co1[i], co2[i], z1[i], z2[i])
                
                combined = {COL_ID_1: ids1[i], COL_ID_2: ids2[i]}
                combined.update(f_name)
                combined.update(f_addr)
                combined.update(f_agree)
                batch_features.append(combined)
                
            return pd.DataFrame(batch_features)

        for id1, id2 in pairs:
            id1_chunk.append(id1)
            id2_chunk.append(id2)
            
            if len(id1_chunk) >= CHUNK_SIZE:
                feature_chunks.append(_process_batch(id1_chunk, id2_chunk))
                processed_count += len(id1_chunk)
                logging.info(f"Processed {processed_count} pairs... dumped chunk.")
                id1_chunk.clear()
                id2_chunk.clear()
                
        if id1_chunk:
            feature_chunks.append(_process_batch(id1_chunk, id2_chunk))
            processed_count += len(id1_chunk)
            logging.info(f"Processed {processed_count} pairs... finished final chunk.")
            
        logging.info("Feature building complete.")
        
        if not feature_chunks:
            return pd.DataFrame()
            
        return pd.concat(feature_chunks, ignore_index=True)
