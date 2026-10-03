import pandas as pd
import logging
from pathlib import Path
from bers.modeling.lgbm_model import EntityMatchingModel

def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_out_dir: Path) -> EntityMatchingModel:
    """
    Trains the LightGBM model on the engineered features and saves it to the artifacts directory.
    """
    logging.info(f"Training LightGBM model on {len(X_train)} candidate pairs...")
    
    # Exclude identifier and meta columns from training features
    exclude_cols = ['id_1', 'id_2', 'label', 'score']
    feature_cols = [c for c in X_train.columns if c not in exclude_cols]
    
    X = X_train[feature_cols]
    
    model = EntityMatchingModel()
    model.fit(X, y_train)
    
    logging.info(f"Training complete. Saving model and metadata to {model_out_dir}")
    model.save(model_out_dir)
    
    return model
