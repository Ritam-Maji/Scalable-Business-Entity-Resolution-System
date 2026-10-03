import pandas as pd
import logging
from pathlib import Path
from bers.modeling.lgbm_model import EntityMatchingModel

def predict_scores(X_test: pd.DataFrame, model_dir: Path) -> pd.Series:
    """
    Loads the trained ML model and predicts match probabilities (scores) 
    for the provided candidate pairs.
    """
    logging.info(f"Loading ML model from {model_dir}")
    model = EntityMatchingModel()
    model.load(model_dir)
    
    logging.info(f"Predicting match scores for {len(X_test)} candidate pairs...")
    
    # The model's predict_proba expects just the feature columns.
    # We rely on the model object to filter features based on its saved metadata.
    # (i.e. it automatically drops id_1, id_2 etc. by strictly selecting self.features)
    scores = model.predict_proba(X_test)
    
    return pd.Series(scores, index=X_test.index, name='score')
