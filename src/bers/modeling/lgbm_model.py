import lightgbm as lgb
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import datetime
from typing import Dict, Any, List

class EntityMatchingModel:
    """
    LightGBM classifier for entity matching (ML-002).
    Includes methods for saving/loading with full tracking metadata (ML-003).
    """
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'random_state': 42,
            'verbose': -1
        }
        self.model = None
        self.features: List[str] = []
        
    def fit(self, X: pd.DataFrame, y: pd.Series, categorical_feature='auto') -> None:
        self.features = list(X.columns)
        train_data = lgb.Dataset(X, label=y)
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=100,
            categorical_feature=categorical_feature
        )
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be trained or loaded before calling predict.")
            
        # Ensure features align exactly as during training
        X_pred = X[self.features]
        return self.model.predict(X_pred)
        
    def save(self, model_dir: Path) -> None:
        """Saves model weights and auditable metadata."""
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / 'lgbm_model.joblib'
        joblib.dump(self.model, model_path)
        
        metadata = {
            'timestamp': datetime.datetime.now().isoformat(),
            'features': self.features,
            'params': self.params,
            'feature_importance': self.model.feature_importance().tolist() if self.model else []
        }
        
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def load(self, model_dir: Path) -> None:
        """Loads model weights and tracking metadata."""
        model_path = model_dir / 'lgbm_model.joblib'
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model = joblib.load(model_path)
        
        meta_path = model_dir / 'metadata.json'
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                self.features = metadata.get('features', [])
                self.params = metadata.get('params', self.params)
