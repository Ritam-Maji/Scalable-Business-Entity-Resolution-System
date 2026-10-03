import yaml
import random
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any

# Project root is assumed to be 3 levels up from this file: src/bers/config.py -> src/bers -> src -> project_root
# Actually it is 2 levels up if we consider src as the package root, but physically:
# config.py is in src/bers. Parent is bers, parent is src, parent is business_entity_resolution.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Loads a YAML config file and returns it as a dictionary."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_config_path(config_name: str) -> Path:
    """Returns the absolute Windows-safe path to a config file."""
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"
    return PROJECT_ROOT / "configs" / config_name

def resolve_path(relative_path: str) -> Path:
    """Resolves a path string from a config file to an absolute Windows-safe Path object."""
    # This ensures that paths like "dataset/train/" are resolved relative to the project root.
    return PROJECT_ROOT / Path(relative_path)

def set_random_seed(seed: int) -> None:
    """Sets random seeds for reproducibility (NFR-004)."""
    random.seed(seed)
    np.random.seed(seed)
    logging.info(f"Global random seed set to {seed}")

class ConfigManager:
    """A simple manager to load and merge configurations."""
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
        
    def load_all(self):
        # Load default
        default_path = get_config_path("default")
        if default_path.exists():
            self.config.update(load_yaml_config(default_path))
            
        # Optional modular configs could be loaded here and merged
        for mod in ["blocking", "features", "model"]:
            mod_path = get_config_path(mod)
            if mod_path.exists():
                self.config[mod] = load_yaml_config(mod_path)
                
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

# Global configuration instance
global_config = ConfigManager()
