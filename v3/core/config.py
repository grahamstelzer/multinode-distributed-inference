import os
import yaml
from typing import Any, Dict

class ModelConfig:
    """
    A lightweight wrapper for reading and accessing model configuration files.
    Intended for SAM2 and similar hierarchical model configs.
    """

    def __init__(self, config_path: str):
        """
        Load a YAML or JSON config file and prepare it for model construction.

        Args:
            config_path (str): Path to the config file (.yaml or .json)
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.config_path = config_path
        self.raw_config = self._load_config(config_path)
        self.model_config = self.raw_config.get("model", {})

        print("config:")
        print(self.raw_config)
        print("\n\n")

    def _load_config(self, path: str) -> Dict[str, Any]:
        """
        Load YAML or JSON configuration file.
        """
        if path.endswith(".yaml") or path.endswith(".yml"):
            with open(path, "r") as f:
                return yaml.safe_load(f)
        elif path.endswith(".json"):
            import json
            with open(path, "r") as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path}")

    def get(self, key: str, default=None) -> Any:
        """
        Access a top-level config key safely.
        """
        return self.raw_config.get(key, default)

    def get_model_param(self, key: str, default=None) -> Any:
        """
        Access a key inside the 'model' section of the config.
        """
        return self.model_config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """
        Allow bracket-style access, e.g. config['model']
        """
        return self.raw_config[key]

    def __repr__(self):
        return f"<ModelConfig path={self.config_path}>"
