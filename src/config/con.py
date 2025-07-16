import yaml
import argparse
from pathlib import Path
from typing import Dict, Any
import collections.abc

def deep_merge_dicts(d1: Dict, d2: Dict) -> Dict:
    """
    Recursively merges d2 into d1. Modifies d1 in place.
    If a key exists in both and the values are dicts, it merges them.
    Otherwise, the value from d2 overwrites the value from d1.
    """
    for k, v in d2.items():
        if k in d1 and isinstance(d1[k], dict) and isinstance(v, collections.abc.Mapping):
            deep_merge_dicts(d1[k], v)
        else:
            d1[k] = v
    return d1


class ConfigLoader:
    """
    A robust configuration loader that handles multi-level YAML files,
    model-specific overrides, and experiment profiles.
    """

    def __init__(self, default_config_path: str):
        self.default_config_path = Path(default_config_path)
        if not self.default_config_path.is_file():
            raise FileNotFoundError(f"Default config file not found at: {self.default_config_path}")

    def _load_yaml(self, path: Path) -> Dict:
        """Loads a single YAML file."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def get_config(self, args: argparse.Namespace) -> Dict:
        """
        Loads and merges configurations based on command-line arguments.

        The merge order is:
        1. Base `default` configuration from the file.
        2. Model-specific overrides (`model.<model_name>`).
        3. Profile-specific overrides (`profiles.<profile_name>`).

        Args:
            args: The parsed arguments from argparse, expected to have
                  `config_file`, `g` (model name), and `profile`.

        Returns:
            The final, merged configuration dictionary.
        """
        # 1. Load the base YAML file specified by the user
        config_path = Path(args.config_file) if args.config_file else self.default_config_path
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        print(f"Loading base configuration from: {config_path}")
        final_cfg = self._load_yaml(config_path)

        # 2. Apply model-specific overrides
        model_name = args.g
        if model_name:
            print(f"Applying overrides for model: '{model_name}'")
            model_overrides = final_cfg.get('model', {}).get(model_name, {})
            if model_overrides:
                final_cfg = deep_merge_dicts(final_cfg, model_overrides)
            else:
                print(f"Warning: No specific configuration found for model '{model_name}'. Using defaults.")

        # 3. Apply profile-specific overrides
        profile_name = args.profile
        if profile_name:
            print(f"Applying overrides for profile: '{profile_name}'")
            profile_overrides = final_cfg.get('profiles', {}).get(profile_name, {})
            if profile_overrides:
                # We need to be careful here. The overrides in the profile are nested.
                # For example, `dataset.batch_size` needs to merge into the `dataset` dict.
                final_cfg = deep_merge_dicts(final_cfg, profile_overrides)
            else:
                raise ValueError(f"Profile '{profile_name}' not found in the configuration file.")

        # Clean up meta-keys that are not part of the final config
        final_cfg.pop('model', None)
        final_cfg.pop('profiles', None)

        return final_cfg
