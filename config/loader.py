"""Configuration loader for strategy parameters."""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Config:
    """Typed configuration container."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"Config has no attribute '{name}'")

    def get(self, key: str, default: Any = None) -> Any:
        """Get nested config value using dot notation."""
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def to_dict(self) -> Dict[str, Any]:
        return self._data


def load_config(path: Optional[str] = None) -> Config:
    """Load configuration from YAML file."""
    if path is None:
        path = Path(__file__).parent / "strategy.yaml"

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return Config(data)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global config instance (lazy load)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """Set global config instance (for testing)."""
    global _config
    _config = config