"""
This is a script for setting system level options for AAanalysis.
"""
from typing import Dict, Any

# System level options
verbose = True

# Enables setting of system level variables like in matplotlib
class Settings:
    def __init__(self):
        self._settings: Dict[str, Any] = {
            'verbose': verbose,
        }

    def __getitem__(self, key: str) -> Any:
        """Retrieve a setting's value using dict-like access."""
        return self._settings.get(key, None)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a setting's value using dict-like access."""
        self._settings[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if a key is in the settings."""
        return key in self._settings

# Global settings instance
options = Settings()