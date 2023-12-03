"""
This is a script for setting system level options for AAanalysis.

Notes
-----
- The sum of length parameters define the total number of positions (``jmd_n_len`` + ``tmd_len`` + ``jmd_c_len``).
- ``ext_len`` < ``jmd_m_len`` and ``ext_len`` < ``jmd_c_len``
ext_len
    Length of TMD-extending part (starting from C and N terminal part of TMD, >=0).
"""
from typing import Dict, Any

# System level options
verbose = True
ext_len = 0


# Enables setting of system level variables like in matplotlib
class Settings:
    def __init__(self):
        self._settings: Dict[str, Any] = {
            'verbose': verbose,
            'ext_len': ext_len,
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
