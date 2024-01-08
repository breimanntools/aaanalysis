"""
This is a script for setting system level options for AAanalysis.

Notes
-----
verbose
    Whether verbose mode should be enabled or not.
random_state
    Random state variable used for stochastic models from packages like scipy or scikit-learn.
ext_len
    Length of TMD-extending part (starting from C and N terminal part of TMD, >=0).
df_scales
    Scale DataFrame used in CPP algorithm. Adjust on system level if non-default scales are used.
df_cat
    Scale category DataFrame used in CPP algorithm. Adjust on system level if non-default scale categories are used.
"""
from typing import Dict, Any

# System level options
verbose = True
random_state = 42
ext_len = 0
df_scales = None
df_cat = None


# Enables setting of system level variables like in matplotlib
class Settings:
    def __init__(self):
        self._settings: Dict[str, Any] = {
            'verbose': verbose,
            'ext_len': ext_len,
            'random_state': random_state,
            'df_scales': df_scales,
            'df_cat': df_cat,
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
