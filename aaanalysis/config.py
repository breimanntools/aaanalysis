"""
This is a script for setting system level options for AAanalysis.
"""
from typing import Dict, Any

# System level options
verbose = True
random_state = 42
name_tmd = "TMD"
name_jmd_n = "JMD-N"
name_jmd_c = "JMD-C"
ext_len = 0
df_scales = None
df_cat = None

# DEV: Parameters are used as directive to get better documentation style
# Enables setting of system level variables like in matplotlib
class Settings:
    """
     A class for managing system-level settings for AAanalysis.

    This class mimics a dictionary-like interface, allowing the setting and retrieving
    of system-level options. It is designed to be used as a single global instance, ``options``.

    Paramters
    ---------
    The following options can be set:

    verbose : bool, default=True
        Whether verbose mode should be enabled or not.
    random_state : int, default=42
        Random state variable used for stochastic models from packages like scipy or scikit-learn.
    name_tmd : str, default='TMD'
        Name of target middle domain (TMD) used in CPP plots.
    name_jmd_n : str, default='JMD_N'
        Name of N-terminal juxta middle domain (JMD-N) used in CPP plots.
    name_jmd_c : str, default='JMD_C'
        Name of C-terminal juxta middle domain (JMD-C) used in CPP plots.
    ext_len : int, default=0
        Length of TMD-extending part (starting from C and N terminal part of TMD, >=0). Disabled (set to 0) by default.
    df_scales : DataFrame, optional
        Scale DataFrame used in CPP algorithm. Adjust on system level if non-default scales are used.
        If ``None``, AAanalysis framework will use the scale DataFrame loaded by :func:`load_scales` with ``name='scales'``.
    df_cat : DataFrame, optional
        Scale category DataFrame used in CPP algorithm. Adjust on system level if non-default scale categories are used.
        If ``None``, AAanalysis framework will use the scale category DataFrame loaded by :func:`load_scales` with ``name='scales_cat'``.

    See Also
    --------
    * :class:`numpy.random.RandomState` for details on the ``random_state`` variable used to make stochastic processes
      yielding consistent results.
    * :class:`SequenceFeature` for definition of sequence ``Parts``.
    * :func:`load_scales` for details on scale and scale category DataFrames.

    Examples
    --------
    .. include:: examples/options.rst
    """
    def __init__(self):
        self._settings: Dict[str, Any] = {
            'verbose': verbose,
            'random_state': random_state,
            'name_tmd': name_tmd,
            'name_jmd_n': name_jmd_n,
            'name_jmd_c': name_jmd_c,
            'ext_len': ext_len,
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

    def __str__(self) -> str:
        """Return a string representation of the settings dictionary."""
        return str(self._settings)


# Global settings instance
options = Settings()
