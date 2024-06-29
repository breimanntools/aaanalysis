"""
This is a script for setting system level options for AAanalysis.
# DEV: Not tested due to effects on plots and user communication -> Only evaluated by examples
"""
from typing import Dict, Any
import os

from ._utils.check_type import check_bool, check_number_val, check_number_range, check_str
from ._utils.check_data import check_df

# System level options
_dict_options = {
    'verbose': "off",
    'random_state': "off",
    'allow_multiprocessing': True,
    'name_tmd': "TMD",
    'name_jmd_n': "JMD-N",
    'name_jmd_c': "JMD-C",
    'ext_len': 0,
    'jmd_n_len': None,
    'jmd_c_len': None,
    'df_scales': None,
    'df_cat': None,
}


# Check system level (option) parameters or depending parameters
def check_verbose(verbose=None):
    """Check if general verbosity is on or off. Adjusted based on options setting and value provided to object"""
    global_verbose = options["verbose"]
    if global_verbose != "off":
        # System level verbosity
        check_bool(name="verbose (option)", val=global_verbose)
        verbose = global_verbose
    else:
        check_bool(name="verbose", val=verbose)
    return verbose


def check_random_state(random_state=None):
    """Adjust random state if global is not 'off' (default)"""
    global_random_state = options["random_state"]
    args = dict(min_val=0, accept_none=True, just_int=True)
    if global_random_state != "off":
        # System-level random state
        check_number_range(name="random_state (option)", val=global_random_state, **args)
        random_state = global_random_state
    else:
        check_number_range(name="random_state", val=random_state, **args)
    return random_state


def check_n_jobs(n_jobs=None):
    """Adjust n_jobs to 1 if multiprocessing is not allowed"""
    allow_multiprocessing = options["allow_multiprocessing"]
    check_bool(name="allow_multiprocessing (options)", val=allow_multiprocessing)
    # Disable multiprocessing
    if not allow_multiprocessing:
        n_jobs = 1
        os.environ['LOKY_MAX_CPU_COUNT'] = "1"
    # Set n_jobs to maximum number of CPUs
    if n_jobs == -1:
        n_jobs = os.cpu_count()
    # Check which n_jobs are allowed
    check_number_range(name="n_jobs", val=n_jobs, accept_none=True, just_int=True, min_val=1)
    return n_jobs


def check_jmd_n_len(jmd_n_len=None):
    """Check if general JMD-N length is given and adjust it globally."""
    global_jmd_n_len = options["jmd_n_len"]
    if global_jmd_n_len is not None:
        return global_jmd_n_len
    else:
        return jmd_n_len


def check_jmd_c_len(jmd_c_len=None):
    """Check if general JMD-C length is given and adjust it globally."""
    global_jmd_c_len = options["jmd_c_len"]
    if global_jmd_c_len is not None:
        return global_jmd_c_len
    else:
        return jmd_c_len


# DEV: Parameters are used as directive to get better documentation style
# Enables setting of system level variables like in matplotlib
def _check_option(name_option="", option=None):
    """Check if option is valid"""
    if name_option == "verbose":
        if option != "off":
            check_verbose(verbose=option)
    if name_option == "random_state":
        if option != "off":
            check_random_state(random_state=option)
    if name_option == "allow_multiprocessing":
        check_bool(name=name_option, val=option)
    if "jmd" in name_option:
        if "len" in name_option:
            check_number_val(name=name_option, val=option, accept_none=True, just_int=True)
        if "name" in name_option:
            check_str(name=name_option, val=option, accept_none=False)
    if name_option == "ext_len":
        check_number_range(name=name_option, val=option, min_val=0, accept_none=False)
    if "df" in name_option:
        check_df(name=name_option, df=option, accept_none=False)


class Settings:
    """
     A class for managing system-level settings for AAanalysis.

    This class mimics a dictionary-like interface, allowing the setting and retrieving
    of system-level options. It is designed to be used as a single global instance, ``options``.

    Parameters
    ----------
    The following options can be set:

    verbose : bool or 'off', default='off'
        Sets verbose mode to ``True`` or ``False`` globally if not 'off'.
    random_state : int, None, or 'off', default='off'
        The seed used by the random number generator.

        * If set to a positive integer, results of stochastic processes are consistent, enabling reproducibility.
        * If set to ``None``, stochastic processes will be truly random.
        * If set to 'off', no global random state variable will be set, allowing the underlying libraries to use
          their default random state behavior.

    allow_multiprocessing : bool, default=True
        Whether multiprocessing is allowed in general. If ``False``, ``n_jobs`` is automatically set to 1.
    name_tmd : str, default='TMD'
        Name of target middle domain (TMD) used in CPP plots.
    name_jmd_n : str, default='JMD_N'
        Name of N-terminal juxta middle domain (JMD-N) used in CPP plots.
    name_jmd_c : str, default='JMD_C'
        Name of C-terminal juxta middle domain (JMD-C) used in CPP plots.
    ext_len : int, default=0
        Length of TMD-extending part (starting from C and N terminal part of TMD, >=0). Disabled (set to 0) by default.
    jmd_n_len: int, default=None
        Length of N-terminal JMD (JMD-N) in number of amino acids. If ``None``, ``jmd_n_len`` is set locally.
    jmd_c_len: int, default=None
        Length of C-terminal JMD (JMD-C) in number of amino acids. If ``None``, ``jmd_c_len`` is set locally.
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

    Warnings
    --------
    * Multiprocessing Compatibility: Enabling multiprocessing (``allow_multiprocessing=True``)
      can lead to issues in environments that don't support forking or when interfacing with
      certain libraries. If encountering errors, consider setting ``allow_multiprocessing=False``.
      Note that this may affect performance in computation-intensive operations.

    Examples
    --------
    .. include:: examples/options.rst
    """
    def __init__(self):
        self._settings: Dict[str, Any] = _dict_options.copy()

    def __getitem__(self, key: str) -> Any:
        """Retrieve a setting's value using dict-like access."""
        return self._settings.get(key, None)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a setting's value using dict-like access.
           Prevent adding new keys that are not already in the system options."""
        if key in self._settings:
            _check_option(name_option=key, option=value)
            self._settings[key] = value
        else:
            valid_options = list(_dict_options.keys())
            raise KeyError(f"'{key}' is not valid options. Valid options are: {valid_options}.")

    def __contains__(self, key: str) -> bool:
        """Check if a key is in the settings."""
        return key in self._settings

    def __str__(self) -> str:
        """Return a string representation of the settings dictionary."""
        return str(self._settings)


# Global settings instance
options = Settings()
