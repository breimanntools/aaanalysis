"""
This is a script for adjusting terminal output.
"""
import numpy as np
import sys
from multiprocessing import Manager, Lock

STR_PROGRESS = "."

# Multiprocessing shared progress bar
manager = Manager()
GLOBAL_SHARED_MAX_PROGRESS = manager.Value('d', 0.0)
GLOBAL_SHARED_VALUE_LOCK = manager.Lock()
GLOBAL_PRINT_LOCK = manager.Lock()


# I Helper Functions
def _get_global_shared_variables(shared_max_progress=None, shared_value_lock=None, print_lock=None):
    """Use the passed-in shared objects if provided, otherwise fallback."""
    if shared_max_progress is None:
        shared_max_progress = GLOBAL_SHARED_MAX_PROGRESS
    if shared_value_lock is None:
        shared_value_lock = GLOBAL_SHARED_VALUE_LOCK
    if print_lock is None:
        print_lock = GLOBAL_PRINT_LOCK
    return shared_max_progress, shared_value_lock, print_lock


# Plotting & print functions
def _print_red(input_str, **args):
    """Prints the given string in red text."""
    print(f"\033[91m{input_str}\033[0m", **args)


def _print_blue(input_str, **args):
    """Prints the given string in blue text."""
    print(f"\033[94m{input_str}\033[0m", **args)


def _print_green(input_str, **args):
    """Prints the given string in Matrix-style green text."""
    print(f"\033[92m{input_str}\033[0m", **args)


def print_out(input_str, **args):
    """Prints the given string in Matrix-style green text."""
    _print_blue(input_str, **args)


# Progress bar
def print_start_progress(start_message=None):
    """Print start progress"""
    # Start message
    if start_message is not None:
        print_out(start_message)
    # Start progress bar
    progress_bar = " " * 25
    print_out(f"\r   |{progress_bar}| 0.0%", end="")
    sys.stdout.flush()


def print_progress(i=0, n_total=0, add_new_line=False,
                   shared_max_progress=None, shared_value_lock=None, print_lock=None):
    """
    Print progress only if new progress exceeds the current shared maximum.
    The shared objects can be passed in; if not provided, defaults defined in this module are used.
    """
    # Use the passed-in shared objects if provided, otherwise fallback.
    args = dict(shared_max_progress=shared_max_progress, shared_value_lock=shared_value_lock, print_lock=print_lock)
    shared_max_progress, shared_value_lock, print_lock = _get_global_shared_variables(**args)
    # Print progress
    progress = min(np.round(i / n_total * 100, 4), 100)
    with shared_value_lock:
        if progress > shared_max_progress.value:
            shared_max_progress.value = progress
            progress_bar = STR_PROGRESS * int(progress / 4) + " " * (25 - int(progress / 4))
            str_end = "\n" if add_new_line else ""
            with print_lock:
                print_out(f"\r   |{progress_bar}| {progress:.1f}%", end=str_end)
    sys.stdout.flush()


def print_end_progress(end_message=None, shared_max_progress=None, shared_value_lock=None, add_new_line=True):
    """Print finished progress bar"""
    # End progress bar
    progress_bar = STR_PROGRESS * 25
    str_end = "\n" if add_new_line else ""
    print_out(f"\r   |{progress_bar}| 100.0%", end=str_end)
    # End message
    if end_message is not None:
        print_out(end_message)
    sys.stdout.flush()
    # Rest shared_value
    args = dict(shared_max_progress=shared_max_progress, shared_value_lock=shared_value_lock)
    shared_max_progress, shared_value_lock, _ = _get_global_shared_variables(**args)
    with shared_value_lock:
        shared_max_progress.value = 0
