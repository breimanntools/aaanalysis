"""
This is a script for adjusting terminal output.
"""
import numpy as np
import sys

STR_PROGRESS = "."


# I Helper Functions
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
    #print(f"\033[32m{input_str}\033[0m", **args)


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


def print_progress(i=0, n=0, add_new_line=False):
    """Print progress"""
    progress = min(np.round(i/n * 100, 4), 100)
    progress_bar = STR_PROGRESS * int(progress/4) + " " * (25-int(progress/4))
    str_end = "\n" if add_new_line else ""
    print_out(f"\r   |{progress_bar}| {progress:.1f}%", end=str_end)
    sys.stdout.flush()


def print_end_progress(end_message=None):
    """Print finished progress bar"""
    # End progress bar
    progress_bar = STR_PROGRESS * 25
    print_out(f"\r   |{progress_bar}| 100.0%")
    # End message
    if end_message is not None:
        print_out(end_message)
    sys.stdout.flush()
