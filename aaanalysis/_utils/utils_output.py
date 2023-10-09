"""
This is a script for adjusting terminal output.
"""
import numpy as np


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

def print_out(input_str, **args):
    """Prints the given string in Matrix-style green text."""
    _print_blue(input_str, **args)

# Progress bar
def print_start_progress():
    """Print start progress"""
    progress_bar = " " * 25
    print_out(f"\r   |{progress_bar}| 0.00%", end="")


def print_progress(i=0, n=0):
    """Print progress"""
    progress = min(np.round(i/n * 100, 2), 100)
    progress_bar = "#" * int(progress/4) + " " * (25-int(progress/4))
    print_out(f"\r   |{progress_bar}| {progress:.2f}%", end="")


def print_finished_progress():
    """Print finished progress bar"""
    progress_bar = "#" * 25
    print_out(f"\r   |{progress_bar}| 100.00%")

