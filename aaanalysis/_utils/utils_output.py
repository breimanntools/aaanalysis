"""
This is a script for adjusting terminal output.

DEV NOTES (cross-platform rules: macOS/Windows + Linux)
- Never start multiprocessing (e.g., Manager()) at import time.
  Reason: on macOS/Windows the default start method is "spawn", which re-imports
  the main module in child processes. Import-time Manager() causes recursion /
  bootstrapping errors (RuntimeError/EOFError).
- If multiprocessing-shared progress is needed, create the Manager in the main
  process and pass the shared objects into workers.
- Importing this module must be side-effect free (fast, deterministic).
"""

import logging
import numpy as np
import sys
import threading

STR_PROGRESS = "."


# -----------------------------------------------------------------------------
# Safe defaults (single-process / single-thread)
# -----------------------------------------------------------------------------
class _FloatBox:
    """Simple `.value` holder to mimic multiprocessing.Value."""
    def __init__(self, v: float = 0.0):
        self.value = v


GLOBAL_SHARED_MAX_PROGRESS = _FloatBox(0.0)
GLOBAL_SHARED_VALUE_LOCK = threading.Lock()
GLOBAL_PRINT_LOCK = threading.Lock()


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


# Logging
# The whole library routes user-facing output through a single named logger,
# ``logging.getLogger("aaanalysis")``, so power users can attach handlers, capture
# output in pytest's ``caplog``, redirect it to a file, or raise/lower verbosity with
# ``setLevel(...)``. ``logging`` is imported here only (not scattered across modules):
# other code always goes through ``ut.print_out`` / ``ut.set_logger_verbosity``.
class _StdoutHandler(logging.StreamHandler):
    """A ``StreamHandler`` that always writes to the *current* ``sys.stdout``.

    The previous ``print_out`` used ``print()``, which resolves ``sys.stdout`` on every
    call, so output followed pytest's ``capsys``, ``contextlib.redirect_stdout`` and
    notebook stream swaps. Binding a handler stream once (at import) would break that,
    so the stream is re-resolved per emit, keeping the shim byte-for-byte compatible
    with the old ``print``-based behaviour and with output-capture tooling.
    """

    def __init__(self):
        super().__init__(stream=sys.stdout)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        # Ignore assignment (from the base __init__/setStream); always use current stdout.
        pass


logger = logging.getLogger("aaanalysis")


def _configure_logger():
    """Attach the package's stdout handler exactly once (idempotent).

    A single ``StreamHandler`` reproduces the previous blue-coloured on-screen output via
    a formatter, so default behaviour is visually unchanged. ``logging.basicConfig`` is
    deliberately not called (it would touch the root logger and impose a format on the
    host application). ``propagate`` stays ``True`` so records reach ancestor handlers such
    as pytest's ``caplog``.
    """
    already = any(getattr(h, "_aaanalysis_stdout_handler", False) for h in logger.handlers)
    if not already:
        handler = _StdoutHandler()
        handler.setFormatter(logging.Formatter("\033[94m%(message)s\033[0m"))
        handler._aaanalysis_stdout_handler = True
        logger.addHandler(handler)
    # Default to INFO so print_out is visible out of the box (matches the previous
    # always-print behaviour); check_verbose(...) maps the resolved verbose flag onto the
    # level from there. propagate=True keeps caplog working.
    logger.setLevel(logging.INFO)
    logger.propagate = True


_configure_logger()


def set_logger_verbosity(verbose):
    """Map a resolved verbosity flag onto the ``aaanalysis`` logger level.

    ``verbose=True`` -> ``INFO`` (``print_out`` messages are emitted); ``verbose=False``
    -> ``WARNING`` (they are suppressed). Called from ``config.check_verbose`` after the
    ``options['verbose']`` global override is resolved, so the logger level always reflects
    the effective verbosity. Users may still call
    ``logging.getLogger("aaanalysis").setLevel(...)`` directly.
    """
    logger.setLevel(logging.INFO if verbose else logging.WARNING)


def print_out(input_str, **args):
    """Emit a user-facing message through the ``aaanalysis`` logger at INFO level.

    This is the sanctioned, permanent public output channel for the library: internal code
    calls ``print_out(...)`` (re-exported as ``ut.print_out``) and never ``print()``. It
    is a thin shim over ``logging.getLogger("aaanalysis").info(...)`` so output can be
    captured, redirected, or silenced through the standard ``logging`` machinery while the
    default stdout handler keeps the previous blue-coloured on-screen appearance.

    Extra keyword arguments are accepted (and ignored) only to keep the historical signature
    stable; they no longer map onto ``print`` kwargs such as ``end=`` (the live progress bar
    that needed ``end=""`` writes to stdout directly, see the progress helpers below).
    """
    logger.info(input_str)


# Progress bar
# Carve-out: the live progress bar redraws one line in place with a leading "\r" and no
# newline (end=""), which the line-oriented logger cannot express. These helpers therefore
# write to stdout directly via _print_blue (byte-identical to the old print_out path, which
# already delegated to _print_blue). print_out itself never calls print().
def print_start_progress(start_message=None):
    """Print start progress"""
    if start_message is not None:
        _print_blue(start_message)
    progress_bar = " " * 25
    _print_blue(f"\r   |{progress_bar}| 0.0%", end="")
    sys.stdout.flush()


def print_progress(i=0, n_total=0, add_new_line=False,
                   shared_max_progress=None, shared_value_lock=None, print_lock=None):
    """
    Print progress only if new progress exceeds the current shared maximum.
    The shared objects can be passed in; if not provided, defaults defined in this module are used.

    Multiprocessing (mac+linux safe):
    - Create shared objects in MAIN and pass them in:
        from multiprocessing import Manager
        m = Manager()
        shared_max = m.Value('d', 0.0)
        shared_lock = m.Lock()
        print_lock = m.Lock()
        ... pass these to print_progress(...)
    """
    if n_total <= 0:
        return

    args = dict(shared_max_progress=shared_max_progress, shared_value_lock=shared_value_lock, print_lock=print_lock)
    shared_max_progress, shared_value_lock, print_lock = _get_global_shared_variables(**args)

    progress = min(np.round(i / n_total * 100, 4), 100)

    with shared_value_lock:
        if progress > shared_max_progress.value:
            shared_max_progress.value = progress
            progress_bar = STR_PROGRESS * int(progress / 4) + " " * (25 - int(progress / 4))
            str_end = "\n" if add_new_line else ""
            with print_lock:
                _print_blue(f"\r   |{progress_bar}| {progress:.1f}%", end=str_end)
    sys.stdout.flush()


def print_end_progress(end_message=None, shared_max_progress=None, shared_value_lock=None, add_new_line=True):
    """Print finished progress bar"""
    progress_bar = STR_PROGRESS * 25
    str_end = "\n" if add_new_line else ""
    _print_blue(f"\r   |{progress_bar}| 100.0%", end=str_end)

    if end_message is not None:
        _print_blue(end_message)
    sys.stdout.flush()

    args = dict(shared_max_progress=shared_max_progress, shared_value_lock=shared_value_lock)
    shared_max_progress, shared_value_lock, _ = _get_global_shared_variables(**args)
    with shared_value_lock:
        shared_max_progress.value = 0.0
