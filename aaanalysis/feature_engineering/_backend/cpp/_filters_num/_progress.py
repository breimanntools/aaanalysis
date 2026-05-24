"""
This is a script for the backend of CPP's numerical-mode pipeline, re-exporting
shared multiprocessing / progress infrastructure from ``_filters._progress``.

Single source of truth for the MP Manager lifecycle stays in ``_filters._progress``;
``_filters_num`` reuses it so seq-mode and numerical-mode share the same shared
state (and same macOS spawn-safety guarantees).
"""
from .._filters._progress import (  # noqa: F401
    _FloatBox,
    DEFAULT_SHARED_MAX_PROGRESS,
    DEFAULT_SHARED_VALUE_LOCK,
    DEFAULT_PRINT_LOCK,
    _is_main_process,
    _get_mp_shared,
    _cleanup_mp_manager,
    _resolve_shared,
    _reset_progress,
)
