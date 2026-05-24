"""
This is a script for the backend of CPP's shared progress / multiprocessing
infrastructure.

Canonical home of these helpers (lifted in PR6 from the now-removed
``_filters/_progress.py``). Shared across the seq-mode and numerical-mode
pipelines for a single source of truth on Manager lifecycle + shared
progress state.

Dev rules (important for macOS/Windows):
- Never create multiprocessing.Manager() (or start subprocesses) at import time.
  macOS/Windows default to "spawn" which re-imports modules in child processes.
  Import-time Manager() will crash with:
  "An attempt has been made to start a new process before the current process has finished..."
- Create multiprocessing shared objects lazily (only when needed) and only from the main process.
- Always allow passing shared_* objects explicitly to support true cross-process progress updates.
"""
import threading
import multiprocessing as mp


# I Helper Functions
# ---------------------------------------------------------------------
# Progress sharing (thread fallback + optional multiprocessing shared)
# ---------------------------------------------------------------------
class _FloatBox:
    """Thread-safe fallback for multiprocessing.Value('d', x) using a `.value` attribute."""
    def __init__(self, v: float = 0.0):
        self.value = v


# Default safe globals (NO multiprocessing side effects at import time)
DEFAULT_SHARED_MAX_PROGRESS = _FloatBox(0.0)
DEFAULT_SHARED_VALUE_LOCK = threading.Lock()
DEFAULT_PRINT_LOCK = threading.Lock()

# Lazy-created manager/shared objects (only for true cross-process sync)
_MP_MANAGER = None
_MP_SHARED_MAX_PROGRESS = None
_MP_SHARED_VALUE_LOCK = None
_MP_PRINT_LOCK = None
_MP_MANAGER_REFCOUNT = 0  # Track usage to enable cleanup


def _is_main_process() -> bool:
    """True only in the original interpreter process (important for macOS spawn safety)."""
    return mp.current_process().name == "MainProcess"


# II Main Functions
def _get_mp_shared():
    """
    Lazily create a multiprocessing.Manager + shared objects.

    IMPORTANT:
    - Must only be called from the main process.
    - Never called at import time.
    """
    global _MP_MANAGER, _MP_SHARED_MAX_PROGRESS, _MP_SHARED_VALUE_LOCK, _MP_PRINT_LOCK, _MP_MANAGER_REFCOUNT

    if not _is_main_process():
        # In workers spawned via "spawn", we must NOT try to start a Manager here.
        return None

    if _MP_MANAGER is None:
        _MP_MANAGER = mp.Manager()
        _MP_SHARED_MAX_PROGRESS = _MP_MANAGER.Value("d", 0.0)
        _MP_SHARED_VALUE_LOCK = _MP_MANAGER.Lock()
        _MP_PRINT_LOCK = _MP_MANAGER.Lock()
        _MP_MANAGER_REFCOUNT = 0

    _MP_MANAGER_REFCOUNT += 1
    return _MP_SHARED_MAX_PROGRESS, _MP_SHARED_VALUE_LOCK, _MP_PRINT_LOCK


def _cleanup_mp_manager():
    """
    Cleanup multiprocessing Manager if no longer needed.
    Should be called after parallel operations complete.
    """
    global _MP_MANAGER, _MP_SHARED_MAX_PROGRESS, _MP_SHARED_VALUE_LOCK, _MP_PRINT_LOCK, _MP_MANAGER_REFCOUNT

    if not _is_main_process():
        return

    if _MP_MANAGER is not None and _MP_MANAGER_REFCOUNT > 0:
        _MP_MANAGER_REFCOUNT -= 1
        # Only shutdown if refcount reaches 0 (all operations done)
        if _MP_MANAGER_REFCOUNT == 0:
            try:
                _MP_MANAGER.shutdown()
            except Exception:
                pass  # Ignore errors during cleanup
            _MP_MANAGER = None
            _MP_SHARED_MAX_PROGRESS = None
            _MP_SHARED_VALUE_LOCK = None
            _MP_PRINT_LOCK = None


def _resolve_shared(shared_max_progress=None, shared_value_lock=None, print_lock=None, prefer_multiprocessing=False):
    """
    Resolve shared objects for progress printing.

    Priority:
    1) If caller passes shared_* explicitly: use them.
    2) Else if prefer_multiprocessing=True and we are in main process: create/use Manager-based shared objects.
    3) Else: use thread-safe defaults.

    This design:
    - Avoids macOS spawn crashes (no Manager at import time, no Manager creation in workers).
    - Allows true cross-process shared progress (Manager) when requested.
    """
    if shared_max_progress is not None and shared_value_lock is not None and print_lock is not None:
        return shared_max_progress, shared_value_lock, print_lock

    if prefer_multiprocessing:
        mp_shared = _get_mp_shared()
        if mp_shared is not None:
            return mp_shared

    return DEFAULT_SHARED_MAX_PROGRESS, DEFAULT_SHARED_VALUE_LOCK, DEFAULT_PRINT_LOCK


def _reset_progress(shared_max_progress, shared_value_lock):
    """Reset shared progress to 0 in a safe way."""
    with shared_value_lock:
        shared_max_progress.value = 0.0
