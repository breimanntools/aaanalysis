"""
This is a script for the backend of CPP's shared progress / multiprocessing
infrastructure.

Shared across the seq-mode and numerical-mode pipelines — single source of
truth on Manager lifecycle + shared progress state.

Dev rules (important for macOS/Windows):
- Never create multiprocessing.Manager() (or start subprocesses) at import time.
  macOS/Windows default to "spawn" which re-imports modules in child processes.
  Import-time Manager() will crash with:
  "An attempt has been made to start a new process before the current process has finished..."
- Create multiprocessing shared objects lazily (only when needed) and only from the main process.
- Always allow passing shared_* objects explicitly to support true cross-process progress updates.

Graceful-degradation contract:
- The multiprocessing.Manager only backs a *cosmetic* cross-process progress bar.
  In some non-interactive contexts (``python -c``, heredocs, certain subprocess
  shells) the Manager's spawn/pipe handshake fails with EOFError / OSError.
  Manager creation is therefore best-effort: on failure ``_get_mp_shared()``
  returns None (leaving all module globals untouched) and ``_resolve_shared()``
  falls back to the thread-safe ``DEFAULT_SHARED_*`` objects, so the run still
  completes single-process instead of aborting. When the Manager succeeds,
  behavior is unchanged.
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

    Returns a ``(shared_max_progress, shared_value_lock, print_lock)`` triple, or
    None when Manager-backed shared state is unavailable. None is returned when:
    - called from a spawned worker (must never start a Manager there), or
    - the Manager could not be created (best-effort; see below).

    IMPORTANT:
    - Must only be called from the main process.
    - Never called at import time.

    Best-effort Manager creation: a ``multiprocessing.Manager`` spawns a helper
    process and talks to it over a pipe. In some non-interactive contexts
    (``python -c``, heredocs, certain subprocess shells) that handshake fails with
    EOFError / OSError. Since the Manager only backs a cosmetic cross-process
    progress bar, such a failure is non-fatal: we return None (leaving every module
    global untouched and the refcount unbumped) and the caller degrades to the
    thread-safe ``DEFAULT_SHARED_*`` objects via ``_resolve_shared()``.
    """
    global _MP_MANAGER, _MP_SHARED_MAX_PROGRESS, _MP_SHARED_VALUE_LOCK, _MP_PRINT_LOCK, _MP_MANAGER_REFCOUNT

    if not _is_main_process():
        # In workers spawned via "spawn", we must NOT try to start a Manager here.
        return None

    if _MP_MANAGER is None:
        # Build into locals first, commit to globals only on full success, so a
        # partial failure never leaves half-initialized globals or a bumped
        # refcount. The broad ``except`` is intentional and scoped to just this
        # creation block: the Manager only powers a cosmetic progress bar, so any
        # spawn/handshake failure (EOFError / OSError and friends) must degrade to
        # the thread-safe defaults instead of aborting the run.
        manager = None
        try:
            manager = mp.Manager()
            shared_max_progress = manager.Value("d", 0.0)
            shared_value_lock = manager.Lock()
            print_lock = manager.Lock()
        except Exception:
            # Shut down a half-started Manager process (if any) so it does not leak,
            # then signal "no shared state" -> caller uses DEFAULT_SHARED_*.
            if manager is not None:
                try:
                    manager.shutdown()
                except Exception:
                    pass
            return None
        _MP_MANAGER = manager
        _MP_SHARED_MAX_PROGRESS = shared_max_progress
        _MP_SHARED_VALUE_LOCK = shared_value_lock
        _MP_PRINT_LOCK = print_lock
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


def _worker_shared(shared_max_progress, shared_value_lock, print_lock):
    """
    Adapt resolved progress objects for handing to spawned *process* workers.

    joblib's process backends (loky) pickle whatever is captured in each
    ``delayed(...)`` call. Manager-backed proxies pickle fine and stay shared
    across processes, so they are returned unchanged (unified progress bar,
    byte-identical to the Manager-available path). The thread-safe
    ``DEFAULT_SHARED_*`` fallbacks (used when the Manager could not be created;
    see ``_get_mp_shared``) are NOT picklable to spawned processes: a
    ``threading.Lock`` cannot cross the process boundary. In that degraded case
    return ``(None, None, None)`` so each worker self-resolves its own
    process-local defaults instead of aborting the whole run with a
    ``PicklingError``. Progress then reports per-worker (cosmetic degradation),
    but the computation still completes.

    Only needed at process-backed dispatch sites; threading-backed parallelism
    shares the objects in-process and does not pickle them.
    """
    is_thread_default = (
        shared_max_progress is DEFAULT_SHARED_MAX_PROGRESS
        or shared_value_lock is DEFAULT_SHARED_VALUE_LOCK
        or print_lock is DEFAULT_PRINT_LOCK
    )
    if is_thread_default:
        return None, None, None
    return shared_max_progress, shared_value_lock, print_lock


def _reset_progress(shared_max_progress, shared_value_lock):
    """Reset shared progress to 0 in a safe way."""
    with shared_value_lock:
        shared_max_progress.value = 0.0
