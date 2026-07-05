"""This is a script to test the shared-progress / multiprocessing backend of CPP
(``_backend/cpp/_filters/_progress.py``): the lazy Manager creation, its
graceful degradation to the thread-safe defaults when the Manager cannot be
spawned (issue #339 — EOFError in ``python -c`` / heredoc / subprocess shells),
and the resolve/cleanup helpers.

Driven directly at the backend because the Manager-failure arm is not reachable
from the CPP frontend (it depends on the interpreter's process-launch context).
"""
import multiprocessing as mp

import pytest

import aaanalysis as aa
import aaanalysis.feature_engineering._backend.cpp._filters._progress as prog

aa.options["verbose"] = False


# I Helper Functions
def _reset_module_globals():
    """Restore the module's lazy-Manager globals to their pristine (import) state."""
    # Best-effort shutdown of any live Manager left by a previous test.
    if prog._MP_MANAGER is not None:
        try:
            prog._MP_MANAGER.shutdown()
        except Exception:
            pass
    prog._MP_MANAGER = None
    prog._MP_SHARED_MAX_PROGRESS = None
    prog._MP_SHARED_VALUE_LOCK = None
    prog._MP_PRINT_LOCK = None
    prog._MP_MANAGER_REFCOUNT = 0


@pytest.fixture(autouse=True)
def _clean_progress_globals():
    """Ensure each test starts and ends with pristine module globals."""
    _reset_module_globals()
    yield
    _reset_module_globals()


# II Test Classes
class TestGetMpShared:
    """The lazy Manager creation and its graceful-degradation arm (issue #339)."""

    def test_happy_path_returns_triple(self):
        # When the Manager can be created, a 3-tuple of shared objects is returned
        # and the module globals are populated (byte-identical to previous behavior).
        result = prog._get_mp_shared()
        assert isinstance(result, tuple)
        assert len(result) == 3
        shared_max_progress, shared_value_lock, print_lock = result
        assert shared_max_progress is not None
        assert prog._MP_MANAGER is not None
        assert prog._MP_MANAGER_REFCOUNT == 1
        # The returned triple mirrors the stored globals.
        assert shared_max_progress is prog._MP_SHARED_MAX_PROGRESS
        assert shared_value_lock is prog._MP_SHARED_VALUE_LOCK
        assert print_lock is prog._MP_PRINT_LOCK
        prog._cleanup_mp_manager()

    def test_manager_eoferror_returns_none(self, monkeypatch):
        # Simulate the issue-#339 failure: Manager() raises EOFError. The function
        # must return None WITHOUT raising.
        def _raise_eof(*args, **kwargs):
            raise EOFError("simulated Manager pipe EOF")

        monkeypatch.setattr(prog.mp, "Manager", _raise_eof)
        assert prog._get_mp_shared() is None

    def test_manager_oserror_returns_none(self, monkeypatch):
        # The other realistic failure class (OSError) also degrades, not aborts.
        def _raise_os(*args, **kwargs):
            raise OSError("simulated Manager spawn failure")

        monkeypatch.setattr(prog.mp, "Manager", _raise_os)
        assert prog._get_mp_shared() is None

    def test_failure_leaves_globals_clean(self, monkeypatch):
        # A failed attempt must not partially-initialize globals or bump the refcount.
        def _raise_eof(*args, **kwargs):
            raise EOFError("simulated Manager pipe EOF")

        monkeypatch.setattr(prog.mp, "Manager", _raise_eof)
        prog._get_mp_shared()
        assert prog._MP_MANAGER is None
        assert prog._MP_SHARED_MAX_PROGRESS is None
        assert prog._MP_SHARED_VALUE_LOCK is None
        assert prog._MP_PRINT_LOCK is None
        assert prog._MP_MANAGER_REFCOUNT == 0

    def test_failure_then_cleanup_is_safe(self, monkeypatch):
        # Cleanup after a never-created Manager must be a no-op (guards on None).
        def _raise_eof(*args, **kwargs):
            raise EOFError("simulated Manager pipe EOF")

        monkeypatch.setattr(prog.mp, "Manager", _raise_eof)
        assert prog._get_mp_shared() is None
        # Must not raise even though no Manager was ever created.
        prog._cleanup_mp_manager()
        assert prog._MP_MANAGER is None

    def test_partial_allocation_failure_returns_none(self, monkeypatch):
        # Manager() succeeds but a later .Value/.Lock allocation fails: still None,
        # globals stay clean, and the half-started Manager is shut down (no leak).
        class _FakeManager:
            def __init__(self):
                self.shutdown_called = False

            def Value(self, *args, **kwargs):
                raise OSError("simulated allocation failure")

            def Lock(self):  # pragma: no cover - not reached (Value fails first)
                raise AssertionError("Lock should not be reached")

            def shutdown(self):
                self.shutdown_called = True

        created = {}

        def _fake_manager_factory(*args, **kwargs):
            m = _FakeManager()
            created["m"] = m
            return m

        monkeypatch.setattr(prog.mp, "Manager", _fake_manager_factory)
        assert prog._get_mp_shared() is None
        assert prog._MP_MANAGER is None
        assert prog._MP_MANAGER_REFCOUNT == 0
        assert created["m"].shutdown_called is True


class TestResolveShared:
    """``_resolve_shared`` priority: explicit > multiprocessing > thread defaults."""

    def test_explicit_objects_passthrough(self):
        # Explicitly passed shared objects are returned unchanged.
        smp, svl, pl = object(), object(), object()
        out = prog._resolve_shared(shared_max_progress=smp, shared_value_lock=svl,
                                   print_lock=pl, prefer_multiprocessing=True)
        assert out == (smp, svl, pl)

    def test_prefer_multiprocessing_uses_manager(self):
        # prefer_multiprocessing=True in the main process returns the Manager triple.
        out = prog._resolve_shared(prefer_multiprocessing=True)
        assert isinstance(out, tuple) and len(out) == 3
        assert out == (prog._MP_SHARED_MAX_PROGRESS, prog._MP_SHARED_VALUE_LOCK, prog._MP_PRINT_LOCK)
        prog._cleanup_mp_manager()

    def test_prefer_multiprocessing_falls_back_to_defaults_on_failure(self, monkeypatch):
        # When the Manager cannot be created, _resolve_shared returns the
        # thread-safe DEFAULT_SHARED_* triple (the core degradation contract).
        def _raise_eof(*args, **kwargs):
            raise EOFError("simulated Manager pipe EOF")

        monkeypatch.setattr(prog.mp, "Manager", _raise_eof)
        out = prog._resolve_shared(prefer_multiprocessing=True)
        assert out == (prog.DEFAULT_SHARED_MAX_PROGRESS,
                       prog.DEFAULT_SHARED_VALUE_LOCK,
                       prog.DEFAULT_PRINT_LOCK)

    def test_default_when_not_preferring_multiprocessing(self):
        # Without prefer_multiprocessing, the thread-safe defaults are used.
        out = prog._resolve_shared(prefer_multiprocessing=False)
        assert out == (prog.DEFAULT_SHARED_MAX_PROGRESS,
                       prog.DEFAULT_SHARED_VALUE_LOCK,
                       prog.DEFAULT_PRINT_LOCK)


class TestParallelRunDegrades:
    """End-to-end: CPP.run(n_jobs=2) still completes when the Manager fails."""

    def test_cpp_run_completes_when_manager_unavailable(self, monkeypatch):
        # Simulate the non-notebook context: force the Manager to raise, then run
        # the parallel CPP path. It must complete (degrading to thread-safe
        # progress) instead of crashing with EOFError.
        import warnings

        def _raise_eof(*args, **kwargs):
            raise EOFError("simulated Manager pipe EOF")

        monkeypatch.setattr(prog.mp, "Manager", _raise_eof)
        df_seq = aa.load_dataset(name="DOM_GSEC", n=8)
        labels = df_seq["label"].to_list()
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
        df_scales = aa.load_scales(top60_n=20).T.head(8).T
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_feat = aa.CPP(df_parts=df_parts, df_scales=df_scales,
                             verbose=True).run(labels=labels, n_filter=10, n_jobs=2)
        assert len(df_feat) == 10
