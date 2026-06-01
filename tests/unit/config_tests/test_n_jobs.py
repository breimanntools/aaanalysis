"""This is a script to test the n_jobs contract: check_n_jobs, resolve_n_jobs, and options['n_jobs'].

Covers Stage-2 decision D1 — the unified parallelism contract:

* ``1`` runs serially; ``-1`` uses all cores (``os.cpu_count()``); ``N > 1`` uses
  exactly N; ``None`` is deferred (passes through ``check_n_jobs`` unchanged) and
  later resolved by ``resolve_n_jobs`` to an optimized worker count.
* ``options['n_jobs']`` is a global override (default ``'off'``): a concrete value
  wins over the per-call argument, mirroring ``verbose`` / ``random_state``.
* ``allow_multiprocessing=False`` forces serial and wins over everything.
"""
import os

import pytest

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.config import resolve_n_jobs

aa.options["verbose"] = False

_N_CPU = os.cpu_count()


@pytest.fixture(autouse=True)
def _reset_options():
    """Restore the n_jobs / allow_multiprocessing options around every test."""
    yield
    aa.options["n_jobs"] = "off"
    aa.options["allow_multiprocessing"] = True


class TestCheckNJobs:
    """Normal + negative cases for check_n_jobs (the contract normalizer)."""

    def test_none_passes_through(self):
        # None is deferred to resolve_n_jobs, so check_n_jobs keeps it as None.
        assert ut.check_n_jobs(n_jobs=None) is None

    def test_one_is_serial(self):
        assert ut.check_n_jobs(n_jobs=1) == 1

    def test_positive_int_passthrough(self):
        assert ut.check_n_jobs(n_jobs=2) == 2

    def test_minus_one_is_cpu_count(self):
        assert ut.check_n_jobs(n_jobs=-1) == _N_CPU

    def test_large_positive_int_passthrough(self):
        assert ut.check_n_jobs(n_jobs=1000) == 1000

    def test_option_override_forces_value(self):
        aa.options["n_jobs"] = 1
        assert ut.check_n_jobs(n_jobs=None) == 1
        assert ut.check_n_jobs(n_jobs=8) == 1

    def test_option_off_uses_per_call(self):
        aa.options["n_jobs"] = "off"
        assert ut.check_n_jobs(n_jobs=None) is None
        assert ut.check_n_jobs(n_jobs=3) == 3

    def test_option_minus_one_resolves_to_cpu_count(self):
        aa.options["n_jobs"] = -1
        assert ut.check_n_jobs(n_jobs=None) == _N_CPU

    def test_allow_multiprocessing_false_forces_serial(self):
        aa.options["allow_multiprocessing"] = False
        assert ut.check_n_jobs(n_jobs=-1) == 1
        assert ut.check_n_jobs(n_jobs=8) == 1

    # Negative tests
    def test_invalid_zero(self):
        with pytest.raises(ValueError):
            ut.check_n_jobs(n_jobs=0)

    def test_invalid_below_minus_one(self):
        with pytest.raises(ValueError):
            ut.check_n_jobs(n_jobs=-2)

    def test_invalid_float(self):
        with pytest.raises(ValueError):
            ut.check_n_jobs(n_jobs=2.5)

    def test_invalid_string(self):
        with pytest.raises(ValueError):
            ut.check_n_jobs(n_jobs="four")


class TestResolveNJobs:
    """Normal + edge cases for resolve_n_jobs (the None -> optimized rule)."""

    def test_none_small_work_is_one(self):
        assert resolve_n_jobs(n_jobs=None, n_work=5) == 1

    def test_none_scales_with_work(self):
        assert resolve_n_jobs(n_jobs=None, n_work=100) == min(_N_CPU, 10)

    def test_none_capped_by_cpu_count(self):
        # Very large work cannot exceed the core count.
        assert resolve_n_jobs(n_jobs=None, n_work=10 ** 9) == _N_CPU

    def test_none_no_work_defaults_to_one(self):
        assert resolve_n_jobs(n_jobs=None) == 1

    def test_none_zero_work_is_one(self):
        assert resolve_n_jobs(n_jobs=None, n_work=0) == 1

    def test_explicit_value_passthrough(self):
        assert resolve_n_jobs(n_jobs=3, n_work=100) == 3

    def test_explicit_one_passthrough(self):
        assert resolve_n_jobs(n_jobs=1, n_work=100) == 1

    def test_explicit_value_ignores_work(self):
        # An already-resolved n_jobs (from check_n_jobs) is never re-optimized.
        assert resolve_n_jobs(n_jobs=_N_CPU, n_work=5) == _N_CPU


class TestNJobsOption:
    """The options['n_jobs'] surface: validation of valid / invalid values."""

    def test_accepts_off(self):
        aa.options["n_jobs"] = "off"
        assert aa.options["n_jobs"] == "off"

    def test_accepts_positive_int(self):
        aa.options["n_jobs"] = 4
        assert aa.options["n_jobs"] == 4

    def test_accepts_minus_one(self):
        aa.options["n_jobs"] = -1
        assert aa.options["n_jobs"] == -1

    def test_rejects_zero(self):
        with pytest.raises(ValueError):
            aa.options["n_jobs"] = 0

    def test_rejects_minus_two(self):
        with pytest.raises(ValueError):
            aa.options["n_jobs"] = -2

    def test_rejects_none(self):
        # None is not a valid option value (use 'off' to defer to the per-call arg).
        with pytest.raises(ValueError):
            aa.options["n_jobs"] = None

    def test_rejects_float(self):
        with pytest.raises(ValueError):
            aa.options["n_jobs"] = 2.5

    def test_rejects_unknown_key(self):
        with pytest.raises(KeyError):
            aa.options["n_jobss"] = 2


class TestNJobsContractComplex:
    """Cross-parameter interactions of the contract."""

    def test_option_overrides_then_normalizes_minus_one(self):
        aa.options["n_jobs"] = -1
        # Option -1 -> all cores, after passing through check_n_jobs.
        assert ut.check_n_jobs(n_jobs=1) == _N_CPU

    def test_allow_multiprocessing_false_beats_option(self):
        aa.options["n_jobs"] = -1
        aa.options["allow_multiprocessing"] = False
        # allow_multiprocessing wins over the option override.
        assert ut.check_n_jobs(n_jobs=-1) == 1

    def test_end_to_end_run_with_option(self):
        # The override flows through a real CPP.run without error.
        df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
        labels = df_seq["label"].to_list()
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
        df_scales = aa.load_scales(top60_n=38).T.head(10).T
        aa.options["n_jobs"] = 1
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        df_feat = cpp.run(labels=labels, n_filter=10)
        import pandas as pd
        assert isinstance(df_feat, pd.DataFrame)

    def test_check_then_resolve_pipeline_none(self):
        # The real call order: check_n_jobs first (contract), then resolve_n_jobs.
        n = ut.check_n_jobs(n_jobs=None)
        assert n is None
        assert resolve_n_jobs(n_jobs=n, n_work=200) == min(_N_CPU, 20)

    def test_check_then_resolve_pipeline_explicit(self):
        n = ut.check_n_jobs(n_jobs=2)
        assert resolve_n_jobs(n_jobs=n, n_work=200) == 2

    def test_option_value_survives_repeated_checks(self):
        aa.options["n_jobs"] = 2
        assert ut.check_n_jobs(n_jobs=None) == 2
        assert ut.check_n_jobs(n_jobs=None) == 2
