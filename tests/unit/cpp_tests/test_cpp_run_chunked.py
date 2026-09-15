"""This is a script to test the chunked (memory-bounded) paths of CPP.run().

``CPP.run`` offers two opt-in chunking modes: ``n_batches`` (scale axis) and
``n_sample_batches`` (sample axis). This file enforces their contract:

* peak memory with many chunks is measurably lower than the in-memory run,
* ``df_feat`` is identical to the in-memory run across >= 2 chunk sizes on both axes
  (sample axis: exact; scale axis: exact except the whole-candidate-set BH FDR p-value),
* feature ids and row order do not depend on the chunk size,
* the default call is byte-identical to an explicit ``None`` chunking call,
* chunking is rejected with a clear error under bootstrap stability annotation.

Addresses #485.
"""
import gc
import tracemalloc

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

LIST_AA = list("ACDEFGHIKLMNPQRSTVWY")
SEED = 0
N_FILTER = 20

# Memory fixture: large enough that the (n_samples x positions x n_scales) tensor dominates
# the fixed overhead, small enough to stay well under 30 s under tracemalloc on a loaded CPU.
MEM_N_SAMPLES = 100
MEM_N_SCALES = 24
MEM_N_CHUNKS = 6
# Chunked peak must be at most this fraction of the in-memory peak. Measured ratios were
# ~0.16-0.31 (see TestCPPRunChunkedPeakMemory), so 0.6 leaves a wide safety margin.
MAX_PEAK_RATIO = 0.6

# BH FDR correction spans the whole candidate set, so scale-axis batching (which applies it
# per feature batch) legitimately changes this column and only this column.
COL_FDR = "p_val_fdr_bh"
# Stats are rounded to 3 decimals; the sample-axis accumulator variance may differ from
# np.std at ULP level, so the documented tolerance is one rounding step.
SAMPLE_AXIS_ATOL = 1e-3


def _build_inputs(n_samples=24, n_scales=8, seed=SEED, tmd_len=20, jmd_len=10):
    """Seeded synthetic df_parts / labels / df_scales (balanced labels, no gaps)."""
    rng = np.random.default_rng(seed)
    seq_len = jmd_len + tmd_len + jmd_len
    seqs = ["".join(rng.choice(LIST_AA, size=seq_len)) for _ in range(n_samples)]
    labels = [1] * (n_samples // 2) + [0] * (n_samples - n_samples // 2)
    df_seq = pd.DataFrame({"entry": [f"P{i}" for i in range(n_samples)], "sequence": seqs,
                           "tmd_start": jmd_len + 1, "tmd_stop": jmd_len + tmd_len,
                           "label": labels})
    df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
    df_scales = aa.load_scales().iloc[:, :n_scales]
    return df_parts, labels, df_scales


@pytest.fixture(scope="module")
def small():
    """Small fixture for identity / ordering tests, plus its in-memory reference df_feat."""
    df_parts, labels, df_scales = _build_inputs(n_samples=24, n_scales=8)
    cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, verbose=False, random_state=SEED)
    df_ref = cpp.run(labels=labels, n_filter=N_FILTER, n_jobs=1)
    return dict(cpp=cpp, labels=labels, df_ref=df_ref, n_samples=len(df_parts),
                n_scales=df_scales.shape[1])


def _traced_peak_mb(func):
    """Peak traced allocation (MB) while running ``func`` (numpy registers with tracemalloc)."""
    gc.collect()
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    base, _ = tracemalloc.get_traced_memory()
    try:
        result = func()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()
    return result, (peak - base) / 1e6


@pytest.fixture(scope="module")
def peaks():
    """Measure peak memory once per module for the in-memory and chunked runs."""
    df_parts, labels, df_scales = _build_inputs(n_samples=MEM_N_SAMPLES, n_scales=MEM_N_SCALES)
    cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, verbose=False, random_state=SEED)
    configs = {"in_memory": {},
               "n_batches": dict(n_batches=MEM_N_CHUNKS),
               "n_sample_batches": dict(n_sample_batches=MEM_N_CHUNKS)}
    out = {}
    for name, kws in configs.items():
        df_feat, peak = _traced_peak_mb(lambda: cpp.run(labels=labels, n_filter=N_FILTER, n_jobs=1, **kws))
        out[name] = dict(df_feat=df_feat, peak=peak)
    return out


def _assert_same_except_fdr(df_ref, df_chunked):
    """Exact equality on every column except the BH FDR p-value."""
    cols = [c for c in df_ref.columns if c != COL_FDR]
    pd.testing.assert_frame_equal(df_ref[cols], df_chunked[cols], check_exact=True)


# ---------------------------------------------------------------------------
# Peak memory
# ---------------------------------------------------------------------------
# Marked slow: tracemalloc hooks every Python allocation, which inflates the three measured
# runs to ~30 s on an idle machine and 70-170 s on a contended one. The ratio itself was
# stable across repeated runs; only the wall-clock makes it unfit for the blocking unit tier.
@pytest.mark.slow
class TestCPPRunChunkedPeakMemory:
    """Chunking bounds peak memory (tracemalloc, single process, no wall-clock assertion).

    Measured on the memory fixture (100 samples x 24 scales, 40-residue sequences, default
    parts and splits, n_filter=20, n_jobs=1), macOS / Python 3.13, 6 chunks, peak traced
    allocation:

    * in-memory: ~116 MB
    * n_batches=6: ~23 MB (ratio ~0.19)
    * n_sample_batches=6: ~23 MB (ratio ~0.20)

    Other probes: 80 x 30 with 8 chunks gave ratios 0.16 / 0.18; 120 x 40 with 10 chunks gave
    0.17 / 0.18; 200 x 60 with 10 chunks gave 0.31 / 0.31. The threshold ``MAX_PEAK_RATIO = 0.6``
    is about 2x the worst measured ratio. Only many chunks are asserted: with 2 chunks the fixed
    per-pass overhead dominates and the peak was NOT reliably lower (n_sample_batches=2 measured
    0.53x to 1.29x of in-memory, above 1.0 on the larger probes).
    """

    def test_n_batches_peak_lower_than_in_memory(self, peaks):
        ratio = peaks["n_batches"]["peak"] / peaks["in_memory"]["peak"]
        assert ratio <= MAX_PEAK_RATIO, (
            f"n_batches={MEM_N_CHUNKS} peak {peaks['n_batches']['peak']:.1f} MB is {ratio:.2f}x the "
            f"in-memory peak {peaks['in_memory']['peak']:.1f} MB (threshold {MAX_PEAK_RATIO})")

    def test_n_sample_batches_peak_lower_than_in_memory(self, peaks):
        ratio = peaks["n_sample_batches"]["peak"] / peaks["in_memory"]["peak"]
        assert ratio <= MAX_PEAK_RATIO, (
            f"n_sample_batches={MEM_N_CHUNKS} peak {peaks['n_sample_batches']['peak']:.1f} MB is "
            f"{ratio:.2f}x the in-memory peak {peaks['in_memory']['peak']:.1f} MB "
            f"(threshold {MAX_PEAK_RATIO})")

    def test_in_memory_peak_is_substantial(self, peaks):
        # Guards the fixture: if the in-memory peak collapsed (e.g. the fixture got too small),
        # the ratio assertions above would stop meaning anything.
        assert peaks["in_memory"]["peak"] >= 20.0

    def test_memory_fixture_outputs_consistent(self, peaks):
        df_ref = peaks["in_memory"]["df_feat"]
        pd.testing.assert_frame_equal(df_ref, peaks["n_sample_batches"]["df_feat"], check_exact=True)
        _assert_same_except_fdr(df_ref, peaks["n_batches"]["df_feat"])


# ---------------------------------------------------------------------------
# Identity across chunk sizes
# ---------------------------------------------------------------------------
class TestCPPRunChunkedIdentity:
    """df_feat of chunked runs vs the in-memory run, across >= 2 chunk sizes per axis."""

    @pytest.mark.parametrize("n_sample_batches", [2, 3, 5, 24])
    def test_sample_axis_same_features_and_order(self, small, n_sample_batches):
        df = small["cpp"].run(labels=small["labels"], n_filter=N_FILTER, n_jobs=1,
                              n_sample_batches=n_sample_batches)
        assert df["feature"].to_list() == small["df_ref"]["feature"].to_list()
        assert list(df.columns) == list(small["df_ref"].columns)

    @pytest.mark.parametrize("n_sample_batches", [2, 3, 5, 24])
    def test_sample_axis_values_within_tolerance(self, small, n_sample_batches):
        df = small["cpp"].run(labels=small["labels"], n_filter=N_FILTER, n_jobs=1,
                              n_sample_batches=n_sample_batches)
        pd.testing.assert_frame_equal(small["df_ref"], df, check_exact=False, rtol=0,
                                      atol=SAMPLE_AXIS_ATOL)

    @pytest.mark.parametrize("n_sample_batches", [2, 5])
    def test_sample_axis_byte_identical_on_fixture(self, small, n_sample_batches):
        # Stronger than the documented tolerance: on this tie-free fixture the rounded stats
        # are byte-identical, so any drift beyond rounding surfaces here.
        df = small["cpp"].run(labels=small["labels"], n_filter=N_FILTER, n_jobs=1,
                              n_sample_batches=n_sample_batches)
        pd.testing.assert_frame_equal(small["df_ref"], df, check_exact=True)

    @pytest.mark.parametrize("n_batches", [2, 3, 8])
    def test_scale_axis_identical_except_fdr(self, small, n_batches):
        df = small["cpp"].run(labels=small["labels"], n_filter=N_FILTER, n_jobs=1, n_batches=n_batches)
        _assert_same_except_fdr(small["df_ref"], df)

    @pytest.mark.parametrize("n_batches", [2, 3, 8])
    def test_scale_axis_fdr_is_valid_pvalue(self, small, n_batches):
        df = small["cpp"].run(labels=small["labels"], n_filter=N_FILTER, n_jobs=1, n_batches=n_batches)
        fdr = df[COL_FDR].to_numpy()
        assert np.all((fdr >= 0) & (fdr <= 1))
        # The FDR-corrected p-value never undercuts the raw Mann-Whitney p-value.
        assert np.all(fdr >= df["p_val_mann_whitney"].to_numpy() - 1e-12)

    @pytest.mark.xfail(strict=True, reason=(
        "n_batches applies the BH FDR correction per feature batch, so p_val_fdr_bh differs "
        "from the in-memory run; the full df_feat is not byte-identical on the scale axis."))
    @pytest.mark.parametrize("n_batches", [2, 3])
    def test_scale_axis_fully_byte_identical(self, small, n_batches):
        df = small["cpp"].run(labels=small["labels"], n_filter=N_FILTER, n_jobs=1, n_batches=n_batches)
        pd.testing.assert_frame_equal(small["df_ref"], df, check_exact=True)

    def test_two_scale_chunk_sizes_agree_except_fdr(self, small):
        df_a = small["cpp"].run(labels=small["labels"], n_filter=N_FILTER, n_jobs=1, n_batches=2)
        df_b = small["cpp"].run(labels=small["labels"], n_filter=N_FILTER, n_jobs=1, n_batches=4)
        _assert_same_except_fdr(df_a, df_b)

    def test_two_sample_chunk_sizes_agree(self, small):
        df_a = small["cpp"].run(labels=small["labels"], n_filter=N_FILTER, n_jobs=1, n_sample_batches=2)
        df_b = small["cpp"].run(labels=small["labels"], n_filter=N_FILTER, n_jobs=1, n_sample_batches=7)
        pd.testing.assert_frame_equal(df_a, df_b, check_exact=True)


# ---------------------------------------------------------------------------
# Deterministic ordering (property-based)
# ---------------------------------------------------------------------------
class TestCPPRunChunkedOrdering:
    """Feature ids + row order are deterministic and independent of the chunk size."""

    @settings(max_examples=5, deadline=None)
    @given(n_batches=some.integers(min_value=2, max_value=8))
    def test_order_independent_of_n_batches(self, small, n_batches):
        df = small["cpp"].run(labels=small["labels"], n_filter=N_FILTER, n_jobs=1, n_batches=n_batches)
        assert df["feature"].to_list() == small["df_ref"]["feature"].to_list()

    @settings(max_examples=5, deadline=None)
    @given(n_sample_batches=some.integers(min_value=2, max_value=24))
    def test_order_independent_of_n_sample_batches(self, small, n_sample_batches):
        df = small["cpp"].run(labels=small["labels"], n_filter=N_FILTER, n_jobs=1,
                              n_sample_batches=n_sample_batches)
        assert df["feature"].to_list() == small["df_ref"]["feature"].to_list()

    @settings(max_examples=3, deadline=None)
    @given(n_chunks=some.integers(min_value=2, max_value=8),
           axis=some.sampled_from(["n_batches", "n_sample_batches"]))
    def test_repeated_chunked_run_reproducible(self, small, n_chunks, axis):
        kws = {axis: n_chunks}
        df_a = small["cpp"].run(labels=small["labels"], n_filter=N_FILTER, n_jobs=1, **kws)
        df_b = small["cpp"].run(labels=small["labels"], n_filter=N_FILTER, n_jobs=1, **kws)
        pd.testing.assert_frame_equal(df_a, df_b, check_exact=True)


# ---------------------------------------------------------------------------
# Default path + bootstrap interaction
# ---------------------------------------------------------------------------
class TestCPPRunChunkedDefaultAndBootstrap:
    """Chunking is opt-in; it is rejected (clearly) under bootstrap stability annotation."""

    def test_default_equals_explicit_none(self, small):
        df = small["cpp"].run(labels=small["labels"], n_filter=N_FILTER, n_jobs=1,
                              n_batches=None, n_sample_batches=None)
        pd.testing.assert_frame_equal(small["df_ref"], df, check_exact=True)

    def test_default_run_reproducible(self, small):
        df = small["cpp"].run(labels=small["labels"], n_filter=N_FILTER, n_jobs=1)
        pd.testing.assert_frame_equal(small["df_ref"], df, check_exact=True)

    def _bootstrap_cpp(self):
        df_parts, labels, df_scales = _build_inputs(n_samples=24, n_scales=8)
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, verbose=False, random_state=SEED,
                     bootstrap=True, bootstrap_kws=dict(rounds=2))
        return cpp, labels

    def test_bootstrap_rejects_n_batches(self):
        cpp, labels = self._bootstrap_cpp()
        with pytest.raises(ValueError, match=r"'n_batches' \(3\).*cannot be combined with bootstrap"):
            cpp.run(labels=labels, n_filter=N_FILTER, n_jobs=1, n_batches=3)

    def test_bootstrap_rejects_n_sample_batches(self):
        cpp, labels = self._bootstrap_cpp()
        with pytest.raises(ValueError, match=r"'n_sample_batches' \(4\).*cannot be combined with bootstrap"):
            cpp.run(labels=labels, n_filter=N_FILTER, n_jobs=1, n_sample_batches=4)
