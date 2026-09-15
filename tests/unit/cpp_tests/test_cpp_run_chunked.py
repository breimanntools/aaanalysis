"""This is a script to test the chunked (memory-bounded) paths of CPP.run().

``CPP.run`` offers two opt-in chunking modes: ``n_batches`` (scale axis) and
``n_sample_batches`` (sample axis). This file enforces their contract:

* peak RSS is driven by the *batch size* rather than by the total sample count, measured in a
  fresh subprocess per configuration across three total-input sizes at a constant effective batch
  size (``TestCPPRunChunkedPeakMemory`` carries the numbers and states what is *not* bounded),
* the sample-batched path never assigns more than one batch of sequences at a time (asserted
  cheaply, without measuring memory, in ``TestCPPRunChunkedBatchSize``),
* ``df_feat`` is identical to the in-memory run across >= 2 chunk sizes on both axes
  (sample axis: exact; scale axis: exact except the whole-candidate-set BH FDR p-value),
* hand-computed statistics are reproduced on both chunked axes,
* feature ids and row order do not depend on the chunk size,
* the default call is byte-identical to an explicit ``None`` chunking call.

Addresses #485.
"""
import json
import os
import subprocess
import sys

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

# Memory fixture: large enough that the (batch x positions x n_scales) tensor dominates the fixed
# interpreter/import footprint, small enough that one configuration runs in ~2 s.
MEM_N_SCALES = 24
# Total sample counts swept at a *constant* effective batch size, so a rising peak can only come
# from a term that scales with the total input rather than with the batch.
MEM_SIZES = (100, 200, 400)
MEM_BATCH_SIZE = 17
# Chunk count for the two comparison configurations (scale axis, and sample axis at a fixed count).
MEM_N_CHUNKS = 6
# Chunked peak-RSS growth must be at most this fraction of the in-memory growth at the same total
# size. Measured ratios were 0.16-0.23, so 0.5 is >2x the worst measurement.
MAX_PEAK_RSS_RATIO = 0.5
# Quadrupling the input at a constant batch size must cost far less than 4x the memory. Measured
# 1.85x; 3.0 keeps a wide margin while still failing if the batch stopped bounding the tensor.
MAX_GROWTH_FACTOR_OVER_4X_INPUT = 3.0
# At the largest size, holding the batch *size* constant must beat holding the batch *count*
# constant (where the batch grows with n). Measured ratio 0.39.
MAX_FIXED_BATCH_VS_FIXED_COUNT_RATIO = 0.7
# Fixture guard: if the in-memory run stopped allocating a big tensor, the ratios would be noise.
MIN_IN_MEMORY_GROWTH_MB = 60.0

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


def _assert_same_except_fdr(df_ref, df_chunked):
    """Exact equality on every column except the BH FDR p-value."""
    cols = [c for c in df_ref.columns if c != COL_FDR]
    pd.testing.assert_frame_equal(df_ref[cols], df_chunked[cols], check_exact=True)


# ---------------------------------------------------------------------------
# Peak-RSS measurement (one fresh subprocess per configuration)
# ---------------------------------------------------------------------------
def _sample_batches_for(n_samples):
    """Chunk count that puts ``MEM_BATCH_SIZE`` samples in every batch of this total size."""
    return -(-n_samples // MEM_BATCH_SIZE)


# name -> (total samples, run kwargs). The ``fixed_batch_*`` entries hold the effective batch size
# constant at MEM_BATCH_SIZE while the total input grows; ``fixed_chunks_*`` holds the batch
# *count* constant instead, so its batch grows with n and it serves as the contrast.
CONFIGS = {}
for _n in MEM_SIZES:
    CONFIGS["in_memory_%d" % _n] = (_n, {})
    CONFIGS["fixed_batch_%d" % _n] = (_n, dict(n_sample_batches=_sample_batches_for(_n)))
CONFIGS["fixed_chunks_%d" % MEM_SIZES[-1]] = (MEM_SIZES[-1], dict(n_sample_batches=MEM_N_CHUNKS))
CONFIGS["scale_batched_%d" % MEM_SIZES[0]] = (MEM_SIZES[0], dict(n_batches=MEM_N_CHUNKS))


def _peak_rss_bytes():
    """Peak resident set size of this process (bytes on macOS, KB on Linux -> normalised)."""
    import resource
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == "darwin" else rss * 1024.0


def _measure_in_this_process(name):
    """Build the memory fixture, then report the peak-RSS growth (MB) caused by ``CPP.run``."""
    import gc
    n_samples, kws = CONFIGS[name]
    df_parts, labels, df_scales = _build_inputs(n_samples=n_samples, n_scales=MEM_N_SCALES)
    cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, verbose=False, random_state=SEED)
    gc.collect()
    base = _peak_rss_bytes()  # after imports and fixture construction
    cpp.run(labels=labels, n_filter=N_FILTER, n_jobs=1, **kws)
    return dict(name=name, n_samples=n_samples, base_mb=base / 1e6,
                peak_mb=_peak_rss_bytes() / 1e6, growth_mb=(_peak_rss_bytes() - base) / 1e6)


def _measure_in_subprocess(name):
    """Run ``_measure_in_this_process(name)`` in a fresh interpreter and parse its JSON line."""
    env = dict(os.environ)
    env.update({"PYTHONPATH": os.pathsep.join([os.path.dirname(os.path.dirname(aa.__file__)),
                                               env.get("PYTHONPATH", "")]).rstrip(os.pathsep),
                "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1"})
    out = subprocess.run([sys.executable, os.path.abspath(__file__), name], check=True,
                         capture_output=True, text=True, env=env, timeout=900)
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def peaks():
    """Peak-RSS growth per configuration, each measured in its own fresh subprocess."""
    return {name: _measure_in_subprocess(name) for name in CONFIGS}


# ---------------------------------------------------------------------------
# Peak memory
# ---------------------------------------------------------------------------
def _growth(peaks, name):
    """Peak-RSS growth in MB attributable to ``CPP.run`` in the named configuration."""
    return peaks[name]["growth_mb"]


# Marked slow: eight fresh interpreters, each importing aaanalysis, building the memory fixture and
# running the full CPP pipeline (~25 s in total on an idle machine, more on a contended one). It is
# also the one class here that shells out, so it stays opt-in rather than running in the blocking
# unit tier; TestCPPRunChunkedBatchSize keeps a cheap guard on the same mechanism in the fast tier.
@pytest.mark.slow
class TestCPPRunChunkedPeakMemory:
    """What sample batching does and does not bound, measured across three total-input sizes.

    Each subprocess reports ``ru_maxrss`` (bytes on macOS, KB on Linux) right before ``CPP.run``
    (after imports and fixture construction) and again afterwards; the difference is the growth
    attributable to the run. Measuring one configuration per process removes the order dependence
    that a single-process measurement has (the first run warms the instance cache).

    Measured on the memory fixture (24 scales, 40-residue sequences, default parts and splits,
    ``n_filter=20``, ``n_jobs=1``), macOS / Python 3.13. 23760 candidate features and 1188
    pre-filter survivors at every size, i.e. the survivor count does not depend on ``n``:

    ====================  =============  ============  ==================
    configuration         total samples  batch size    peak-RSS growth
    ====================  =============  ============  ==================
    in-memory                       100           100          157.7 MB
    in-memory                       200           200          292.1 MB
    in-memory                       400           400          279.6 MB
    ``n_sample_batches``            100            17           34.8 MB
    ``n_sample_batches``            200            17           45.5 MB
    ``n_sample_batches``            400            17           64.5 MB
    ``n_sample_batches``            400            67          164.1 MB
    ``n_batches=6``                 100           100           29.6 MB
    ====================  =============  ============  ==================

    Reading the middle block (batch size held at 17 while the input quadruples): growth fits
    ``25 MB + 0.099 MB * n`` almost exactly. So sample batching bounds the *dominant* term, the
    per-batch ``(batch_size x positions x n_scales)`` scale-value tensor, but peak memory is **not**
    independent of ``n``: the ``(n_samples, n_pre_filter)`` survivor matrix and the vectorized
    Mann-Whitney temporaries computed on it stay resident, and account for that residual ~0.1 MB
    per sample. The single-pass run costs ~1.34 MB per sample over the same range, so the slope is
    about 13x flatter, which is the real (and now documented) promise of the parameter.

    The assertions below therefore bound the slope rather than claim a constant peak, and
    ``test_fixed_batch_size_growth_still_rises_with_n`` pins the residual term so the docstring
    stays honest. If the survivor matrix and its statistics are ever streamed too, relax that test
    (and the ``CPP.run`` docstring) rather than deleting it.
    """

    @pytest.mark.parametrize("n_samples", MEM_SIZES)
    def test_fixed_batch_size_peak_rss_below_in_memory(self, peaks, n_samples):
        # The headline contract, asserted at every total size: at a constant batch size the
        # sample-batched run costs a small fraction of the single-pass run on the same input.
        chunked, in_memory = _growth(peaks, "fixed_batch_%d" % n_samples), _growth(peaks, "in_memory_%d" % n_samples)
        ratio = chunked / in_memory
        assert ratio <= MAX_PEAK_RSS_RATIO, (
            f"at n={n_samples} the batch-size-{MEM_BATCH_SIZE} run grew {chunked:.1f} MB, "
            f"{ratio:.2f}x the in-memory growth {in_memory:.1f} MB "
            f"(threshold {MAX_PEAK_RSS_RATIO})")

    def test_fixed_batch_size_growth_is_sub_proportional_to_input(self, peaks):
        # 4x the samples at the same batch size must cost far less than 4x the memory.
        small, large = _growth(peaks, "fixed_batch_%d" % MEM_SIZES[0]), _growth(peaks, "fixed_batch_%d" % MEM_SIZES[-1])
        factor = large / small
        assert factor <= MAX_GROWTH_FACTOR_OVER_4X_INPUT, (
            f"{MEM_SIZES[-1]}/{MEM_SIZES[0]} = 4x the input at a constant batch size grew memory "
            f"{factor:.2f}x ({small:.1f} -> {large:.1f} MB), above the documented "
            f"{MAX_GROWTH_FACTOR_OVER_4X_INPUT}x")

    def test_fixed_batch_size_growth_still_rises_with_n(self, peaks):
        # The documented non-bound: the (n_samples, n_pre_filter) survivor matrix and its test
        # statistics are not batched, so peak memory keeps a linear term in n. This pins the
        # claim the CPP.run docstring makes; relax it if that term is ever streamed away.
        small, large = _growth(peaks, "fixed_batch_%d" % MEM_SIZES[0]), _growth(peaks, "fixed_batch_%d" % MEM_SIZES[-1])
        assert large > small, (
            f"peak growth at n={MEM_SIZES[-1]} ({large:.1f} MB) did not exceed n={MEM_SIZES[0]} "
            f"({small:.1f} MB); if sample batching now bounds memory outright, update the "
            f"'CPP.run' n_sample_batches docstring together with this test")

    def test_batch_size_not_batch_count_bounds_the_peak(self, peaks):
        # Same total input, same axis: a constant batch size beats a constant batch count (whose
        # batch grows with n). This is what identifies the batch as the bounded dimension.
        n = MEM_SIZES[-1]
        fixed_size, fixed_count = _growth(peaks, "fixed_batch_%d" % n), _growth(peaks, "fixed_chunks_%d" % n)
        ratio = fixed_size / fixed_count
        assert ratio <= MAX_FIXED_BATCH_VS_FIXED_COUNT_RATIO, (
            f"at n={n}, batch size {MEM_BATCH_SIZE} grew {fixed_size:.1f} MB vs {fixed_count:.1f} MB "
            f"for {MEM_N_CHUNKS} batches (ratio {ratio:.2f}, threshold "
            f"{MAX_FIXED_BATCH_VS_FIXED_COUNT_RATIO})")

    def test_scale_axis_peak_rss_lower_than_in_memory(self, peaks):
        n = MEM_SIZES[0]
        chunked, in_memory = _growth(peaks, "scale_batched_%d" % n), _growth(peaks, "in_memory_%d" % n)
        ratio = chunked / in_memory
        assert ratio <= MAX_PEAK_RSS_RATIO, (
            f"n_batches={MEM_N_CHUNKS} peak-RSS growth {chunked:.1f} MB is {ratio:.2f}x the "
            f"in-memory growth {in_memory:.1f} MB (threshold {MAX_PEAK_RSS_RATIO})")

    def test_in_memory_growth_is_substantial(self, peaks):
        # Guards the fixture: if the in-memory growth collapsed (e.g. the fixture got too small),
        # the ratio assertions above would stop meaning anything.
        assert _growth(peaks, "in_memory_%d" % MEM_SIZES[0]) >= MIN_IN_MEMORY_GROWTH_MB

    def test_each_configuration_measured_in_its_own_process(self, peaks):
        # Baselines are taken after imports + fixture construction, so they must be comparable
        # across the fresh interpreters (no warm cache carried over). The fixture itself is small
        # next to the interpreter footprint, so even the 400-sample bases line up.
        bases = [peaks[name]["base_mb"] for name in CONFIGS]
        assert max(bases) - min(bases) < 0.25 * max(bases)


# ---------------------------------------------------------------------------
# Fast-tier guard on the bounded term (no memory measurement)
# ---------------------------------------------------------------------------
class TestCPPRunChunkedBatchSize:
    """The sample-batched path never assigns more than one batch of sequences at a time.

    The peak-RSS class above is ``slow`` (it shells out eight times), so this class keeps the
    mechanism it measures under the blocking unit tier: the bounded term is the per-batch
    ``(batch_size x positions x n_scales)`` tensor built by ``assign_scale_values_to_seq``, so
    spying on that call records exactly how many sequences are resident per assignment. It runs in
    milliseconds on the small fixture and fails if ``n_sample_batches`` is ever silently ignored.
    """

    @staticmethod
    def _assigned_batch_sizes(cpp, labels, monkeypatch, **kws):
        """Number of sequences handed to each ``assign_scale_values_to_seq`` call of one run."""
        from aaanalysis.feature_engineering._backend import cpp_run as backend
        sizes = []
        original = backend.assign_scale_values_to_seq

        def _spy(*args, **kwargs):
            df_parts = kwargs.get("df_parts", args[0] if args else None)
            sizes.append(len(df_parts))
            return original(*args, **kwargs)

        monkeypatch.setattr(backend, "assign_scale_values_to_seq", _spy)
        cpp.run(labels=labels, n_filter=N_FILTER, n_jobs=1, **kws)
        return sizes

    @pytest.mark.parametrize("n_sample_batches", [2, 3, 5])
    def test_no_batch_exceeds_the_requested_batch_size(self, small, monkeypatch, n_sample_batches):
        n = small["n_samples"]
        expected = -(-n // n_sample_batches)
        sizes = self._assigned_batch_sizes(small["cpp"], small["labels"], monkeypatch,
                                           n_sample_batches=n_sample_batches)
        assert sizes, "the sample-batched path did not assign any scale values"
        assert max(sizes) <= expected, (
            f"n_sample_batches={n_sample_batches} on {n} samples assigned up to {max(sizes)} "
            f"sequences at once, above the batch size {expected}")

    @pytest.mark.parametrize("n_sample_batches", [2, 3, 5])
    def test_both_passes_cover_every_sample_exactly_once(self, small, monkeypatch, n_sample_batches):
        # Two passes (stats, then survivor recompute) over disjoint batches spanning all samples.
        n = small["n_samples"]
        sizes = self._assigned_batch_sizes(small["cpp"], small["labels"], monkeypatch,
                                           n_sample_batches=n_sample_batches)
        assert sum(sizes) == 2 * n

    def test_unbatched_run_assigns_every_sample_at_once(self, small, monkeypatch):
        # The contrast that gives the assertions above their meaning.
        sizes = self._assigned_batch_sizes(small["cpp"], small["labels"], monkeypatch)
        assert max(sizes) == small["n_samples"]

    def test_more_batches_means_smaller_assignments(self, small, monkeypatch):
        few = self._assigned_batch_sizes(small["cpp"], small["labels"], monkeypatch, n_sample_batches=2)
        many = self._assigned_batch_sizes(small["cpp"], small["labels"], monkeypatch, n_sample_batches=6)
        assert max(many) < max(few)

    def test_scale_batching_does_not_bound_the_sample_axis(self, small, monkeypatch):
        # n_batches chunks the scale axis only, so every assignment still spans all samples. This
        # is why n_sample_batches exists and why the two are documented differently.
        sizes = self._assigned_batch_sizes(small["cpp"], small["labels"], monkeypatch, n_batches=2)
        assert set(sizes) == {small["n_samples"]}


# ---------------------------------------------------------------------------
# Identity across chunk sizes + default path
# ---------------------------------------------------------------------------
class TestCPPRunChunked:
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

    def test_default_equals_explicit_none(self, small):
        df = small["cpp"].run(labels=small["labels"], n_filter=N_FILTER, n_jobs=1,
                              n_batches=None, n_sample_batches=None)
        pd.testing.assert_frame_equal(small["df_ref"], df, check_exact=True)

    def test_default_run_reproducible(self, small):
        df = small["cpp"].run(labels=small["labels"], n_filter=N_FILTER, n_jobs=1)
        pd.testing.assert_frame_equal(small["df_ref"], df, check_exact=True)


# ---------------------------------------------------------------------------
# Combinations, ordering, cross-chunk-size agreement
# ---------------------------------------------------------------------------
class TestCPPRunChunkedComplex:
    """Chunk sizes crossed against each other: ordering, reproducibility, agreement."""

    def test_two_scale_chunk_sizes_agree_except_fdr(self, small):
        df_a = small["cpp"].run(labels=small["labels"], n_filter=N_FILTER, n_jobs=1, n_batches=2)
        df_b = small["cpp"].run(labels=small["labels"], n_filter=N_FILTER, n_jobs=1, n_batches=4)
        _assert_same_except_fdr(df_a, df_b)

    def test_two_sample_chunk_sizes_agree(self, small):
        df_a = small["cpp"].run(labels=small["labels"], n_filter=N_FILTER, n_jobs=1, n_sample_batches=2)
        df_b = small["cpp"].run(labels=small["labels"], n_filter=N_FILTER, n_jobs=1, n_sample_batches=7)
        pd.testing.assert_frame_equal(df_a, df_b, check_exact=True)

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
# Golden (hand-computed) values
# ---------------------------------------------------------------------------
GOLDEN_SEQS = ["AAAA", "AAAC", "CCCC", "CCCA"]
GOLDEN_LABELS = [1, 1, 0, 0]


def _golden_cpp():
    """Tiny hand-verifiable fixture: 4 sequences of length 4, 2 scales, one Segment(1,1) split.

    ``SC1`` scores A=1 and every other residue 0; ``SC2`` is its complement (C=0, else 1). With a
    single whole-part Segment(1,1) split over 'tmd', each feature value is the mean scale value of
    the 4 residues, so all statistics are computable by hand (see the test bodies).
    """
    df_seq = pd.DataFrame({"entry": ["P1", "P2", "P3", "P4"], "sequence": GOLDEN_SEQS,
                           "tmd_start": 1, "tmd_stop": 4})
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=["tmd"], jmd_n_len=0, jmd_c_len=0)
    df_scales = pd.DataFrame({"SC1": [0.0] * 20, "SC2": [1.0] * 20}, index=LIST_AA)
    df_scales.loc["A", "SC1"] = 1.0
    df_scales.loc["C", "SC2"] = 0.0
    df_cat = pd.DataFrame({"scale_id": ["SC1", "SC2"], "category": ["CatA", "CatB"],
                           "subcategory": ["SubA", "SubB"], "scale_name": ["N1", "N2"],
                           "scale_description": ["D1", "D2"]})
    split_kws = sf.get_split_kws(split_types="Segment", n_split_min=1, n_split_max=1)
    cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, df_cat=df_cat,
                 verbose=False, random_state=SEED)
    return cpp


def _golden_run(**kws):
    """Run the golden fixture with the filters opened up so both candidate features survive."""
    return _golden_cpp().run(labels=GOLDEN_LABELS, n_filter=2, n_jobs=1, max_std_test=0.99,
                             max_overlap=1.0, max_cor=1.0, check_cat=False,
                             tmd_len=4, jmd_n_len=0, jmd_c_len=0, **kws)


class TestCPPRunChunkedGoldenValues:
    """Hand-computed statistics, reproduced on both chunked axes.

    Per-sequence feature values (mean of the 4 residue scale values):

    ====  ========  ================  ================
    seq   residues  SC1 (A=1, else 0)  SC2 (C=0, else 1)
    ====  ========  ================  ================
    P1    AAAA      4/4 = 1.00         4/4 = 1.00
    P2    AAAC      3/4 = 0.75         3/4 = 0.75
    P3    CCCC      0/4 = 0.00         0/4 = 0.00
    P4    CCCA      1/4 = 0.25         1/4 = 0.25
    ====  ========  ================  ================

    Test group = (P1, P2), reference = (P3, P4), so for both scales:
    mean_test = (1.00 + 0.75)/2 = 0.875, mean_ref = (0.00 + 0.25)/2 = 0.125,
    mean_dif = abs_mean_dif = 0.875 - 0.125 = 0.75,
    std_test = std_ref = |1.00 - 0.75|/2 = 0.125 (population std of two values),
    abs_auc = 0.5 (both test values exceed both reference values: perfect separation).
    """

    GOLDEN = dict(mean_dif=0.75, abs_mean_dif=0.75, std_test=0.125, std_ref=0.125, abs_auc=0.5)
    FEATURES = ["TMD-Segment(1,1)-SC1", "TMD-Segment(1,1)-SC2"]

    @pytest.mark.parametrize("kws", [{}, dict(n_batches=2), dict(n_sample_batches=2),
                                     dict(n_sample_batches=4)])
    def test_golden_feature_ids(self, kws):
        df = _golden_run(**kws)
        assert sorted(df["feature"].to_list()) == self.FEATURES

    @pytest.mark.parametrize("kws", [{}, dict(n_batches=2), dict(n_sample_batches=2),
                                     dict(n_sample_batches=4)])
    def test_golden_mean_dif(self, kws):
        df = _golden_run(**kws)
        assert df["mean_dif"].to_list() == [self.GOLDEN["mean_dif"]] * 2

    @pytest.mark.parametrize("kws", [{}, dict(n_batches=2), dict(n_sample_batches=2),
                                     dict(n_sample_batches=4)])
    def test_golden_abs_mean_dif(self, kws):
        df = _golden_run(**kws)
        assert df["abs_mean_dif"].to_list() == [self.GOLDEN["abs_mean_dif"]] * 2

    @pytest.mark.parametrize("kws", [{}, dict(n_batches=2), dict(n_sample_batches=2)])
    def test_golden_std_and_auc(self, kws):
        df = _golden_run(**kws)
        assert df["std_test"].to_list() == [self.GOLDEN["std_test"]] * 2
        assert df["std_ref"].to_list() == [self.GOLDEN["std_ref"]] * 2
        assert df["abs_auc"].to_list() == [self.GOLDEN["abs_auc"]] * 2

    @pytest.mark.parametrize("kws", [{}, dict(n_batches=2), dict(n_sample_batches=2)])
    def test_golden_positions(self, kws):
        # tmd_len=4 with no JMDs: the whole-part segment spans residues 1-4.
        df = _golden_run(**kws)
        assert df["positions"].to_list() == ["1,2,3,4"] * 2

    def test_golden_chunked_axes_match_in_memory_except_fdr(self):
        df_ref = _golden_run()
        _assert_same_except_fdr(df_ref, _golden_run(n_batches=2))
        pd.testing.assert_frame_equal(df_ref, _golden_run(n_sample_batches=2), check_exact=True)


if __name__ == "__main__":  # pragma: no cover - subprocess entry point for the RSS measurement
    print(json.dumps(_measure_in_this_process(sys.argv[1])))
