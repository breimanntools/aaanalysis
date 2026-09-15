"""This is a script to test the chunked (memory-bounded) paths of CPP.run().

``CPP.run`` offers two opt-in chunking modes: ``n_batches`` (scale axis) and
``n_sample_batches`` (sample axis). This file enforces their contract:

* peak RSS with many chunks is measurably lower than the in-memory run (measured in a
  fresh subprocess per configuration, see ``TestCPPRunChunkedPeakMemory``),
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

# Memory fixture: large enough that the (n_samples x positions x n_scales) tensor dominates
# the fixed interpreter/import footprint, small enough to run in seconds per subprocess.
MEM_N_SAMPLES = 100
MEM_N_SCALES = 24
MEM_N_CHUNKS = 6
# Chunked peak RSS growth must be at most this fraction of the in-memory growth. Measured
# ratios were 0.19-0.24 (see TestCPPRunChunkedPeakMemory), so 0.5 is ~2x the worst measurement.
MAX_PEAK_RSS_RATIO = 0.5
# Fixture guard: if the in-memory run stopped allocating a big tensor, the ratio would be noise.
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
CONFIGS = {"in_memory": {},
           "n_batches": dict(n_batches=MEM_N_CHUNKS),
           "n_sample_batches": dict(n_sample_batches=MEM_N_CHUNKS)}


def _peak_rss_bytes():
    """Peak resident set size of this process (bytes on macOS, KB on Linux -> normalised)."""
    import resource
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == "darwin" else rss * 1024.0


def _measure_in_this_process(name):
    """Build the memory fixture, then report the peak-RSS growth (MB) caused by ``CPP.run``."""
    import gc
    df_parts, labels, df_scales = _build_inputs(n_samples=MEM_N_SAMPLES, n_scales=MEM_N_SCALES)
    cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, verbose=False, random_state=SEED)
    gc.collect()
    base = _peak_rss_bytes()  # after imports and fixture construction
    cpp.run(labels=labels, n_filter=N_FILTER, n_jobs=1, **CONFIGS[name])
    return dict(name=name, base_mb=base / 1e6, peak_mb=_peak_rss_bytes() / 1e6,
                growth_mb=(_peak_rss_bytes() - base) / 1e6)


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
# Marked slow: three fresh interpreters, each importing aaanalysis, building the memory fixture
# and running the full CPP pipeline (~8 s in total on an idle machine, more on a contended one).
# It is also the one class here that shells out, so it stays opt-in rather than running in the
# blocking unit tier; the measured ratios themselves were stable across repetitions.
@pytest.mark.slow
class TestCPPRunChunkedPeakMemory:
    """Chunking bounds peak RSS (one fresh subprocess per configuration, no wall-clock assertion).

    Each subprocess reports ``ru_maxrss`` (bytes on macOS, KB on Linux) right before ``CPP.run``
    (after imports and fixture construction) and again afterwards; the difference is the growth
    attributable to the run. Measuring one configuration per process removes the order dependence
    that a single-process measurement has (the first run warms the instance cache).

    Measured on the memory fixture (100 samples x 24 scales, 40-residue sequences, default parts
    and splits, ``n_filter=20``, ``n_jobs=1``), macOS / Python 3.13, 6 chunks, three repetitions:

    * in-memory growth: 158.9 / 159.0 / 157.3 MB
    * ``n_batches=6``: 38.6 / 32.4 / 29.7 MB (ratios 0.24 / 0.20 / 0.19)
    * ``n_sample_batches=6``: 35.6 / 38.1 / 34.8 MB (ratios 0.22 / 0.24 / 0.22)

    ``MAX_PEAK_RSS_RATIO = 0.5`` is about twice the worst measured ratio (0.24), leaving room for
    a different allocator / platform while still failing if chunking stops bounding memory. Only
    many chunks are asserted: with 2 chunks the fixed per-pass overhead dominates and the peak is
    not reliably lower.
    """

    def test_n_batches_peak_rss_lower_than_in_memory(self, peaks):
        ratio = peaks["n_batches"]["growth_mb"] / peaks["in_memory"]["growth_mb"]
        assert ratio <= MAX_PEAK_RSS_RATIO, (
            f"n_batches={MEM_N_CHUNKS} peak-RSS growth {peaks['n_batches']['growth_mb']:.1f} MB is "
            f"{ratio:.2f}x the in-memory growth {peaks['in_memory']['growth_mb']:.1f} MB "
            f"(threshold {MAX_PEAK_RSS_RATIO})")

    def test_n_sample_batches_peak_rss_lower_than_in_memory(self, peaks):
        ratio = peaks["n_sample_batches"]["growth_mb"] / peaks["in_memory"]["growth_mb"]
        assert ratio <= MAX_PEAK_RSS_RATIO, (
            f"n_sample_batches={MEM_N_CHUNKS} peak-RSS growth "
            f"{peaks['n_sample_batches']['growth_mb']:.1f} MB is {ratio:.2f}x the in-memory growth "
            f"{peaks['in_memory']['growth_mb']:.1f} MB (threshold {MAX_PEAK_RSS_RATIO})")

    def test_in_memory_growth_is_substantial(self, peaks):
        # Guards the fixture: if the in-memory growth collapsed (e.g. the fixture got too small),
        # the ratio assertions above would stop meaning anything.
        assert peaks["in_memory"]["growth_mb"] >= MIN_IN_MEMORY_GROWTH_MB

    def test_each_configuration_measured_in_its_own_process(self, peaks):
        # Baselines are taken after imports + fixture construction, so they must be comparable
        # across the three fresh interpreters (no warm cache carried over).
        bases = [peaks[name]["base_mb"] for name in CONFIGS]
        assert max(bases) - min(bases) < 0.25 * max(bases)


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
