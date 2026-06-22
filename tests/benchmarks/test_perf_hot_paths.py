"""Committed perf benchmark suite over the hot public entry points (issue #187).

``pytest-benchmark`` micro-benchmarks for the genuinely hot AAanalysis call
paths, on small deterministic bundled fixtures (a tiny ``load_dataset`` slice,
bundled scales, and the bundled PDB test fixtures). These exist so perf work is
data-driven and a regression is caught instead of shipping silently.

**Merge-gating via a same-runner A/B.** The benchmarks need the opt-in ``[bench]``
install extra (``pytest-benchmark``); when it is absent the whole module is
skipped at import (``importorskip`` below). The unit / coverage / integration
matrices install only ``[pro]`` / ``[dev]`` / core, so they never collect these.
The perf workflow (``.github/workflows/perf_nightly.yml``) instead runs this
suite **twice on the same runner in the same job** — the current working tree and
the latest stable release installed ``--no-deps`` onto the same dependency set —
and ``check_perf_regression.py`` compares the two. Running both builds on one
runner cancels hardware / OS / Python / dependency variance, so the comparison is
reliable enough to **gate merges** (per-benchmark ``1.3×`` on the meaty paths,
``2.0×`` on the sub-2ms micro-paths). There is no committed static baseline; the
baseline is the live release.

Run the A/B locally (mirrors the CI job). ``PYTHONSAFEPATH=1`` +
``--import-mode=importlib`` are MANDATORY: without them, running pytest from the
repo root puts the working-tree ``aaanalysis/`` on ``sys.path`` and BOTH venvs
import master -> a meaningless master-vs-master comparison::

    python -m venv .v_cur && .v_cur/bin/pip install -e ".[dev,pro,bench]"
    PYTHONSAFEPATH=1 .v_cur/bin/pytest tests/benchmarks --import-mode=importlib \
        --benchmark-json=current_run.json -c tests/pytest.ini
    .v_cur/bin/pip freeze | grep -viE '^(-e |aaanalysis([=@ ]|$))' > deps.txt
    python -m venv .v_rel && .v_rel/bin/pip install -r deps.txt && .v_rel/bin/pip install aaanalysis --no-deps
    PYTHONSAFEPATH=1 .v_rel/bin/pytest tests/benchmarks --import-mode=importlib \
        --benchmark-json=released_run.json -c tests/pytest.ini || true
    python .github/scripts/check_perf_regression.py current_run.json --baseline-run released_run.json

Each benchmark builds its input data ONCE in a module-scoped fixture, so only
the hot call itself is timed (not fixture construction). Fixtures are tiny and
seeded for determinism; the whole suite runs in well under ~2 minutes.
"""
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Opt-in [bench] extra. Absent in the blocking matrix -> the suite is skipped
# there rather than erroring on the missing ``benchmark`` fixture.
pytest.importorskip("pytest_benchmark")

import aaanalysis as aa

aa.options["verbose"] = False

# Bundled PDB test fixtures: <repo>/aaanalysis/_data/pdb_test (P1/P2/P3.pdb).
# From <repo>/tests/benchmarks/this_file -> parents[2] is the repo root.
PDB_FIXTURES = Path(__file__).resolve().parents[2] / \
    "aaanalysis" / "_data" / "pdb_test"

# The release that first contains each benchmarked method (its `versionadded`).
# The A/B gate (.github/scripts/check_perf_regression.py) reads this from the run
# JSON to tell apart two cases for a benchmark absent from the released run:
#   * gated_since <= released baseline -> MUST be present; if missing the method
#     was renamed/removed and the gate silently lost it -> the gate FAILS.
#   * gated_since  > released baseline -> "pending-gate": measured now, auto-gates
#     the moment that release ships (the dynamic baseline catches up).
# So a method added against the unreleased 1.1 line is tagged "1.1.0": measured
# today, gated automatically when 1.1.0 reaches PyPI -- no manual trigger.
GATED_SINCE = {
    "test_cpp_run": "1.0.0",
    "test_cpp_run_num": "1.1.0",
    "test_aaclust_fit": "1.0.0",
    "test_sequence_feature_matrix": "1.0.0",
    "test_aa_window_sampler": "1.1.0",
    "test_dpulearn_fit": "1.0.0",
    "test_tree_model_fit": "1.0.0",
    "test_encode_pdb": "1.1.0",
    "test_cpp_eval": "1.0.0",
    "test_cpp_simplify": "1.1.0",
    "test_sequence_feature_get_df_feat": "1.0.0",
    "test_prune_by_correlation": "1.1.0",
    "test_comp_auc_adjusted": "1.0.0",
    "test_comp_kld": "1.0.0",
    "test_get_sliding_aa_window": "1.0.0",
    "test_encode_one_hot": "1.0.0",
}


@pytest.fixture(autouse=True)
def _tag_gated_since(request, benchmark):
    """Stamp each benchmark's ``gated_since`` into the pytest-benchmark JSON."""
    name = request.node.name.split("[")[0]
    if name in GATED_SINCE:
        benchmark.extra_info["gated_since"] = GATED_SINCE[name]


def _digest(obj):
    """Byte-exact digest of a benchmark result, or None if it has no deterministic
    serialization (e.g. a fitted model object).

    Used for the OUTPUT A/B: check_perf_regression.py compares current vs released
    digests so a speedup that silently changes output is caught. Both venvs share
    the SAME pandas/numpy (Option A --no-deps), so pandas' hash and ndarray bytes
    are directly comparable. Model-returning methods (``*.fit`` -> ``self``) return
    None here and are simply not output-compared (stochastic; covered by the
    reproducibility tests instead)."""
    h = hashlib.sha256()

    def upd(x):
        if isinstance(x, pd.DataFrame):
            h.update(b"DF|"); h.update("|".join(map(str, x.columns)).encode())
            h.update(pd.util.hash_pandas_object(x, index=True).values.tobytes())
        elif isinstance(x, pd.Series):
            h.update(b"S|")
            h.update(pd.util.hash_pandas_object(x, index=True).values.tobytes())
        elif isinstance(x, np.ndarray):
            h.update(b"A|"); h.update(str(x.dtype).encode()); h.update(str(x.shape).encode())
            h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, dict):
            h.update(b"D|")
            for k in sorted(x, key=str):
                h.update(str(k).encode()); h.update(b"="); upd(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(b"L|")
            for it in x:
                upd(it)
        elif isinstance(x, (str, bytes, int, float, bool, type(None))):
            h.update(repr(x).encode())
        else:
            raise TypeError  # no deterministic serialization (e.g. a fitted model)

    try:
        upd(obj)
    except TypeError:
        return None
    return h.hexdigest()


def _bench(benchmark, fn):
    """Run a benchmark and, when the result is digestible data, stamp a byte-exact
    ``output_digest`` into the run JSON for the current-vs-released output A/B."""
    result = benchmark(fn)
    digest = _digest(result)
    if digest is not None:
        benchmark.extra_info["output_digest"] = digest
    return result


# I Shared fixtures (built once per module; only the hot call is benchmarked)
@pytest.fixture(scope="module")
def seq_inputs():
    """DOM_GSEC slice + parts/scales for the CPP / SequenceFeature paths.

    Uses the FULL scale set (~586) on purpose: the CPP hot path is dominated by
    fixed Python/setup overhead on a tiny scale set, so a ``head(10)`` fixture
    measures overhead rather than the vectorized feature kernel and produces
    misleading A/B ratios. Full scales + Segment splits puts the benchmark in the
    regime that actually matters (and keeps the slow released-side runtime to a
    few seconds; Segment+Pattern would be ~3x slower again, too costly for the
    A/B's released side).
    """
    df_seq = aa.load_dataset(name="DOM_GSEC", n=20)
    labels = df_seq["label"].to_list()
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq)
    df_scales = aa.load_scales()
    split_kws = sf.get_split_kws(split_types=["Segment"],
                                 n_split_min=1, n_split_max=3)
    return dict(df_seq=df_seq, labels=labels, sf=sf, df_parts=df_parts,
                df_scales=df_scales, split_kws=split_kws)


@pytest.fixture(scope="module")
def feature_matrix_inputs():
    """A real feature list + parts for SequenceFeature.feature_matrix.

    Uses the full DOM_GSEC feature set (~150) over 80 sequences so the vectorized
    assembly is actually exercised; the speedup over the released build grows with
    feature count (~1.5x at 20 features, ~2.5x at the full 150).
    """
    df_seq = aa.load_dataset(name="DOM_GSEC", n=40)
    features = aa.load_features(name="DOM_GSEC")["feature"].to_list()
    df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
    return dict(features=features, df_parts=df_parts)


@pytest.fixture(scope="module")
def num_inputs(seq_inputs):
    """dict_num_parts derived from the scale lookup, for CPP.run_num."""
    df_seq, df_scales = seq_inputs["df_seq"], seq_inputs["df_scales"]
    aa_to_idx = {a: i for i, a in enumerate(df_scales.index)}
    n_aa = len(aa_to_idx)
    scale_matrix = np.full((n_aa + 1, df_scales.shape[1]), np.nan, dtype=np.float64)
    for col_idx, scale in enumerate(df_scales.columns):
        for a, idx in aa_to_idx.items():
            scale_matrix[idx, col_idx] = df_scales[scale][a]
    dict_num = {}
    for _, row in df_seq.iterrows():
        idxs = np.array([aa_to_idx.get(c, n_aa) for c in row["sequence"]],
                        dtype=np.int64)
        dict_num[row["entry"]] = scale_matrix[idxs, :]
    _, dict_num_parts = aa.NumericalFeature().get_parts(df_seq=df_seq,
                                                        dict_num=dict_num)
    return dict(dict_num_parts=dict_num_parts, labels=seq_inputs["labels"])


@pytest.fixture(scope="module")
def cluster_inputs():
    """Deterministic dense matrix for AAclust.fit."""
    return np.random.RandomState(0).rand(80, 12)


@pytest.fixture(scope="module")
def pu_inputs():
    """X + PU labels (1 = positive, 2 = unlabeled) for dPULearn.fit."""
    X = np.random.RandomState(0).rand(100, 6)
    labels = np.array([1, 2] + list(np.random.RandomState(1).choice([1, 2], size=98)))
    return dict(X=X, labels=labels)


@pytest.fixture(scope="module")
def tree_inputs():
    """X + binary labels for TreeModel.fit."""
    X = np.random.RandomState(0).rand(30, 8)
    labels = np.array([1, 1, 0, 0] + list(np.random.RandomState(2).choice([1, 0], size=26)))
    return dict(X=X, labels=labels)


@pytest.fixture(scope="module")
def window_inputs():
    """Small df_seq with per-protein positive positions for AAWindowSampler."""
    unit = "ACDEFGHIKLMNPQRSTVWY"
    df_seq = pd.DataFrame({
        "entry": ["P1", "P2", "P3"],
        "sequence": [unit * 3, unit * 2, unit * 2],
        "pos": [[5, 25, 40], [10, 30], [15]],
    })
    return df_seq


@pytest.fixture(scope="module")
def pdb_inputs():
    """df_seq matching the bundled P1/P2 PDB fixtures (StructurePreprocessor)."""
    df_seq = pd.DataFrame({
        "entry": ["P1", "P2"],
        "sequence": ["ACDEFGHIKLMNPQRS", "VLIMKRSTGADE"],
    })
    return df_seq


# II Benchmarks (one per hot entry point; >=7 covered)
def test_cpp_run(benchmark, seq_inputs):
    """CPP.run — the sequence-mode feature-engineering hot path."""
    cpp = aa.CPP(df_parts=seq_inputs["df_parts"],
                 df_scales=seq_inputs["df_scales"],
                 split_kws=seq_inputs["split_kws"])
    result = _bench(benchmark, lambda: cpp.run(labels=seq_inputs["labels"], n_jobs=1))
    assert isinstance(result, pd.DataFrame)


def test_cpp_run_num(benchmark, seq_inputs, num_inputs):
    """CPP.run_num — the numerical-mode (dict_num) recompute hot path."""
    cpp = aa.CPP(df_parts=seq_inputs["df_parts"], df_scales=seq_inputs["df_scales"])
    result = _bench(benchmark, lambda: cpp.run_num(
        dict_num_parts=num_inputs["dict_num_parts"],
        labels=num_inputs["labels"], n_jobs=1))
    assert isinstance(result, pd.DataFrame)


def test_aaclust_fit(benchmark, cluster_inputs):
    """AAclust.fit — clustering / medoid selection."""
    result = _bench(benchmark, lambda: aa.AAclust().fit(cluster_inputs, n_clusters=5))
    assert result is not None


def test_sequence_feature_matrix(benchmark, feature_matrix_inputs):
    """SequenceFeature.feature_matrix — feature-vector assembly."""
    sf = aa.SequenceFeature()
    result = _bench(benchmark, lambda: sf.feature_matrix(
        features=feature_matrix_inputs["features"],
        df_parts=feature_matrix_inputs["df_parts"]))
    assert isinstance(result, np.ndarray)


def test_aa_window_sampler(benchmark, window_inputs):
    """AAWindowSampler.sample_same_protein — the vectorized window filters."""
    aaws = aa.AAWindowSampler()
    result = _bench(benchmark, lambda: aaws.sample_same_protein(
        df_seq=window_inputs, pos_col="pos", n=50, window_size=9, seed=0))
    assert isinstance(result, pd.DataFrame)


def test_dpulearn_fit(benchmark, pu_inputs):
    """dPULearn.fit — reliable-negative identification."""
    result = _bench(benchmark, lambda: aa.dPULearn().fit(
        pu_inputs["X"], pu_inputs["labels"], n_unl_to_neg=10))
    assert result is not None


def test_tree_model_fit(benchmark, tree_inputs):
    """TreeModel.fit — tree-ensemble training + feature importance."""
    result = _bench(benchmark, lambda: aa.TreeModel().fit(
        X=tree_inputs["X"], labels=tree_inputs["labels"],
        use_rfe=False, n_cv=2, n_rounds=2))
    assert result is not None


def test_encode_pdb(benchmark, pdb_inputs):
    """StructurePreprocessor.encode_pdb — PDB -> per-residue tensor (pro)."""
    pytest.importorskip("Bio")  # biopython, from the [pro] extra
    stp = aa.StructurePreprocessor()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _bench(benchmark, lambda: stp.encode_pdb(
            df_seq=pdb_inputs, pdb_folder=str(PDB_FIXTURES), features=["bfactor"]))
    assert isinstance(result, dict)


# III Extended coverage — other compute-heavy public entry points (not plotting).
# Built on the same tiny seeded fixtures; methods absent from the latest release
# (e.g. CPP.simplify) are simply reported unbaselined by the A/B gate.
@pytest.fixture(scope="module")
def df_feat(seq_inputs):
    """A real df_feat (built once) for the CPP.eval / simplify / prune paths."""
    cpp = aa.CPP(df_parts=seq_inputs["df_parts"], df_scales=seq_inputs["df_scales"],
                 split_kws=seq_inputs["split_kws"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cpp.run(labels=seq_inputs["labels"], n_jobs=1)


@pytest.fixture(scope="module")
def metrics_inputs():
    """A feature matrix X + binary class labels for the comp_* metric scorers."""
    X = np.random.RandomState(0).rand(80, 300)
    labels = np.array([1] * 40 + [0] * 40)
    return dict(X=X, labels=labels)


@pytest.fixture(scope="module")
def seq_strings():
    """A handful of equal-length sequences for the per-sequence encoders."""
    unit = "ACDEFGHIKLMNPQRSTVWY"
    return [unit * 2 for _ in range(50)]


def test_cpp_eval(benchmark, seq_inputs, df_feat):
    """CPP.eval — feature-set quality scoring (trains models + redundancy)."""
    cpp = aa.CPP(df_parts=seq_inputs["df_parts"], df_scales=seq_inputs["df_scales"])
    # eval compares feature SETS, so it requires >= 2 of them.
    half = max(1, len(df_feat) // 2)
    list_df_feat = [df_feat, df_feat.head(half)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _bench(benchmark, lambda: cpp.eval(list_df_feat=list_df_feat,
                                            labels=seq_inputs["labels"], n_jobs=1))
    assert isinstance(result, pd.DataFrame)


def test_cpp_simplify(benchmark, seq_inputs, df_feat):
    """CPP.simplify — interpretability-driven scale swapping (fast candidate search)."""
    cpp = aa.CPP(df_parts=seq_inputs["df_parts"], df_scales=seq_inputs["df_scales"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _bench(benchmark, lambda: cpp.simplify(
            df_feat=df_feat, labels=seq_inputs["labels"], candidate_search="fast"))
    assert isinstance(result, pd.DataFrame)


def test_sequence_feature_get_df_feat(benchmark, seq_inputs):
    """SequenceFeature.get_df_feat — full feature-DataFrame build for given features."""
    features = aa.load_features(name="DOM_GSEC").head(50)["feature"].to_list()
    sf = aa.SequenceFeature()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _bench(benchmark, lambda: sf.get_df_feat(
            features=features, df_parts=seq_inputs["df_parts"],
            labels=seq_inputs["labels"], df_scales=seq_inputs["df_scales"], n_jobs=1))
    assert isinstance(result, pd.DataFrame)


def test_prune_by_correlation(benchmark, seq_inputs, df_feat):
    """SequenceFeature.prune_by_correlation — O(n^2) redundancy pruning."""
    sf = aa.SequenceFeature()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _bench(benchmark, lambda: sf.prune_by_correlation(
            df_feat=df_feat, df_parts=seq_inputs["df_parts"],
            df_scales=seq_inputs["df_scales"], max_cor=0.7, n_jobs=1))
    assert isinstance(result, pd.DataFrame)


def test_comp_auc_adjusted(benchmark, metrics_inputs):
    """comp_auc_adjusted — per-feature adjusted AUC across two groups."""
    result = _bench(benchmark, lambda: aa.comp_auc_adjusted(
        X=metrics_inputs["X"], labels=metrics_inputs["labels"], n_jobs=1))
    assert result is not None


def test_comp_kld(benchmark, metrics_inputs):
    """comp_kld — per-feature Kullback-Leibler divergence across two groups."""
    result = _bench(benchmark, lambda: aa.comp_kld(
        X=metrics_inputs["X"], labels=metrics_inputs["labels"]))
    assert result is not None


def test_get_sliding_aa_window(benchmark):
    """SequencePreprocessor.get_sliding_aa_window — windowed slice over one sequence."""
    seq = "ACDEFGHIKLMNPQRSTVWY" * 10
    sp = aa.SequencePreprocessor()
    result = _bench(benchmark, lambda: sp.get_sliding_aa_window(seq=seq, window_size=9))
    assert result is not None


def test_encode_one_hot(benchmark, seq_strings):
    """SequencePreprocessor.encode_one_hot — vectorized one-hot scatter."""
    sp = aa.SequencePreprocessor()
    result = _bench(benchmark, lambda: sp.encode_one_hot(list_seq=seq_strings))
    assert result is not None
