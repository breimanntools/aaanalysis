"""Committed perf benchmark suite over the hot public entry points (issue #187).

``pytest-benchmark`` micro-benchmarks for the genuinely hot AAanalysis call
paths, on small deterministic bundled fixtures (a tiny ``load_dataset`` slice,
bundled scales, and the bundled PDB test fixtures). These exist so perf work is
data-driven and a regression is caught instead of shipping silently.

**Non-blocking by construction.** The benchmarks need the opt-in ``[bench]``
install extra (``pytest-benchmark``); when it is absent the whole module is
skipped at import (``importorskip`` below). The unit / coverage / integration
matrices install only ``[pro]`` / ``[dev]`` / core, so they never collect these
— wall-clock is far too noisy to gate merges on. They run in the dedicated perf
nightly (``.github/workflows/perf_nightly.yml``) and on demand.

Run locally::

    pip install -e ".[dev,pro,bench]"
    pytest tests/benchmarks --benchmark-json=perf_run.json -c tests/pytest.ini
    python .github/scripts/check_perf_regression.py perf_run.json

Each benchmark builds its input data ONCE in a module-scoped fixture, so only
the hot call itself is timed (not fixture construction). Fixtures are tiny and
seeded for determinism; the whole suite runs in well under ~2 minutes.
"""
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


# I Shared fixtures (built once per module; only the hot call is benchmarked)
@pytest.fixture(scope="module")
def seq_inputs():
    """Small DOM_GSEC slice + parts/scales for the CPP / SequenceFeature paths."""
    df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
    labels = df_seq["label"].to_list()
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq)
    df_scales = aa.load_scales(top60_n=38).T.head(10).T
    split_kws = sf.get_split_kws(split_types=["Segment"],
                                 n_split_min=1, n_split_max=3)
    return dict(df_seq=df_seq, labels=labels, sf=sf, df_parts=df_parts,
                df_scales=df_scales, split_kws=split_kws)


@pytest.fixture(scope="module")
def feature_matrix_inputs():
    """A real feature list + parts for SequenceFeature.feature_matrix."""
    df_seq = aa.load_dataset(name="DOM_GSEC", n=20)
    features = aa.load_features(name="DOM_GSEC").head(20)["feature"].to_list()
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
    result = benchmark(lambda: cpp.run(labels=seq_inputs["labels"], n_jobs=1))
    assert isinstance(result, pd.DataFrame)


def test_cpp_run_num(benchmark, seq_inputs, num_inputs):
    """CPP.run_num — the numerical-mode (dict_num) recompute hot path."""
    cpp = aa.CPP(df_parts=seq_inputs["df_parts"], df_scales=seq_inputs["df_scales"])
    result = benchmark(lambda: cpp.run_num(
        dict_num_parts=num_inputs["dict_num_parts"],
        labels=num_inputs["labels"], n_jobs=1))
    assert isinstance(result, pd.DataFrame)


def test_aaclust_fit(benchmark, cluster_inputs):
    """AAclust.fit — clustering / medoid selection."""
    result = benchmark(lambda: aa.AAclust().fit(cluster_inputs, n_clusters=5))
    assert result is not None


def test_sequence_feature_matrix(benchmark, feature_matrix_inputs):
    """SequenceFeature.feature_matrix — feature-vector assembly."""
    sf = aa.SequenceFeature()
    result = benchmark(lambda: sf.feature_matrix(
        features=feature_matrix_inputs["features"],
        df_parts=feature_matrix_inputs["df_parts"]))
    assert isinstance(result, np.ndarray)


def test_aa_window_sampler(benchmark, window_inputs):
    """AAWindowSampler.sample_same_protein — the vectorized window filters."""
    aaws = aa.AAWindowSampler()
    result = benchmark(lambda: aaws.sample_same_protein(
        df_seq=window_inputs, pos_col="pos", n=50, window_size=9, seed=0))
    assert isinstance(result, pd.DataFrame)


def test_dpulearn_fit(benchmark, pu_inputs):
    """dPULearn.fit — reliable-negative identification."""
    result = benchmark(lambda: aa.dPULearn().fit(
        pu_inputs["X"], pu_inputs["labels"], n_unl_to_neg=10))
    assert result is not None


def test_tree_model_fit(benchmark, tree_inputs):
    """TreeModel.fit — tree-ensemble training + feature importance."""
    result = benchmark(lambda: aa.TreeModel().fit(
        X=tree_inputs["X"], labels=tree_inputs["labels"],
        use_rfe=False, n_cv=2, n_rounds=2))
    assert result is not None


def test_encode_pdb(benchmark, pdb_inputs):
    """StructurePreprocessor.encode_pdb — PDB -> per-residue tensor (pro)."""
    pytest.importorskip("Bio")  # biopython, from the [pro] extra
    stp = aa.StructurePreprocessor()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = benchmark(lambda: stp.encode_pdb(
            df_seq=pdb_inputs, pdb_folder=str(PDB_FIXTURES), features=["bfactor"]))
    assert isinstance(result, dict)
