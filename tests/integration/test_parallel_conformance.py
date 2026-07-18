"""This is a script to test deterministic parallel-execution conformance (n_jobs invariance).

Integration tier: a *metamorphic* guard that the same computation returns the same
result regardless of how it is parallelized (``n_jobs``) or chunked (``n_batches`` /
``n_sample_batches``). ``n_jobs`` only splits the per-feature work loop across
workers and concatenates the per-chunk results in a fixed chunk order, so no
cross-worker reduction ever changes a value: the serial and parallel paths are
byte-identical. This suite pins that contract so a future parallelism change (or a
non-deterministic third-party op) that silently altered a scientific output fails
CI. It complements, not duplicates, the unit ``test_n_jobs`` contract tests (which
check the ``check_n_jobs`` / ``resolve_n_jobs`` normalizer, not output equivalence).

Documented tolerance per compared output (numerical-equivalence policy tiers):

* ``n_jobs`` (worker count) invariance -> EXACT (T1), byte-identical:
    - ``CPP.run`` ``df_feat`` (values AND row order)
    - ``SequenceFeature.feature_matrix`` ``X``
    - ``comp_auc_adjusted`` AUC vector
    - ``dPULearn.eval`` ``df_eval`` (the KLD parallel path, ``comp_kld=True``)
    - ``CPPGrid`` per-config ``df_feat``
* ``n_sample_batches`` (sample-axis chunking) invariance -> EXACT (T1): ``CPP.run``
  ``df_feat`` is byte-identical to the unchunked run.
* ``n_batches`` (scale-axis chunking) invariance -> EXACT on the *selection-driving*
  outputs (selected feature set, row order, ``abs_auc``, ``abs_mean_dif``) but NOT on
  ``p_val_fdr_bh``: the Benjamini-Hochberg FDR correction is by definition computed
  across the whole candidate set, so batching the scale axis changes the corrected
  p-value. This is a known, expected batch-dependence (documented on ``CPP.run``),
  not nondeterminism -- feature selection and ranking are unaffected.

No third-party op in the covered paths is legitimately non-deterministic: joblib
splits an independent per-feature loop and the results are concatenated in a fixed
chunk order, so determinism is a structural property of the design, not luck. The
guard is exact-equality based, so any change that reorders or reduces across the
worker split breaks it immediately (verified by an order-mutation check in review).
"""
import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa

pytestmark = pytest.mark.integration

SEED = 0
N_PER_CLASS = 12
N_SCALES = 16
N_FILTER = 15
# Selection-driving columns: identical regardless of how the scale axis is chunked
# (only p_val_fdr_bh depends on the whole-candidate-set FDR correction).
SELECTION_COLS = ["feature", "abs_auc", "abs_mean_dif"]
# Worker counts >1 compared against the serial (n_jobs=1) baseline.
PARALLEL_N_JOBS = [2, 3]


@pytest.fixture(scope="module")
def inputs():
    """Small, fixed, seeded CPP inputs (parts + labels + scales), built once."""
    df_seq = aa.load_dataset(name="DOM_GSEC", n=N_PER_CLASS)
    labels = df_seq["label"].to_list()
    df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
    df_scales = aa.load_scales(top60_n=20).T.head(N_SCALES).T
    return dict(df_seq=df_seq, labels=labels, df_parts=df_parts, df_scales=df_scales)


def _cpp(inputs):
    """A fresh, seeded CPP over the shared inputs."""
    return aa.CPP(df_parts=inputs["df_parts"], df_scales=inputs["df_scales"],
                  verbose=False, random_state=SEED)


# ---------------------------------------------------------------------------
# CPP.run: n_jobs invariance (the headline scientific output, df_feat)
# ---------------------------------------------------------------------------
class TestCPPRunNJobsInvariance:
    """CPP.run df_feat is byte-identical across worker counts (values + order)."""

    @pytest.mark.parametrize("n_jobs", PARALLEL_N_JOBS)
    def test_df_feat_identical_to_serial(self, inputs, n_jobs):
        cpp = _cpp(inputs)
        df_serial = cpp.run(labels=inputs["labels"], n_filter=N_FILTER, n_jobs=1)
        df_parallel = cpp.run(labels=inputs["labels"], n_filter=N_FILTER, n_jobs=n_jobs)
        pd.testing.assert_frame_equal(df_serial, df_parallel, check_exact=True)

    @pytest.mark.parametrize("n_jobs", PARALLEL_N_JOBS)
    def test_feature_order_stable(self, inputs, n_jobs):
        # KPI: output ordering is stable across worker counts.
        cpp = _cpp(inputs)
        order_serial = cpp.run(labels=inputs["labels"], n_filter=N_FILTER, n_jobs=1)["feature"].to_list()
        order_parallel = cpp.run(labels=inputs["labels"], n_filter=N_FILTER, n_jobs=n_jobs)["feature"].to_list()
        assert order_serial == order_parallel

    def test_seeded_parallel_run_reproducible(self, inputs):
        # KPI: a seeded parallel run reproduces byte-for-byte across repeats AND
        # matches the serial run -> the seed propagates through the worker split.
        cpp = _cpp(inputs)
        df_a = cpp.run(labels=inputs["labels"], n_filter=N_FILTER, n_jobs=2)
        df_b = cpp.run(labels=inputs["labels"], n_filter=N_FILTER, n_jobs=2)
        df_serial = cpp.run(labels=inputs["labels"], n_filter=N_FILTER, n_jobs=1)
        pd.testing.assert_frame_equal(df_a, df_b, check_exact=True)
        pd.testing.assert_frame_equal(df_a, df_serial, check_exact=True)


# ---------------------------------------------------------------------------
# CPP.run: chunked vs unchunked invariance
# ---------------------------------------------------------------------------
class TestCPPRunChunkingInvariance:
    """Chunking the scale / sample axis does not change the scientific selection."""

    def test_sample_batches_identical_to_unchunked(self, inputs):
        # n_sample_batches bounds peak memory by batch size; the pass-2 stats are
        # computed on the full data, so df_feat is byte-identical to the single pass.
        cpp = _cpp(inputs)
        df_unchunked = cpp.run(labels=inputs["labels"], n_filter=N_FILTER, n_jobs=1)
        df_chunked = cpp.run(labels=inputs["labels"], n_filter=N_FILTER, n_jobs=1, n_sample_batches=3)
        pd.testing.assert_frame_equal(df_unchunked, df_chunked, check_exact=True)

    def test_scale_batches_identical_selection(self, inputs):
        # n_batches chunks the SCALE axis. Feature selection + ranking are driven by
        # abs_auc / abs_mean_dif, which are byte-identical; only p_val_fdr_bh differs
        # because BH FDR correction spans the whole candidate set (documented).
        cpp = _cpp(inputs)
        df_unchunked = cpp.run(labels=inputs["labels"], n_filter=N_FILTER, n_jobs=1)
        df_chunked = cpp.run(labels=inputs["labels"], n_filter=N_FILTER, n_jobs=1, n_batches=3)
        pd.testing.assert_frame_equal(df_unchunked[SELECTION_COLS], df_chunked[SELECTION_COLS],
                                      check_exact=True)

    def test_scale_batches_perturb_only_fdr_pvalue(self, inputs):
        # Pin the known batch-dependence: the ONLY numeric column that may differ
        # under scale-batching is p_val_fdr_bh. A new batch-dependent column would
        # surface here as an unexpected difference.
        cpp = _cpp(inputs)
        df_unchunked = cpp.run(labels=inputs["labels"], n_filter=N_FILTER, n_jobs=1)
        df_chunked = cpp.run(labels=inputs["labels"], n_filter=N_FILTER, n_jobs=1, n_batches=3)
        numeric_cols = df_unchunked.select_dtypes(include=[np.number]).columns
        differing = [c for c in numeric_cols
                     if not np.array_equal(df_unchunked[c].to_numpy(), df_chunked[c].to_numpy())]
        assert differing in ([], ["p_val_fdr_bh"])


# ---------------------------------------------------------------------------
# SequenceFeature.feature_matrix: n_jobs invariance
# ---------------------------------------------------------------------------
class TestFeatureMatrixNJobsInvariance:
    """The X matrix builder chunks features across workers; X is byte-identical."""

    @pytest.mark.parametrize("n_jobs", PARALLEL_N_JOBS)
    def test_matrix_identical_to_serial(self, inputs, n_jobs):
        sf = aa.SequenceFeature()
        df_feat = _cpp(inputs).run(labels=inputs["labels"], n_filter=N_FILTER, n_jobs=1)
        features = df_feat["feature"]
        X_serial = sf.feature_matrix(features=features, df_parts=inputs["df_parts"],
                                     df_scales=inputs["df_scales"], n_jobs=1)
        X_parallel = sf.feature_matrix(features=features, df_parts=inputs["df_parts"],
                                       df_scales=inputs["df_scales"], n_jobs=n_jobs)
        np.testing.assert_array_equal(np.asarray(X_serial), np.asarray(X_parallel))


# ---------------------------------------------------------------------------
# Metrics: comp_auc_adjusted n_jobs invariance
# ---------------------------------------------------------------------------
class TestMetricsNJobsInvariance:
    """comp_auc_adjusted splits the feature axis across workers; AUCs are identical."""

    @pytest.mark.parametrize("n_jobs", PARALLEL_N_JOBS)
    def test_auc_adjusted_identical_to_serial(self, inputs, n_jobs):
        sf = aa.SequenceFeature()
        df_feat = _cpp(inputs).run(labels=inputs["labels"], n_filter=N_FILTER, n_jobs=1)
        X = np.asarray(sf.feature_matrix(features=df_feat["feature"], df_parts=inputs["df_parts"],
                                         df_scales=inputs["df_scales"], n_jobs=1))
        labels = np.asarray(inputs["labels"])
        auc_serial = aa.comp_auc_adjusted(X=X, labels=labels, n_jobs=1)
        auc_parallel = aa.comp_auc_adjusted(X=X, labels=labels, n_jobs=n_jobs)
        np.testing.assert_array_equal(np.asarray(auc_serial), np.asarray(auc_parallel))


# ---------------------------------------------------------------------------
# Predictor: dPULearn.eval (KLD parallel path) n_jobs invariance
# ---------------------------------------------------------------------------
class TestPredictorNJobsInvariance:
    """dPULearn.eval reaches the parallel KLD backend; df_eval is byte-identical."""

    def test_dpulearn_eval_identical_to_serial(self, inputs):
        sf = aa.SequenceFeature()
        df_feat = _cpp(inputs).run(labels=inputs["labels"], n_filter=N_FILTER, n_jobs=1)
        X = np.asarray(sf.feature_matrix(features=df_feat["feature"], df_parts=inputs["df_parts"],
                                         df_scales=inputs["df_scales"], n_jobs=1))
        pu_labels = [1 if y == 1 else 2 for y in inputs["labels"]]
        dpul = aa.dPULearn(verbose=False).fit(X, labels=pu_labels, n_neg=4)
        list_labels = [dpul.labels_]
        eval_serial = aa.dPULearn.eval(X, list_labels=list_labels, comp_kld=True, n_jobs=1)
        eval_parallel = aa.dPULearn.eval(X, list_labels=list_labels, comp_kld=True, n_jobs=2)
        pd.testing.assert_frame_equal(eval_serial, eval_parallel, check_exact=True)


# ---------------------------------------------------------------------------
# CPPGrid: n_jobs invariance (parallelizes across configs)
# ---------------------------------------------------------------------------
class TestCPPGridNJobsInvariance:
    """CPPGrid parallelizes over sweep configs; each config's df_feat is identical."""

    def test_grid_per_config_identical_to_serial(self, inputs):
        params_cpp = dict(n_filter=[8, 12])
        grid_serial = aa.CPPGrid(df_seq=inputs["df_seq"], labels=inputs["labels"],
                                 verbose=False, random_state=SEED, n_jobs=1)
        grid_parallel = aa.CPPGrid(df_seq=inputs["df_seq"], labels=inputs["labels"],
                                   verbose=False, random_state=SEED, n_jobs=2)
        list_serial, _ = grid_serial.run(params_cpp=params_cpp, params_scales=inputs["df_scales"])
        list_parallel, _ = grid_parallel.run(params_cpp=params_cpp, params_scales=inputs["df_scales"])
        assert len(list_serial) == len(list_parallel)
        for df_serial, df_parallel in zip(list_serial, list_parallel):
            pd.testing.assert_frame_equal(df_serial, df_parallel, check_exact=True)
