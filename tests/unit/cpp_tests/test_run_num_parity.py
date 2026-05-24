"""This is a script to test CPP.run_num()'s numerical-mode contract.

Contract (per ADR-0001 — docs/adr/0001-cpp-backend-architecture.md):
- ``run_num`` ALWAYS requires ``dict_num_parts`` (from ``NumericalFeature.get_parts``);
  seq-mode goes through ``cpp.run`` instead.
- Round-trip parity: when ``dict_num`` is built by applying the AA→scale lookup
  to ``df_seq``, ``run_num(dict_num_parts=...)`` produces a ``df_feat`` that
  matches ``cpp.run(...)`` to within tolerance (not bit-exact — the numerical
  recompute uses vectorized nanmean over NaN-padded buffers, which can drift
  at ULP from per-sample ``np.mean`` and cascade through Mann-Whitney into
  small rank-order differences).
- Same dict_num_parts → same df_feat (within-path determinism).
- Layer-2 validation fires on malformed ``dict_num_parts``.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa


aa.options["verbose"] = False


def _build_fixture(n=10, n_scales=10):
    """Small DOM_GSEC slice for fast parity assertions."""
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
    labels = df_seq["label"].to_list()
    df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
    df_scales = aa.load_scales(top60_n=38).T.head(n_scales).T
    return df_seq, labels, df_parts, df_scales


def _build_dict_num_from_scales(df_seq, df_scales):
    """Build ``dict_num: Dict[entry, (L, D)]`` by applying the AA→scale lookup to df_seq.

    Float64 throughout to match the value source consumed by the seq-mode
    path's scale_matrix (full-precision per-residue values).
    """
    aa_to_idx = {a: i for i, a in enumerate(df_scales.index)}
    n_aa = len(aa_to_idx)
    scale_matrix = np.full((n_aa + 1, df_scales.shape[1]), np.nan, dtype=np.float64)
    for col_idx, scale in enumerate(df_scales.columns):
        for a, idx in aa_to_idx.items():
            scale_matrix[idx, col_idx] = df_scales[scale][a]
    out = {}
    for _, row in df_seq.iterrows():
        seq = row["sequence"]
        idxs = np.array([aa_to_idx.get(c, n_aa) for c in seq], dtype=np.int64)
        out[row["entry"]] = scale_matrix[idxs, :]
    return out


class TestRunNumRequiresDictNumParts:
    """``run_num`` rejects calls without ``dict_num_parts`` — use ``run`` for seq-mode."""

    def test_none_raises(self):
        df_seq, labels, df_parts, df_scales = _build_fixture()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        with pytest.raises(ValueError, match="'dict_num_parts' .* required"):
            cpp.run_num(dict_num_parts=None, labels=labels, n_jobs=1)

    def test_missing_raises(self):
        df_seq, labels, df_parts, df_scales = _build_fixture()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        with pytest.raises(ValueError, match="'dict_num_parts' .* required"):
            cpp.run_num(labels=labels, n_jobs=1)


class TestRunNumRoundTrip:
    """Round-trip: dict_num derived from df_scales lookup ≈ seq-mode run."""

    def test_round_trip_tolerance(self):
        """Statistical columns should match within tolerance.

        Bit-exact parity is not guaranteed: the numerical recompute path
        (``recompute_feature_matrix``) uses vectorized nanmean over
        NaN-padded buffers, which can drift at ULP from per-sample
        ``np.mean`` used in the seq-mode path. Drift cascades into
        Mann-Whitney rank order. We assert numerical closeness on shared
        features rather than bit-equality on the row order.
        """
        df_seq, labels, df_parts, df_scales = _build_fixture()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        dict_num = _build_dict_num_from_scales(df_seq, df_scales)
        nf = aa.NumericalFeature()
        _, dict_num_parts = nf.get_parts(df_seq=df_seq, dict_num=dict_num)

        df_seq_mode = cpp.run(labels=labels, n_jobs=1)
        df_num_mode = cpp.run_num(dict_num_parts=dict_num_parts,
                                  labels=labels, n_jobs=1)

        # Both should return the same number of features and same schema.
        assert df_seq_mode.shape == df_num_mode.shape
        assert list(df_seq_mode.columns) == list(df_num_mode.columns)

        # Most features overlap (>= 90% by feature ID); a few near-tie
        # features in the redundancy filter can swap order.
        shared = set(df_seq_mode["feature"]) & set(df_num_mode["feature"])
        assert len(shared) >= int(0.9 * len(df_seq_mode))

        # For the shared features, numerical columns match to within tolerance.
        common_seq = df_seq_mode[df_seq_mode["feature"].isin(shared)].set_index("feature")
        common_num = df_num_mode[df_num_mode["feature"].isin(shared)].set_index("feature")
        common_num = common_num.reindex(common_seq.index)
        for col in ["abs_auc", "abs_mean_dif", "mean_dif", "std_test", "std_ref"]:
            np.testing.assert_allclose(common_seq[col].astype(float).values,
                                       common_num[col].astype(float).values,
                                       rtol=1e-3, atol=1e-3,
                                       err_msg=f"Round-trip mismatch in column {col!r}")


class TestRunNumDeterminism:
    """Same input → same output, within the numerical-mode path itself."""

    def test_two_calls_match(self):
        df_seq, labels, df_parts, df_scales = _build_fixture()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        dict_num = _build_dict_num_from_scales(df_seq, df_scales)
        nf = aa.NumericalFeature()
        _, dict_num_parts = nf.get_parts(df_seq=df_seq, dict_num=dict_num)

        df1 = cpp.run_num(dict_num_parts=dict_num_parts, labels=labels, n_jobs=1)
        df2 = cpp.run_num(dict_num_parts=dict_num_parts, labels=labels, n_jobs=1)
        pd.testing.assert_frame_equal(df1, df2, check_exact=True)


class TestRunNumValidation:
    """Layer-2 validation: dict_num_parts must align with self.df_parts."""

    def test_wrong_part_names_raises(self):
        df_seq, labels, df_parts, df_scales = _build_fixture()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        dict_num = _build_dict_num_from_scales(df_seq, df_scales)
        nf = aa.NumericalFeature()
        _, good_parts = nf.get_parts(df_seq=df_seq, dict_num=dict_num)
        bad_parts = {f"renamed_{k}": v for k, v in good_parts.items()}
        with pytest.raises(ValueError, match="part names"):
            cpp.run_num(dict_num_parts=bad_parts, labels=labels, n_jobs=1)

    def test_wrong_n_samples_raises(self):
        df_seq, labels, df_parts, df_scales = _build_fixture()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        dict_num = _build_dict_num_from_scales(df_seq, df_scales)
        nf = aa.NumericalFeature()
        _, good_parts = nf.get_parts(df_seq=df_seq, dict_num=dict_num)
        # Truncate to half the samples.
        bad_parts = {k: v[: len(v) // 2] for k, v in good_parts.items()}
        with pytest.raises(ValueError, match="n_samples"):
            cpp.run_num(dict_num_parts=bad_parts, labels=labels, n_jobs=1)

    def test_inconsistent_D_raises(self):
        df_seq, labels, df_parts, df_scales = _build_fixture()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        dict_num = _build_dict_num_from_scales(df_seq, df_scales)
        nf = aa.NumericalFeature()
        _, good_parts = nf.get_parts(df_seq=df_seq, dict_num=dict_num)
        # Mutate one part to a different D.
        keys = list(good_parts.keys())
        good_parts[keys[0]] = good_parts[keys[0]][:, :, :3]
        with pytest.raises(ValueError, match="inconsistent D"):
            cpp.run_num(dict_num_parts=good_parts, labels=labels, n_jobs=1)

    def test_D_mismatch_df_scales_raises(self):
        df_seq, labels, df_parts, df_scales = _build_fixture(n_scales=10)
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)  # D=10
        # Build dict_num with D=5 (different from df_scales).
        dict_num = {row["entry"]: np.zeros((len(row["sequence"]), 5))
                    for _, row in df_seq.iterrows()}
        nf = aa.NumericalFeature()
        _, bad_parts = nf.get_parts(df_seq=df_seq, dict_num=dict_num)
        with pytest.raises(ValueError, match="should equal len\\(self\\.df_scales\\.columns\\)"):
            cpp.run_num(dict_num_parts=bad_parts, labels=labels, n_jobs=1)

    def test_dict_input_type_raises(self):
        df_seq, labels, df_parts, df_scales = _build_fixture()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        with pytest.raises(ValueError, match="should be a"):
            cpp.run_num(dict_num_parts="not-a-dict", labels=labels, n_jobs=1)

    def test_wrong_ndim_raises(self):
        df_seq, labels, df_parts, df_scales = _build_fixture()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        bad = {p: np.zeros((10, 20)) for p in df_parts.columns}  # 2D not 3D
        with pytest.raises(ValueError, match="ndim=2"):
            cpp.run_num(dict_num_parts=bad, labels=labels, n_jobs=1)

    def test_n_batches_raises(self):
        df_seq, labels, df_parts, df_scales = _build_fixture()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        dict_num = _build_dict_num_from_scales(df_seq, df_scales)
        nf = aa.NumericalFeature()
        _, parts = nf.get_parts(df_seq=df_seq, dict_num=dict_num)
        with pytest.raises(NotImplementedError, match="'n_batches' is not yet supported"):
            cpp.run_num(dict_num_parts=parts, labels=labels, n_jobs=1, n_batches=2)
