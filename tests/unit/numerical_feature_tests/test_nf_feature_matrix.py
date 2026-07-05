"""Tests for NumericalFeature.feature_matrix().

``nf.feature_matrix(features, dict_num_parts, df_scales=...)`` reconstructs the
``(n_samples, n_features)`` model matrix ``X`` from the per-part numerical tensors
``dict_num_parts`` (numerical analog of ``SequenceFeature.feature_matrix``). The
values are byte-identical to those ``CPP.run_num`` computes for the same feature
ids — the golden-value tests below pin that against ``recompute_feature_matrix``
(the exact backend ``run_num`` uses) and against hand-computed means.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.feature_engineering._backend.cpp._filters._recompute import recompute_feature_matrix
from aaanalysis.feature_engineering._cpp import _derive_dict_part_lens

aa.options["verbose"] = False
warnings.filterwarnings("ignore")


# I Helper functions
def _name_dims(D):
    """Synthetic df_scales / df_cat naming D embedding dimensions."""
    dim_names = [f"dim{i}" for i in range(D)]
    df_scales = pd.DataFrame(np.zeros((20, D)), index=list("ACDEFGHIKLMNPQRSTVWY"), columns=dim_names)
    df_cat = pd.DataFrame({ut.COL_SCALE_ID: dim_names, ut.COL_CAT: ["Emb"] * D,
                           ut.COL_SUBCAT: [f"b{i // 2}" for i in range(D)],
                           ut.COL_SCALE_NAME: dim_names, ut.COL_SCALE_DES: dim_names})
    return df_scales, df_cat


def _run_num_fixture(n=8, D=6, seed=0, stops=None):
    """Deterministic (df_parts, dict_num_parts, df_scales, df_cat, features) via get_parts + run_num."""
    rng = np.random.default_rng(seed)
    seqs = ["ACDEFGHIKLMNPQRSTVWY" * 3] * n            # length 60
    if stops is None:
        stops = [50] * n
    df_seq = pd.DataFrame({"entry": [f"P{i}" for i in range(n)], "sequence": seqs,
                           "tmd_start": [11] * n, "tmd_stop": stops})
    dict_num = {e: rng.random((60, D)) for e in df_seq["entry"]}
    labels = [1] * (n // 2) + [0] * (n - n // 2)
    df_scales, df_cat = _name_dims(D)
    nf = aa.NumericalFeature()
    df_parts, dict_num_parts = nf.get_parts(df_seq=df_seq, dict_num=dict_num)
    cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, df_cat=df_cat, verbose=False)
    df_feat = cpp.run_num(dict_num_parts=dict_num_parts, labels=labels, n_filter=15, n_jobs=1)
    return dict(df_parts=df_parts, dict_num_parts=dict_num_parts, df_scales=df_scales,
               df_cat=df_cat, df_feat=df_feat, features=df_feat["feature"].to_list(),
               cpp=cpp, labels=labels)


# II Per-parameter tests
class TestFeatureMatrix:
    """Per-parameter positive and negative coverage."""

    # features
    def test_valid_features_list(self):
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        X = nf.feature_matrix(features=fx["features"], dict_num_parts=fx["dict_num_parts"], df_parts=fx["df_parts"],
                              df_scales=fx["df_scales"])
        assert X.shape == (len(fx["df_parts"]), len(fx["features"]))

    def test_valid_features_df_feat(self):
        """A df_feat DataFrame is accepted: its 'feature' column is used."""
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        X_ids = nf.feature_matrix(features=fx["features"], dict_num_parts=fx["dict_num_parts"], df_parts=fx["df_parts"],
                                  df_scales=fx["df_scales"])
        X_df = nf.feature_matrix(features=fx["df_feat"], dict_num_parts=fx["dict_num_parts"], df_parts=fx["df_parts"],
                                 df_scales=fx["df_scales"])
        assert np.array_equal(X_ids, X_df)

    def test_valid_features_single_str(self):
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        X = nf.feature_matrix(features=fx["features"][0], dict_num_parts=fx["dict_num_parts"], df_parts=fx["df_parts"],
                              df_scales=fx["df_scales"])
        assert X.shape == (len(fx["df_parts"]), 1)

    def test_invalid_features_empty(self):
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        with pytest.raises(ValueError):
            nf.feature_matrix(features=[], dict_num_parts=fx["dict_num_parts"], df_parts=fx["df_parts"], df_scales=fx["df_scales"])

    def test_invalid_features_malformed(self):
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        with pytest.raises(ValueError):
            nf.feature_matrix(features=["TMD-Segment(1,2)"], dict_num_parts=fx["dict_num_parts"], df_parts=fx["df_parts"],
                              df_scales=fx["df_scales"])

    def test_invalid_features_unknown_part(self):
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        with pytest.raises(ValueError):
            nf.feature_matrix(features=["NOPART-Segment(1,2)-dim0"], dict_num_parts=fx["dict_num_parts"], df_parts=fx["df_parts"],
                              df_scales=fx["df_scales"])

    def test_invalid_features_unknown_scale(self):
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        with pytest.raises(ValueError):
            nf.feature_matrix(features=["TMD-Segment(1,2)-nope"], dict_num_parts=fx["dict_num_parts"], df_parts=fx["df_parts"],
                              df_scales=fx["df_scales"])

    # dict_num_parts
    def test_valid_dict_num_parts(self):
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        X = nf.feature_matrix(features=fx["features"], dict_num_parts=fx["dict_num_parts"], df_parts=fx["df_parts"],
                              df_scales=fx["df_scales"])
        assert np.isfinite(X).all()

    def test_invalid_dict_num_parts_none(self):
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        with pytest.raises(ValueError):
            nf.feature_matrix(features=fx["features"], dict_num_parts=None, df_parts=fx["df_parts"],
                              df_scales=fx["df_scales"])

    def test_invalid_dict_num_parts_empty(self):
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        with pytest.raises(ValueError):
            nf.feature_matrix(features=fx["features"], dict_num_parts={}, df_parts=fx["df_parts"],
                              df_scales=fx["df_scales"])

    def test_invalid_dict_num_parts_2d(self):
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        bad = {p: a[:, :, 0] for p, a in fx["dict_num_parts"].items()}  # drop D axis -> 2D
        with pytest.raises(ValueError):
            nf.feature_matrix(features=fx["features"], dict_num_parts=bad, df_parts=fx["df_parts"], df_scales=fx["df_scales"])

    def test_invalid_dict_num_parts_inconsistent_n(self):
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        bad = dict(fx["dict_num_parts"])
        first = list(bad)[0]
        bad[first] = bad[first][:-1]  # different n_samples
        with pytest.raises(ValueError):
            nf.feature_matrix(features=fx["features"], dict_num_parts=bad, df_parts=fx["df_parts"], df_scales=fx["df_scales"])

    def test_invalid_dict_num_parts_inconsistent_D(self):
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        bad = dict(fx["dict_num_parts"])
        first = list(bad)[0]
        bad[first] = bad[first][:, :, :-1]  # different D
        with pytest.raises(ValueError):
            nf.feature_matrix(features=fx["features"], dict_num_parts=bad, df_parts=fx["df_parts"], df_scales=fx["df_scales"])

    # df_parts
    def test_valid_df_parts_lengths_drive_splits(self):
        """df_parts supplies the real lengths; passing get_parts' df_parts yields finite X."""
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        X = nf.feature_matrix(features=fx["features"], dict_num_parts=fx["dict_num_parts"],
                              df_parts=fx["df_parts"], df_scales=fx["df_scales"])
        assert np.isfinite(X).all()

    def test_invalid_df_parts_row_mismatch(self):
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        with pytest.raises(ValueError):
            nf.feature_matrix(features=fx["features"], dict_num_parts=fx["dict_num_parts"],
                              df_parts=fx["df_parts"].iloc[:-1], df_scales=fx["df_scales"])

    def test_invalid_df_parts_missing_part_column(self):
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        part0 = list(fx["dict_num_parts"])[0]
        with pytest.raises(ValueError):
            nf.feature_matrix(features=fx["features"], dict_num_parts=fx["dict_num_parts"],
                              df_parts=fx["df_parts"].drop(columns=[part0]), df_scales=fx["df_scales"])

    def test_invalid_df_parts_length_exceeds_tensor(self):
        """A df_parts real length beyond the padded tensor length is a get_parts mismatch."""
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        part0 = list(fx["dict_num_parts"])[0]
        l_max = fx["dict_num_parts"][part0].shape[1]
        bad = fx["df_parts"].copy()
        bad[part0] = ["A" * (l_max + 1)] * len(bad)   # real length exceeds tensor L_part_max
        with pytest.raises(ValueError):
            nf.feature_matrix(features=fx["features"], dict_num_parts=fx["dict_num_parts"],
                              df_parts=bad, df_scales=fx["df_scales"])

    # df_scales
    def test_valid_df_scales_custom(self):
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        X = nf.feature_matrix(features=fx["features"], dict_num_parts=fx["dict_num_parts"], df_parts=fx["df_parts"],
                              df_scales=fx["df_scales"])
        assert X.shape[1] == len(fx["features"])

    def test_valid_df_scales_default_when_D_matches(self):
        """Omitting df_scales falls back to the bundled scales; works only if D matches."""
        df_default = ut.load_default_scales()
        D = len(df_default.columns)
        fx = _run_num_fixture(D=D)
        # Feature scales come from our synthetic dim-names, not the default columns,
        # so features must reference default scale ids for the default path. Build a
        # trivial feature on a default scale id instead.
        scale0 = list(df_default.columns)[0]
        feat = f"TMD-Segment(1,1)-{scale0}"
        nf = aa.NumericalFeature()
        X = nf.feature_matrix(features=[feat], dict_num_parts=fx["dict_num_parts"], df_parts=fx["df_parts"])
        assert X.shape == (len(fx["df_parts"]), 1)

    def test_invalid_df_scales_D_mismatch(self):
        fx = _run_num_fixture(D=6)
        nf = aa.NumericalFeature()
        with pytest.raises(ValueError):
            nf.feature_matrix(features=fx["features"], dict_num_parts=fx["dict_num_parts"], df_parts=fx["df_parts"],
                              df_scales=fx["df_scales"].iloc[:, :4])  # D=4 != 6

    def test_invalid_df_scales_nan(self):
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        bad = fx["df_scales"].copy()
        bad.iloc[0, 0] = np.nan
        with pytest.raises(ValueError):
            nf.feature_matrix(features=fx["features"], dict_num_parts=fx["dict_num_parts"], df_parts=fx["df_parts"], df_scales=bad)

    # n_jobs
    def test_valid_n_jobs(self):
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        X_ref = nf.feature_matrix(features=fx["features"], dict_num_parts=fx["dict_num_parts"], df_parts=fx["df_parts"],
                                  df_scales=fx["df_scales"], n_jobs=1)
        for n_jobs in [1, -1, None]:
            X = nf.feature_matrix(features=fx["features"], dict_num_parts=fx["dict_num_parts"], df_parts=fx["df_parts"],
                                  df_scales=fx["df_scales"], n_jobs=n_jobs)
            assert np.array_equal(X, X_ref), f"n_jobs={n_jobs} changed the result"

    def test_invalid_n_jobs(self):
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        with pytest.raises(ValueError):
            nf.feature_matrix(features=fx["features"], dict_num_parts=fx["dict_num_parts"], df_parts=fx["df_parts"],
                              df_scales=fx["df_scales"], n_jobs=0)


# III Complex / contract tests
class TestFeatureMatrixComplex:
    """Shape, dtype, column ordering, and cross-form consistency."""

    def test_output_shape_and_dtype(self):
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        X = nf.feature_matrix(features=fx["features"], dict_num_parts=fx["dict_num_parts"], df_parts=fx["df_parts"],
                              df_scales=fx["df_scales"])
        assert isinstance(X, np.ndarray)
        assert X.dtype == np.float64
        assert X.shape == (len(fx["df_parts"]), len(fx["features"]))

    def test_column_order_matches_features(self):
        """Reordering the feature list permutes X's columns the same way."""
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        feats = fx["features"]
        X = nf.feature_matrix(features=feats, dict_num_parts=fx["dict_num_parts"], df_parts=fx["df_parts"], df_scales=fx["df_scales"])
        perm = list(reversed(range(len(feats))))
        feats_perm = [feats[i] for i in perm]
        X_perm = nf.feature_matrix(features=feats_perm, dict_num_parts=fx["dict_num_parts"], df_parts=fx["df_parts"],
                                   df_scales=fx["df_scales"])
        assert np.array_equal(X_perm, X[:, perm])

    def test_variable_tmd_length(self):
        """Ragged (variable-TMD) parts are handled and stay run_num-consistent."""
        fx = _run_num_fixture(stops=[50, 50, 50, 50, 45, 45, 45, 45])
        nf = aa.NumericalFeature()
        X = nf.feature_matrix(features=fx["features"], dict_num_parts=fx["dict_num_parts"], df_parts=fx["df_parts"],
                              df_scales=fx["df_scales"])
        X_ref = recompute_feature_matrix(
            dict_part_vals=fx["dict_num_parts"],
            dict_part_lens=_derive_dict_part_lens(df_parts=fx["df_parts"]),
            list_scales=list(fx["df_scales"].columns), features=fx["features"],
            split_kws=fx["cpp"].split_kws,
        )
        assert np.array_equal(X, X_ref, equal_nan=True)


# IV Golden-value tests
class TestFeatureMatrixGoldenValues:
    """Pin exact values against hand computation and run_num's own backend."""

    def test_consistency_with_run_num_recompute(self):
        """feature_matrix == the exact matrix run_num built for the same feature ids."""
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        X = nf.feature_matrix(features=fx["features"], dict_num_parts=fx["dict_num_parts"], df_parts=fx["df_parts"],
                              df_scales=fx["df_scales"])
        X_ref = recompute_feature_matrix(
            dict_part_vals=fx["dict_num_parts"],
            dict_part_lens=_derive_dict_part_lens(df_parts=fx["df_parts"]),
            list_scales=list(fx["df_scales"].columns), features=fx["features"],
            split_kws=fx["cpp"].split_kws,
        )
        assert np.array_equal(X, X_ref, equal_nan=True)

    def test_all_nan_real_residue_matches_run_num(self):
        """A genuine (non-padding) residue that is all-NaN across D still matches run_num.

        The length is taken from the string ``df_parts`` (as run_num does), not inferred
        from the tensor's NaN pattern, so the split boundaries do not shift. Uses whole /
        half-part segments that span the blanked residue plus others, so ``nanmean`` stays
        finite (a split selecting only the blanked residue would legitimately raise).
        """
        fx = _run_num_fixture()
        nf = aa.NumericalFeature()
        dict_num_parts = dict(fx["dict_num_parts"])
        tmd = dict_num_parts["tmd"].copy()
        tmd[:, 5, :] = np.nan          # blank a real interior TMD residue for every sample
        dict_num_parts["tmd"] = tmd
        feats = ["TMD-Segment(1,1)-dim0", "TMD-Segment(1,2)-dim1"]
        X = nf.feature_matrix(features=feats, dict_num_parts=dict_num_parts,
                              df_parts=fx["df_parts"], df_scales=fx["df_scales"])
        X_ref = recompute_feature_matrix(
            dict_part_vals=dict_num_parts,
            dict_part_lens=_derive_dict_part_lens(df_parts=fx["df_parts"]),
            list_scales=list(fx["df_scales"].columns), features=feats,
            split_kws=fx["cpp"].split_kws,
        )
        assert np.array_equal(X, X_ref, equal_nan=True)

    def test_golden_whole_part_segment(self):
        """Segment(1,1) = whole-part mean of the chosen dimension (hand-computed)."""
        n, L, D = 4, 10, 3
        rng = np.random.default_rng(7)
        arr = rng.random((n, L, D))
        dict_num_parts = {"tmd": arr}
        df_parts = pd.DataFrame({"tmd": ["A" * L] * n})  # real length L (dense, no padding)
        df_scales, _ = _name_dims(D)
        nf = aa.NumericalFeature()
        feat = "TMD-Segment(1,1)-dim2"
        X = nf.feature_matrix(features=[feat], dict_num_parts=dict_num_parts, df_parts=df_parts, df_scales=df_scales)
        expected = np.round(arr[:, :, 2].mean(axis=1), 5)
        assert np.allclose(X[:, 0], expected)

    def test_golden_first_half_segment(self):
        """Segment(1,2) = mean over the first half of the part (int floor split)."""
        n, L, D = 4, 10, 3
        rng = np.random.default_rng(8)
        arr = rng.random((n, L, D))
        dict_num_parts = {"tmd": arr}
        df_parts = pd.DataFrame({"tmd": ["A" * L] * n})  # real length L (dense, no padding)
        df_scales, _ = _name_dims(D)
        nf = aa.NumericalFeature()
        feat = "TMD-Segment(1,2)-dim0"
        X = nf.feature_matrix(features=[feat], dict_num_parts=dict_num_parts, df_parts=df_parts, df_scales=df_scales)
        expected = np.round(arr[:, 0:5, 0].mean(axis=1), 5)  # first half of L=10
        assert np.allclose(X[:, 0], expected)

    def test_golden_pattern_positions(self):
        """Pattern(N,1,3) selects 0-based residues 0 and 2 (list_pos - 1)."""
        n, L, D = 4, 10, 3
        rng = np.random.default_rng(9)
        arr = rng.random((n, L, D))
        dict_num_parts = {"tmd": arr}
        df_parts = pd.DataFrame({"tmd": ["A" * L] * n})  # real length L (dense, no padding)
        df_scales, _ = _name_dims(D)
        nf = aa.NumericalFeature()
        feat = "TMD-Pattern(N,1,3)-dim1"
        X = nf.feature_matrix(features=[feat], dict_num_parts=dict_num_parts, df_parts=df_parts, df_scales=df_scales)
        expected = np.round(arr[:, [0, 2], 1].mean(axis=1), 5)
        assert np.allclose(X[:, 0], expected)

    def test_nan_split_raises(self):
        """A split that selects only all-NaN residues yields a NaN value and raises a clear ValueError."""
        n, L, D = 3, 6, 2
        arr = np.full((n, L, D), np.nan)   # every selected residue is NaN
        dict_num_parts = {"tmd": arr}
        df_parts = pd.DataFrame({"tmd": ["A" * L] * n})  # real length L (dense, no padding)
        df_scales, _ = _name_dims(D)
        nf = aa.NumericalFeature()
        with pytest.raises(ValueError):
            nf.feature_matrix(features=["TMD-Segment(1,1)-dim0"], dict_num_parts=dict_num_parts,
                              df_parts=df_parts, df_scales=df_scales)
