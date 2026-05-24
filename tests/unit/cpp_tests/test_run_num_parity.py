"""This is a script to test CPP.run_num() parity with CPP.run() and dict_num round-trip.

Seq-only parity (dict_num=None): bit-identical to CPP.run() over the same inputs.
Dict_num round-trip: when dict_num is built from the scale_matrix lookup applied to
df_seq, run_num(dict_num=...) must produce a df_feat bit-identical to
run_num(dict_num=None, same scales). This verifies the dict_num assign path
correctly mirrors the seq-mode assign path.
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


def _assert_parity(cpp, df_seq, labels, **kwargs):
    """Run both paths with the same kwargs and assert bit-EXACT identical df_feat.

    ``check_exact=True`` — every numerical column must match byte-for-byte,
    including ``p_val_mann_whitney`` (which is the most sensitive to
    summation-order ULP drift).
    """
    df_feat_old = cpp.run(labels=labels, n_jobs=1, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df_feat_new = cpp.run_num(df_seq=df_seq, dict_num=None, labels=labels,
                                  n_jobs=1, **kwargs)
    pd.testing.assert_frame_equal(df_feat_old, df_feat_new, check_exact=True)


def _build_dict_num_from_scales(df_seq, df_scales):
    """Build ``dict_num: Dict[entry, (L, D)]`` by applying the AA→scale lookup to df_seq.

    Uses ``float32`` scale_matrix (matching legacy ``_filters/_assign.py``'s dtype)
    so that seq-mode and dict_num-mode produce identical per-residue values
    after float32→float64 promotion downstream.
    """
    aa_to_idx = {a: i for i, a in enumerate(df_scales.index)}
    n_aa = len(aa_to_idx)
    scale_matrix = np.full((n_aa + 1, df_scales.shape[1]), np.nan, dtype=np.float32)
    for col_idx, scale in enumerate(df_scales.columns):
        for a, idx in aa_to_idx.items():
            scale_matrix[idx, col_idx] = df_scales[scale][a]
    out = {}
    for _, row in df_seq.iterrows():
        seq = row["sequence"]
        idxs = np.array(
            [aa_to_idx.get(c, n_aa) for c in seq], dtype=np.int64
        )
        out[row["entry"]] = scale_matrix[idxs, :]
    return out


class TestRunNumParity:
    """Bit-identical parity between CPP.run and CPP.run_num(dict_num=None)."""

    def test_defaults(self):
        df_seq, labels, df_parts, df_scales = _build_fixture()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        _assert_parity(cpp, df_seq, labels)

    def test_parametric(self):
        df_seq, labels, df_parts, df_scales = _build_fixture()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        _assert_parity(cpp, df_seq, labels, parametric=True)

    def test_check_cat_false(self):
        df_seq, labels, df_parts, df_scales = _build_fixture()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        _assert_parity(cpp, df_seq, labels, check_cat=False)

    def test_n_batches(self):
        df_seq, labels, df_parts, df_scales = _build_fixture(n_scales=10)
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        _assert_parity(cpp, df_seq, labels, n_batches=2)

    def test_n_sample_batches_removed(self):
        """``n_sample_batches`` is removed — accumulator variance is not bit-exact."""
        df_seq, labels, df_parts, df_scales = _build_fixture()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(NotImplementedError, match="n_sample_batches"):
                cpp.run_num(df_seq=df_seq, dict_num=None, labels=labels,
                            n_sample_batches=2, n_jobs=1)


class TestRunNumDictNum:
    """dict_num round-trip: tensor-mode equivalent to seq-mode under the same scales."""

    def test_dict_num_round_trip(self):
        df_seq, labels, df_parts, df_scales = _build_fixture()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        # Build dict_num matching the scale_matrix lookup; seq-mode and tensor-mode
        # produce identical per-residue values for canonical AAs.
        dict_num = _build_dict_num_from_scales(df_seq, df_scales)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            df_seq_mode = cpp.run_num(df_seq=df_seq, dict_num=None, labels=labels, n_jobs=1)
            df_dict_mode = cpp.run_num(df_seq=df_seq, dict_num=dict_num,
                                       df_scales=df_scales, labels=labels, n_jobs=1)
        pd.testing.assert_frame_equal(df_seq_mode, df_dict_mode)


class TestRunNumParityComplex:
    """Negative + edge-case parity checks: mismatch detection and error paths."""

    def test_df_seq_mismatch_raises(self):
        df_seq, labels, df_parts, df_scales = _build_fixture()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        df_seq_shuffled = df_seq.iloc[::-1].reset_index(drop=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(ValueError, match="disagree with the CPP instance"):
                cpp.run_num(df_seq=df_seq_shuffled, dict_num=None, labels=labels, n_jobs=1)

    def test_dict_num_missing_entries_raises(self):
        df_seq, labels, df_parts, df_scales = _build_fixture()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(ValueError, match="missing entries"):
                cpp.run_num(df_seq=df_seq, dict_num={"unknown_entry": np.zeros((10, 5))},
                            df_scales=df_scales, labels=labels, n_jobs=1)

    def test_dict_num_wrong_D_raises(self):
        df_seq, labels, df_parts, df_scales = _build_fixture()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        # Build dict_num with D=3 but df_scales has D=10 → should raise.
        dict_num = {row["entry"]: np.zeros((len(row["sequence"]), 3))
                    for _, row in df_seq.iterrows()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(ValueError, match="D=3 columns"):
                cpp.run_num(df_seq=df_seq, dict_num=dict_num,
                            df_scales=df_scales, labels=labels, n_jobs=1)

    def test_df_scales_kwarg_requires_dict_num(self):
        df_seq, labels, df_parts, df_scales = _build_fixture()
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(ValueError, match="apply only when 'dict_num' is supplied"):
                cpp.run_num(df_seq=df_seq, dict_num=None,
                            df_scales=df_scales, labels=labels, n_jobs=1)
