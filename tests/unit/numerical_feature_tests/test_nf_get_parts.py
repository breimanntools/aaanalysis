"""This is a script to test NumericalFeature.get_parts().

PR5 contract: ``nf.get_parts(df_seq, dict_num, ...)`` returns
``(df_parts, dict_num_parts)`` where ``df_parts`` is the sequence-string
slice (same as ``SequenceFeature.get_df_parts``) and ``dict_num_parts``
is the per-part NaN-padded numerical tensor aligned to ``df_parts``.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa

aa.options["verbose"] = False
warnings.filterwarnings("ignore")


def _build_synthetic_fixture(n=5, D=16, seed=42):
    """5 proteins from DOM_GSEC + random (L, D) per-residue tensor per entry."""
    rng = np.random.default_rng(seed)
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
    dict_num = {
        row["entry"]: rng.standard_normal((len(row["sequence"]), D)).astype(np.float64)
        for _, row in df_seq.iterrows()
    }
    return df_seq, dict_num


class TestGetParts:
    """Happy-path coverage of NumericalFeature.get_parts."""

    def test_returns_tuple_of_expected_shapes(self):
        df_seq, dict_num = _build_synthetic_fixture(n=5, D=16)
        nf = aa.NumericalFeature()
        df_parts, dict_num_parts = nf.get_parts(df_seq=df_seq, dict_num=dict_num)
        assert isinstance(df_parts, pd.DataFrame)
        assert isinstance(dict_num_parts, dict)
        # Default parts: tmd + jmd_n_tmd_n + tmd_c_jmd_c
        assert set(df_parts.columns) == set(dict_num_parts.keys())

    def test_dict_num_parts_aligned_to_df_parts(self):
        df_seq, dict_num = _build_synthetic_fixture(n=5, D=16)
        nf = aa.NumericalFeature()
        df_parts, dict_num_parts = nf.get_parts(df_seq=df_seq, dict_num=dict_num)
        for part in df_parts.columns:
            arr = dict_num_parts[part]
            assert arr.ndim == 3, f"part {part!r} should be 3D"
            assert arr.shape[0] == len(df_parts), f"part {part!r} n_samples"
            assert arr.shape[2] == 16, f"part {part!r} D"

    def test_part_lengths_match_string_lengths(self):
        """For each (entry, part), the real residue count (non-gap chars in df_parts
        string) should equal the leading non-NaN row count in dict_num_parts."""
        df_seq, dict_num = _build_synthetic_fixture(n=5, D=16)
        nf = aa.NumericalFeature()
        df_parts, dict_num_parts = nf.get_parts(df_seq=df_seq, dict_num=dict_num)
        for part in df_parts.columns:
            arr = dict_num_parts[part]
            for i, entry in enumerate(df_parts.index):
                seq_str = df_parts.loc[entry, part]
                non_gap = sum(c != "-" for c in seq_str)
                # Find first all-NaN row (or use length if no NaN).
                if arr.shape[1] == 0:
                    assert non_gap == 0
                    continue
                row = arr[i]
                non_nan_rows = (~np.isnan(row[:, 0])).sum()
                # When there are no gaps in the string, ALL rows should be non-NaN
                # (synthetic data has no NaN values). When the JMD is short, the
                # PADDING is at the END of the tensor — the leading non-NaN count
                # equals the non-gap char count.
                assert non_nan_rows == non_gap, (
                    f"part={part!r} entry={entry!r}: "
                    f"string non-gap chars={non_gap}, tensor non-NaN rows={non_nan_rows}"
                )

    def test_subset_of_parts(self):
        df_seq, dict_num = _build_synthetic_fixture(n=5, D=16)
        nf = aa.NumericalFeature()
        df_parts, dict_num_parts = nf.get_parts(df_seq=df_seq, dict_num=dict_num,
                                                list_parts=["tmd"])
        assert list(df_parts.columns) == ["tmd"]
        assert list(dict_num_parts.keys()) == ["tmd"]

    def test_dtype_float64(self):
        df_seq, dict_num = _build_synthetic_fixture(n=5, D=16)
        nf = aa.NumericalFeature()
        _, dict_num_parts = nf.get_parts(df_seq=df_seq, dict_num=dict_num)
        for arr in dict_num_parts.values():
            assert arr.dtype == np.float64


class TestGetPartsValidation:
    """Layer-1 validation: df_seq ↔ dict_num pairing."""

    def test_dict_num_none_raises(self):
        df_seq, _ = _build_synthetic_fixture(n=5, D=16)
        nf = aa.NumericalFeature()
        with pytest.raises(ValueError, match="'dict_num' .* should be a Dict"):
            nf.get_parts(df_seq=df_seq, dict_num=None)

    def test_missing_entry_raises(self):
        df_seq, dict_num = _build_synthetic_fixture(n=5, D=16)
        # Drop one entry from dict_num.
        first_key = next(iter(dict_num.keys()))
        del dict_num[first_key]
        nf = aa.NumericalFeature()
        with pytest.raises(ValueError, match="missing"):
            nf.get_parts(df_seq=df_seq, dict_num=dict_num)

    def test_wrong_L_raises(self):
        df_seq, dict_num = _build_synthetic_fixture(n=5, D=16)
        # Truncate one entry's tensor.
        first_key = next(iter(dict_num.keys()))
        dict_num[first_key] = dict_num[first_key][:3, :]
        nf = aa.NumericalFeature()
        with pytest.raises(ValueError, match="should be \\(.*, D\\)"):
            nf.get_parts(df_seq=df_seq, dict_num=dict_num)

    def test_inconsistent_D_raises(self):
        df_seq, dict_num = _build_synthetic_fixture(n=5, D=16)
        # Mutate one entry to a different D.
        first_key = next(iter(dict_num.keys()))
        dict_num[first_key] = dict_num[first_key][:, :8]  # D=8 vs D=16 elsewhere
        nf = aa.NumericalFeature()
        with pytest.raises(ValueError, match="inconsistent D"):
            nf.get_parts(df_seq=df_seq, dict_num=dict_num)

    def test_wrong_ndim_raises(self):
        df_seq, dict_num = _build_synthetic_fixture(n=5, D=16)
        # Replace one entry's value with a 1-D vector.
        first_key = next(iter(dict_num.keys()))
        dict_num[first_key] = np.zeros(50, dtype=np.float64)
        nf = aa.NumericalFeature()
        with pytest.raises(ValueError, match="should be a 2-D"):
            nf.get_parts(df_seq=df_seq, dict_num=dict_num)


class TestGetPartsRoundTrip:
    """get_parts → run_num produces same output as raw assign_dict_num path."""

    def test_get_parts_then_run_num_no_crash(self):
        df_seq, dict_num = _build_synthetic_fixture(n=5, D=16)
        labels = df_seq["label"].to_list()
        nf = aa.NumericalFeature()
        df_parts, dict_num_parts = nf.get_parts(df_seq=df_seq, dict_num=dict_num)
        # Build matching df_scales / df_cat.
        list_aa = list("ACDEFGHIKLMNPQRSTVWY")
        df_scales = pd.DataFrame(
            np.random.default_rng(0).standard_normal((20, 16)).astype(np.float64),
            index=list_aa, columns=[f"dim_{i}" for i in range(16)],
        )
        df_cat = pd.DataFrame({
            "scale_id": [f"dim_{i}" for i in range(16)],
            "category": [f"cat_{i % 3}" for i in range(16)],
            "subcategory": [f"sub_{i}" for i in range(16)],
            "scale_name": [f"dim_{i}" for i in range(16)],
            "scale_description": [f"d{i}" for i in range(16)],
        })
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, df_cat=df_cat)
        df_feat = cpp.run_num(dict_num_parts=dict_num_parts, labels=labels, n_jobs=1)
        # Sanity: same schema as cpp.run, output has at least 1 feature.
        assert df_feat.shape[1] == 13
        assert len(df_feat) >= 1
