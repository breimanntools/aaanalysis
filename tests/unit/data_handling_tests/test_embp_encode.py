"""This is a script to test EmbeddingPreprocessor.encode()."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa

aa.options["verbose"] = False

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


# Helpers --------------------------------------------------------------
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _make_fixture(n=5, D=8, seed=0):
    """Build a deterministic (df_seq, embeddings) fixture with raw (unbounded) values."""
    rng = np.random.default_rng(seed)
    seqs = ["".join(rng.choice(list(ALPHABET), size=10 + (i % 5))) for i in range(n)]
    df_seq = pd.DataFrame({"entry": [f"P{i}" for i in range(n)], "sequence": seqs})
    embeddings = {f"P{i}": (rng.standard_normal((len(seqs[i]), D)) * 10.0 + 3.0)
                  for i in range(n)}
    return df_seq, embeddings


# Normal cases ---------------------------------------------------------
class TestEncode:
    """Positive and parameter-level negative tests for encode."""

    @given(n=some.integers(min_value=2, max_value=6))
    def test_returns_dict_keyed_by_entry(self, n):
        df_seq, embeddings = _make_fixture(n=n)
        dict_num = aa.EmbeddingPreprocessor().encode(df_seq=df_seq, embeddings=embeddings)
        assert set(dict_num) == set(df_seq["entry"])

    @given(D=some.integers(min_value=1, max_value=12))
    def test_preserves_shape(self, D):
        df_seq, embeddings = _make_fixture(D=D)
        dict_num = aa.EmbeddingPreprocessor().encode(df_seq=df_seq, embeddings=embeddings)
        for e, arr in dict_num.items():
            assert arr.shape == embeddings[e].shape

    @given(seed=some.integers(min_value=0, max_value=50))
    def test_minmax_in_unit_range(self, seed):
        df_seq, embeddings = _make_fixture(seed=seed)
        dict_num = aa.EmbeddingPreprocessor().encode(df_seq=df_seq, embeddings=embeddings,
                                                     method="minmax")
        allv = np.concatenate([a.ravel() for a in dict_num.values()])
        assert allv.min() >= -1e-9 and allv.max() <= 1 + 1e-9

    def test_quantile_in_unit_range(self):
        df_seq, embeddings = _make_fixture()
        dict_num = aa.EmbeddingPreprocessor().encode(df_seq=df_seq, embeddings=embeddings,
                                                     method="quantile")
        allv = np.concatenate([a.ravel() for a in dict_num.values()])
        assert allv.min() >= -1e-9 and allv.max() <= 1 + 1e-9

    def test_sigmoid_in_open_unit_range(self):
        df_seq, embeddings = _make_fixture()
        dict_num = aa.EmbeddingPreprocessor().encode(df_seq=df_seq, embeddings=embeddings,
                                                     method="sigmoid")
        allv = np.concatenate([a.ravel() for a in dict_num.values()])
        assert allv.min() >= 0.0 and allv.max() <= 1.0

    def test_stores_norm_params(self):
        df_seq, embeddings = _make_fixture()
        embp = aa.EmbeddingPreprocessor()
        embp.encode(df_seq=df_seq, embeddings=embeddings)
        assert isinstance(embp.norm_params_, dict) and embp.norm_params_["method"] == "minmax"

    def test_return_df_returns_tuple(self):
        df_seq, embeddings = _make_fixture()
        out = aa.EmbeddingPreprocessor().encode(df_seq=df_seq, embeddings=embeddings,
                                                return_df=True)
        assert isinstance(out, tuple) and len(out) == 2
        dict_num, df_out = out
        assert isinstance(dict_num, dict) and isinstance(df_out, pd.DataFrame)

    def test_deterministic(self):
        df_seq, embeddings = _make_fixture()
        a = aa.EmbeddingPreprocessor().encode(df_seq=df_seq, embeddings=embeddings)
        b = aa.EmbeddingPreprocessor().encode(df_seq=df_seq, embeddings=embeddings)
        for e in a:
            assert np.allclose(a[e], b[e])

    def test_constant_dimension_maps_to_zero(self):
        df_seq, embeddings = _make_fixture()
        for e in embeddings:
            embeddings[e][:, 0] = 5.0  # constant dim
        dict_num = aa.EmbeddingPreprocessor().encode(df_seq=df_seq, embeddings=embeddings)
        assert all(np.allclose(a[:, 0], 0.0) for a in dict_num.values())

    # Negative tests
    def test_invalid_df_seq_none(self):
        _, embeddings = _make_fixture()
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor().encode(df_seq=None, embeddings=embeddings)

    def test_invalid_embeddings_none(self):
        df_seq, _ = _make_fixture()
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor().encode(df_seq=df_seq, embeddings=None)

    def test_invalid_missing_entry(self):
        df_seq, embeddings = _make_fixture()
        embeddings.pop("P0")
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor().encode(df_seq=df_seq, embeddings=embeddings)

    def test_invalid_length_mismatch(self):
        df_seq, embeddings = _make_fixture()
        embeddings["P0"] = embeddings["P0"][:-1]  # wrong L
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor().encode(df_seq=df_seq, embeddings=embeddings)

    def test_invalid_inconsistent_D(self):
        df_seq, embeddings = _make_fixture()
        embeddings["P0"] = np.random.default_rng(0).standard_normal((len(df_seq["sequence"][0]), 3))
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor().encode(df_seq=df_seq, embeddings=embeddings)

    def test_invalid_method(self):
        df_seq, embeddings = _make_fixture()
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor().encode(df_seq=df_seq, embeddings=embeddings, method="zscore")

    def test_invalid_clip_order(self):
        df_seq, embeddings = _make_fixture()
        with pytest.raises(ValueError):
            aa.EmbeddingPreprocessor().encode(df_seq=df_seq, embeddings=embeddings,
                                              method="quantile", clip=(99.0, 1.0))


# Combinations ---------------------------------------------------------
class TestEncodeComplex:
    """Combinations and edge interactions for encode."""

    def test_encode_then_run_num_pipeline(self):
        seqs = ["ACDEFGHIKLMNPQRSTVWY" * 3] * 4   # length 60
        df_seq = pd.DataFrame({"entry": [f"P{i}" for i in range(4)], "sequence": seqs,
                               "tmd_start": 11, "tmd_stop": 50})
        rng = np.random.default_rng(0)
        embeddings = {e: rng.standard_normal((60, 4)) * 5.0 for e in df_seq["entry"]}
        embp = aa.EmbeddingPreprocessor()
        dict_num = embp.encode(df_seq=df_seq, embeddings=embeddings)
        df_parts, dict_num_parts = aa.NumericalFeature().get_parts(df_seq=df_seq, dict_num=dict_num)
        # get_parts returns df_parts (one row per entry) and dict_num_parts keyed by part name.
        assert len(df_parts) == len(df_seq) and len(dict_num_parts) >= 1

    def test_encode_then_build_scales(self):
        df_seq, embeddings = _make_fixture(n=6, D=6)
        embp = aa.EmbeddingPreprocessor()
        dict_num = embp.encode(df_seq=df_seq, embeddings=embeddings)
        df_scales = embp.build_scales(df_seq=df_seq, dict_num=dict_num)
        assert df_scales.shape[1] == 6

    def test_methods_agree_on_shape(self):
        df_seq, embeddings = _make_fixture()
        embp = aa.EmbeddingPreprocessor()
        shapes = []
        for m in ("minmax", "quantile", "sigmoid"):
            dn = embp.encode(df_seq=df_seq, embeddings=embeddings, method=m)
            shapes.append({e: a.shape for e, a in dn.items()})
        assert shapes[0] == shapes[1] == shapes[2]

    def test_quantile_clip_robust_to_outlier(self):
        df_seq, embeddings = _make_fixture()
        embeddings["P0"][0, 0] = 1e6  # extreme outlier
        dn = aa.EmbeddingPreprocessor().encode(df_seq=df_seq, embeddings=embeddings,
                                               method="quantile", clip=(1, 99))
        allv = np.concatenate([a.ravel() for a in dn.values()])
        assert allv.max() <= 1 + 1e-9

    def test_single_protein(self):
        df_seq = pd.DataFrame({"entry": ["P0"], "sequence": ["ACDEFGHIKL"]})
        embeddings = {"P0": np.random.default_rng(0).standard_normal((10, 4))}
        dn = aa.EmbeddingPreprocessor().encode(df_seq=df_seq, embeddings=embeddings)
        assert dn["P0"].shape == (10, 4)

    def test_return_df_echoes_entries(self):
        df_seq, embeddings = _make_fixture()
        _, df_out = aa.EmbeddingPreprocessor().encode(df_seq=df_seq, embeddings=embeddings,
                                                      return_df=True)
        assert df_out["entry"].tolist() == df_seq["entry"].tolist()
