"""Unit tests for SequenceFeatureTransformer (leak-free CPP selection as an sklearn transformer)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.utils.estimator_checks import (check_no_attributes_set_in_init,
                                            check_parameters_default_constructible)

import aaanalysis as aa

aa.options["verbose"] = False


@pytest.fixture(scope="module")
def data():
    df_seq = aa.load_dataset(name="DOM_GSEC", n=20)
    labels = df_seq["label"].to_list()
    return df_seq, labels


# --------------------------------------------------------------------------- sklearn contract
class TestSklearnContract:
    def test_no_attributes_set_in_init(self):
        # Constructor stores only params -> cloneable (sklearn estimator contract).
        check_no_attributes_set_in_init("SequenceFeatureTransformer", aa.SequenceFeatureTransformer())

    def test_parameters_default_constructible(self):
        check_parameters_default_constructible("SequenceFeatureTransformer",
                                               aa.SequenceFeatureTransformer())

    def test_set_output_available(self):
        # get_feature_names_out enables the inherited sklearn set_output.
        sft = aa.SequenceFeatureTransformer()
        assert sft.set_output(transform="default") is sft

    def test_clone_preserves_params(self):
        sft = aa.SequenceFeatureTransformer(n_filter=25, simplify=True, random_state=3)
        cloned = clone(sft)
        assert cloned.get_params() == sft.get_params()

    def test_get_set_params_roundtrip(self):
        sft = aa.SequenceFeatureTransformer()
        sft.set_params(n_filter=10, simplify=True)
        assert sft.get_params()["n_filter"] == 10 and sft.get_params()["simplify"] is True

    def test_sklearn_tags(self):
        tags = aa.SequenceFeatureTransformer().__sklearn_tags__()
        assert tags.target_tags.required is True
        assert tags.no_validation is True


# --------------------------------------------------------------------------- constructor params
class TestInit:
    def test_all_params_by_name(self):
        sft = aa.SequenceFeatureTransformer(
            split_kws=aa.SequenceFeature.get_split_kws(split_types=["Segment"]),
            df_scales=aa.load_scales(name="scales"),
            n_filter=25, label_test=1, label_ref=0, max_overlap=0.5, max_cor=0.5,
            simplify=False, n_jobs=1, random_state=42, verbose=False)
        assert sft.n_filter == 25 and sft.random_state == 42


# --------------------------------------------------------------------------- fit
class TestFit:
    def test_fit_returns_self(self, data):
        df_seq, labels = data
        sft = aa.SequenceFeatureTransformer(n_filter=25, random_state=0)
        assert sft.fit(X=df_seq, y=labels) is sft

    def test_fit_sets_learned_state(self, data):
        df_seq, labels = data
        sft = aa.SequenceFeatureTransformer(n_filter=25, random_state=0).fit(df_seq, labels)
        assert len(sft.features_) == 25
        assert isinstance(sft.df_feat_, pd.DataFrame)

    def test_fit_requires_y(self, data):
        df_seq, _ = data
        with pytest.raises(ValueError):
            aa.SequenceFeatureTransformer().fit(df_seq)

    def test_fit_accepts_df_parts(self, data):
        df_seq, labels = data
        df_parts = aa.SequenceFeature().get_df_parts(df_seq=df_seq)
        sft = aa.SequenceFeatureTransformer(n_filter=25, random_state=0).fit(df_parts, labels)
        assert len(sft.features_) == 25

    def test_fit_simplify(self, data):
        df_seq, labels = data
        sft = aa.SequenceFeatureTransformer(n_filter=50, simplify=True, random_state=0).fit(df_seq, labels)
        assert len(sft.features_) >= 1

    def test_invalid_n_filter(self, data):
        df_seq, labels = data
        with pytest.raises(ValueError):
            aa.SequenceFeatureTransformer(n_filter=0).fit(df_seq, labels)

    def test_invalid_max_overlap(self, data):
        df_seq, labels = data
        with pytest.raises(ValueError):
            aa.SequenceFeatureTransformer(max_overlap=2.0).fit(df_seq, labels)

    def test_invalid_max_cor(self, data):
        df_seq, labels = data
        with pytest.raises(ValueError):
            aa.SequenceFeatureTransformer(max_cor=-0.1).fit(df_seq, labels)

    def test_invalid_label_ref_not_present(self, data):
        df_seq, labels = data
        with pytest.raises(ValueError):
            aa.SequenceFeatureTransformer(label_ref=9).fit(df_seq, labels)

    def test_invalid_label_test_equals_ref(self, data):
        df_seq, labels = data
        with pytest.raises(ValueError):
            aa.SequenceFeatureTransformer(label_test=1, label_ref=1).fit(df_seq, labels)


# --------------------------------------------------------------------------- transform
class TestTransform:
    def test_transform_shape(self, data):
        df_seq, labels = data
        sft = aa.SequenceFeatureTransformer(n_filter=25, random_state=0).fit(df_seq, labels)
        X = sft.transform(X=df_seq)
        assert X.shape == (len(df_seq), 25)
        assert isinstance(X, np.ndarray)

    def test_transform_before_fit_raises(self, data):
        df_seq, _ = data
        from sklearn.exceptions import NotFittedError
        with pytest.raises(NotFittedError):
            aa.SequenceFeatureTransformer().transform(df_seq)

    def test_fit_transform_equals_fit_then_transform(self, data):
        df_seq, labels = data
        a = aa.SequenceFeatureTransformer(n_filter=25, random_state=0).fit_transform(df_seq, labels)
        sft = aa.SequenceFeatureTransformer(n_filter=25, random_state=0).fit(df_seq, labels)
        b = sft.transform(df_seq)
        assert np.allclose(a, b)

    def test_reproducible_same_seed(self, data):
        df_seq, labels = data
        a = aa.SequenceFeatureTransformer(n_filter=25, random_state=7).fit(df_seq, labels).features_
        b = aa.SequenceFeatureTransformer(n_filter=25, random_state=7).fit(df_seq, labels).features_
        assert a == b

    def test_get_feature_names_out(self, data):
        df_seq, labels = data
        sft = aa.SequenceFeatureTransformer(n_filter=25, random_state=0).fit(df_seq, labels)
        names = sft.get_feature_names_out(input_features=None)
        assert list(names) == sft.features_

    def test_pandas_output(self, data):
        df_seq, labels = data
        sft = aa.SequenceFeatureTransformer(n_filter=25, random_state=0).set_output(transform="pandas")
        out = sft.fit_transform(df_seq, labels)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == sft.features_
