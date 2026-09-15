"""This is a script to test the output contract of the prediction tier.

``AAPred.predict`` (``df_pred``, all three levels), ``ReliabilityModel.predict`` (``df_rel``)
and ``ReliabilityModel.eval`` (``df_eval_reliability``) are consumed by downstream tools, so
their columns are pinned in ``ut.DICT_DF_SCHEMAS``. These tests generate each frame on a tiny
seeded fixture and fail when a column is renamed, dropped, reordered, retyped or undocumented.
Addresses #26.
"""
import numpy as np
import pandas.api.types as pdt
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

import aaanalysis as aa
import aaanalysis.utils as ut

aa.options["verbose"] = False

# Frozen literal column orders: a rename in the constants layer breaks these on purpose.
COLS_PRED_SEQUENCE = ["entry", "score", "score_std"]
COLS_PRED_DOMAIN = ["entry", "offset", "score", "is_best"]
COLS_PRED_WINDOW = ["entry", "position", "score", "score_std"]
COLS_REL = ["score", "score_std", "ci_low", "ci_high", "ood_score", "in_domain", "ad_knn",
            "ad_mahalanobis", "ad_leverage", "score_calibrated", "margin", "entropy",
            "conformal_set", "reliable", "ad_status", "ad_nearest_train"]
COLS_EVAL_REL = ["bin", "mean_score", "empirical_pos", "n_samples"]


def _kind(series):
    if pdt.is_bool_dtype(series):
        return "bool"
    if pdt.is_float_dtype(series):
        return "float"
    if pdt.is_integer_dtype(series):
        return "int"
    if pdt.is_string_dtype(series) or series.dtype == object:
        return "str"
    return str(series.dtype)


def _assert_conforms(df, frame):
    """Every column documented, with the documented dtype, nullability, range and values."""
    schema = ut.DICT_DF_SCHEMAS[frame]["columns"]
    undocumented = [c for c in df.columns if c not in schema]
    assert not undocumented, f"{frame}: undocumented columns {undocumented}"
    for col in df.columns:
        rec = schema[col]
        assert _kind(df[col]) == rec["dtype"], f"{frame}.{col}: dtype {_kind(df[col])} != {rec['dtype']}"
        if not rec["nullable"]:
            assert df[col].notna().all(), f"{frame}.{col} has missing values"
        if rec["unique"]:
            assert df[col].is_unique, f"{frame}.{col} not unique"
        if "range" in rec and rec["dtype"] in {"int", "float"}:
            lo, hi = rec["range"]
            vals = df[col].dropna().to_numpy(dtype=float)
            if lo is not None:
                assert (vals >= lo - 1e-9).all(), f"{frame}.{col} below {lo}"
            if hi is not None:
                assert (vals <= hi + 1e-9).all(), f"{frame}.{col} above {hi}"
        if "allowed_values" in rec:
            assert set(df[col]).issubset(rec["allowed_values"]), f"{frame}.{col} unexpected values"
    for col, rec in schema.items():
        if rec["required"]:
            assert col in df.columns, f"{frame}: required column '{col}' missing"


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def aap_fitted():
    df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
    labels = df_seq["label"].to_list()
    df_feat = aa.load_features(name="DOM_GSEC").head(5)
    sf = aa.SequenceFeature()
    X = sf.feature_matrix(features=df_feat, df_parts=sf.get_df_parts(df_seq=df_seq))
    aap = aa.AAPred(df_feat=df_feat, list_model_classes=[RandomForestClassifier],
                    random_state=42).fit(X, labels)
    return aap, df_seq


@pytest.fixture(scope="module")
def rm_fitted():
    X, y = make_classification(n_samples=120, n_features=8, n_informative=5, n_redundant=1,
                               random_state=0)
    rm = aa.ReliabilityModel(random_state=0).fit(X[:90], y[:90])
    return rm, X[90:], y[90:]


# ------------------------------------------------------------------------ structure
class TestPredictionSchemaRegistered:
    @pytest.mark.parametrize("frame", ["df_pred", "df_rel", "df_eval_reliability"])
    def test_frame_registered(self, frame):
        assert frame in ut.DICT_DF_SCHEMAS
        assert ut.DICT_DF_SCHEMAS[frame]["columns"]

    def test_df_rel_schema_matches_constant_bundle(self):
        assert list(ut.DICT_DF_SCHEMAS["df_rel"]["columns"]) == list(ut.COLS_RELIABILITY)

    def test_df_eval_reliability_schema_matches_constant_bundle(self):
        assert list(ut.DICT_DF_SCHEMAS["df_eval_reliability"]["columns"]) == list(ut.COLS_EVAL_RELIABILITY)

    def test_frozen_names_match_constants(self):
        assert COLS_REL == list(ut.COLS_RELIABILITY)
        assert COLS_EVAL_REL == list(ut.COLS_EVAL_RELIABILITY)
        pred_cols = set(ut.DICT_DF_SCHEMAS["df_pred"]["columns"])
        assert set(COLS_PRED_SEQUENCE + COLS_PRED_DOMAIN + COLS_PRED_WINDOW + ["predicted_label"]) == pred_cols


# -------------------------------------------------------------------------- df_pred
class TestDfPredContract:
    def test_sequence_level(self, aap_fitted):
        aap, df_seq = aap_fitted
        df = aap.predict(df_seq, level="sequence")
        assert list(df.columns) == COLS_PRED_SEQUENCE
        assert len(df) == len(df_seq)
        _assert_conforms(df, "df_pred")
        assert df["entry"].is_unique

    def test_domain_level(self, aap_fitted):
        aap, df_seq = aap_fitted
        df = aap.predict(df_seq.head(3), level="domain", window=1)
        assert list(df.columns) == COLS_PRED_DOMAIN
        _assert_conforms(df, "df_pred")
        assert (df.groupby("entry")["is_best"].sum() == 1).all()

    def test_window_level(self, aap_fitted):
        aap, df_seq = aap_fitted
        df = aap.predict(df_seq.head(2), level="window", tmd_len=20, step=10)
        assert list(df.columns) == COLS_PRED_WINDOW
        assert len(df) > 0
        _assert_conforms(df, "df_pred")

    def test_threshold_appends_predicted_label(self, aap_fitted):
        aap, df_seq = aap_fitted
        df = aap.predict(df_seq, level="sequence", threshold=0.5)
        assert list(df.columns) == COLS_PRED_SEQUENCE + ["predicted_label"]
        _assert_conforms(df, "df_pred")

    def test_percent_range_within_schema(self, aap_fitted):
        aap, df_seq = aap_fitted
        df = aap.predict(df_seq, level="sequence", score_range="percent")
        _assert_conforms(df, "df_pred")


# --------------------------------------------------------------- df_rel / df_eval_reliability
class TestReliabilityOutputContract:
    def test_df_rel(self, rm_fitted):
        rm, X_test, _ = rm_fitted
        df = rm.predict(X_test)
        assert list(df.columns) == COLS_REL
        assert len(df) == len(X_test)
        _assert_conforms(df, "df_rel")

    def test_df_rel_ood_sample_conforms(self, rm_fitted):
        rm, X_test, _ = rm_fitted
        X_ood = X_test[:1] + 50.0
        df = rm.predict(X_ood)
        _assert_conforms(df, "df_rel")
        assert not bool(df["in_domain"].iloc[0])

    def test_df_eval_reliability(self, rm_fitted):
        rm, X_test, y_test = rm_fitted
        df = rm.eval(X=X_test, labels=y_test, n_bins=4)
        assert list(df.columns) == COLS_EVAL_REL
        assert len(df) == 5 and df["bin"].iloc[-1] == ut.STR_BIN_SUMMARY
        _assert_conforms(df, "df_eval_reliability")
        assert int(df["n_samples"].iloc[-1]) == len(X_test)
        assert int(df["n_samples"].iloc[:-1].sum()) == len(X_test)

    def test_df_eval_reliability_with_calibration_metrics(self, rm_fitted):
        rm, X_test, y_test = rm_fitted
        df = rm.eval(X=X_test, labels=y_test, n_bins=4, use_calibrated=True, add_metrics=True)
        assert list(df.columns) == COLS_EVAL_REL
        assert df["bin"].iloc[-3:].to_list() == [ut.STR_BIN_SUMMARY, ut.STR_BIN_BRIER, ut.STR_BIN_ECE]
        _assert_conforms(df, "df_eval_reliability")

    def test_df_rel_ad_status_matches_in_domain(self, rm_fitted):
        rm, X_test, _ = rm_fitted
        df = rm.predict(X_test)
        assert (df["in_domain"] == (df["ad_status"] == "inside")).all()

    def test_renamed_column_is_detected(self, rm_fitted):
        rm, X_test, _ = rm_fitted
        df = rm.predict(X_test).rename(columns={"ad_knn": "ad_knn_dist"})
        with pytest.raises(AssertionError, match="undocumented"):
            _assert_conforms(df, "df_rel")

    def test_dropped_required_column_is_detected(self, rm_fitted):
        rm, X_test, _ = rm_fitted
        df = rm.predict(X_test).drop(columns=["reliable"])
        with pytest.raises(AssertionError, match="required column"):
            _assert_conforms(df, "df_rel")
