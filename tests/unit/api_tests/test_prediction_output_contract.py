"""This is a script to test the output contract of the prediction tier.

``AAPred.predict`` (``df_pred``, all three levels), ``ReliabilityModel.predict``
(``df_rel``) and ``ReliabilityModel.eval`` (``df_eval_reliability``) are consumed by
downstream tools, so their columns are pinned in ``ut.DICT_DF_SCHEMAS``. These tests
build each frame on a tiny seeded fixture and fail when a column is renamed, dropped,
reordered, retyped or undocumented.

Every frame is guarded by two independent layers, because a schema-driven check on its
own would pass a *coordinated* code + schema edit:

- literal expectations hard-coded in this file -- the column order, the dtype kind per
  column, the required ``df_feat`` columns with their dtypes, and the raw
  ``PART-SPLIT-SCALE`` feature-id grammar as a regex (not the production parser);
- the live ``ut.DICT_DF_SCHEMAS`` record, asserted to agree with both the literal map
  and the produced frame.

This addresses the per-sample and per-residue half of the documented boundary contract;
the ``df_feat`` half is guarded by ``test_df_feat_contract.py`` (the committed
``load_features`` frame vs ``DICT_DF_FEAT``) and ``test_cpp_schema.py`` (canonical column
order). What is added here is the literal grammar and the literal dtypes of a freshly
computed ``CPP.run``, exercised once per supported split type.

``df_pred.score`` has no single numeric range: ``AAPred.predict(score_range=...)`` emits
``[0, 1]`` on the default ``'proba'`` scale and ``[0, 100]`` on ``'percent'``. The schema
therefore contracts one range per scale (``scale_ranges``), and both scales are checked
by the same conformance helper -- neither is exempted from it.
"""
import re
import warnings

import numpy as np
import pandas.api.types as pdt
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

import aaanalysis as aa
import aaanalysis.utils as ut

# Frozen literal column orders: a rename in the constants layer breaks these on purpose.
COLS_PRED_SEQUENCE = ["entry", "score", "score_std"]
COLS_PRED_DOMAIN = ["entry", "offset", "score", "is_best"]
COLS_PRED_WINDOW = ["entry", "position", "score", "score_std"]
COLS_REL = ["score", "score_std", "ci_low", "ci_high", "ood_score", "in_domain", "ad_knn",
            "ad_mahalanobis", "ad_leverage", "score_calibrated", "margin", "entropy",
            "conformal_set", "reliable", "ad_status", "ad_nearest_train"]
COLS_EVAL_REL = ["bin", "mean_score", "empirical_pos", "n_samples"]

# Frozen literal dtype maps -- the half that does NOT read the schema, so a coordinated
# code + schema retype still fails here. The coarse kind ('int' / 'float') is pinned on
# purpose: the exact numpy width (int64 vs int32) legitimately differs across platforms.
DTYPES_PRED_SEQUENCE = {"entry": "str", "score": "float", "score_std": "float"}
DTYPES_PRED_DOMAIN = {"entry": "str", "offset": "int", "score": "float",
                      "is_best": "bool"}
DTYPES_PRED_WINDOW = {"entry": "str", "position": "int", "score": "float",
                      "score_std": "float"}
DTYPES_PRED_LABEL = {"predicted_label": "int"}
DTYPES_REL = {"score": "float", "score_std": "float", "ci_low": "float",
              "ci_high": "float", "ood_score": "float", "in_domain": "bool",
              "ad_knn": "float", "ad_mahalanobis": "float", "ad_leverage": "float",
              "score_calibrated": "float", "margin": "float", "entropy": "float",
              "conformal_set": "str", "reliable": "bool", "ad_status": "str",
              "ad_nearest_train": "int"}
DTYPES_EVAL_REL = {"bin": "str", "mean_score": "float", "empirical_pos": "float",
                   "n_samples": "int"}

# The df_feat columns downstream tools consume, written out rather than imported from
# ut.LIST_COLS_FEAT, so editing that constant cannot silently move the contract.
COLS_FEAT_REQUIRED = ["feature", "category", "subcategory", "scale_name",
                      "scale_description", "abs_auc", "abs_mean_dif", "mean_dif",
                      "std_test", "std_ref", "p_val_mann_whitney", "p_val_fdr_bh",
                      "positions"]
# ... and their dtypes, likewise written out: a retype of any contracted df_feat field
# fails here even when DICT_DF_SCHEMAS is edited to agree with the new dtype.
DTYPES_FEAT_REQUIRED = {"feature": "str", "category": "str", "subcategory": "str",
                        "scale_name": "str", "scale_description": "str",
                        "abs_auc": "float", "abs_mean_dif": "float",
                        "mean_dif": "float", "std_test": "float", "std_ref": "float",
                        "p_val_mann_whitney": "float", "p_val_fdr_bh": "float",
                        "positions": "str"}
# Raw PART-SPLIT-SCALE grammar as a literal regex (NOT ut.split_feat_id): PART is upper
# case, SPLIT is one of the three split types with its parenthesised arguments, and the
# scale id carries no '-'.
RE_FEAT_ID = re.compile(r"^(?P<part>[A-Z][A-Z0-9_]*)"
                        r"-(?P<split>(?:Segment|PeriodicPattern|Pattern)\([^()]*\))"
                        r"-(?P<scale>[A-Za-z0-9_]+)$")
SPLIT_TYPES = ("Segment", "Pattern", "PeriodicPattern")
# Parts reachable from the default df_parts of the fixture below.
PARTS_FIXTURE = {"TMD", "JMD_N_TMD_N", "TMD_C_JMD_C"}


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


def _range_for(rec, scale):
    """The numeric range a record contracts, on ``scale`` when it is scale-dependent."""
    if "scale_ranges" in rec:
        assert scale is not None, "a scale-dependent range needs an explicit scale"
        assert scale in rec["scale_ranges"], f"scale '{scale}' is not documented"
        return rec["scale_ranges"][scale]
    return rec.get("range")


def _assert_conforms(df, frame, scale=None):
    """Every column documented, with the documented dtype, nullability, range, values.

    ``scale`` picks the branch of a per-scale range record, so an alternative output
    scale is checked by this same machinery instead of being exempted from it.
    """
    schema = ut.DICT_DF_SCHEMAS[frame]["columns"]
    undocumented = [c for c in df.columns if c not in schema]
    assert not undocumented, f"{frame}: undocumented columns {undocumented}"
    for col in df.columns:
        rec = schema[col]
        assert _kind(df[col]) == rec["dtype"], (
            f"{frame}.{col}: dtype {_kind(df[col])} != {rec['dtype']}")
        if not rec["nullable"]:
            assert df[col].notna().all(), f"{frame}.{col} has missing values"
        if rec["unique"]:
            assert df[col].is_unique, f"{frame}.{col} not unique"
        rng = _range_for(rec, scale)
        if rng is not None and rec["dtype"] in {"int", "float"}:
            lo, hi = rng
            vals = df[col].dropna().to_numpy(dtype=float)
            if lo is not None:
                assert (vals >= lo - 1e-9).all(), f"{frame}.{col} below {lo}"
            if hi is not None:
                assert (vals <= hi + 1e-9).all(), f"{frame}.{col} above {hi}"
        if "allowed_values" in rec:
            assert set(df[col]).issubset(rec["allowed_values"]), (
                f"{frame}.{col} unexpected values")
    for col, rec in schema.items():
        if rec["required"]:
            assert col in df.columns, f"{frame}: required column '{col}' missing"


def _assert_literal_dtypes(df, expected, frame):
    """The live frame AND the schema must both match the hard-coded literal dtype map."""
    schema = ut.DICT_DF_SCHEMAS[frame]["columns"]
    for col, kind in expected.items():
        assert col in df.columns, f"{frame}: column '{col}' missing"
        assert _kind(df[col]) == kind, (
            f"{frame}.{col}: live dtype {_kind(df[col])} != literal {kind}")
        assert schema[col]["dtype"] == kind, (
            f"{frame}.{col}: schema dtype {schema[col]['dtype']} != literal {kind}")


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
    X, y = make_classification(n_samples=120, n_features=8, n_informative=5,
                               n_redundant=1, random_state=0)
    rm = aa.ReliabilityModel(random_state=0).fit(X[:90], y[:90])
    return rm, X[90:], y[90:]


@pytest.fixture(scope="module")
def rm_uncalibrated():
    X, y = make_classification(n_samples=120, n_features=8, n_informative=5,
                               n_redundant=1, random_state=0)
    rm = aa.ReliabilityModel(random_state=0).fit(X[:90], y[:90], calibrate=False)
    return rm, X[90:]


@pytest.fixture(scope="module")
def rm_degenerate():
    """A degenerate training reference: with n_features >= n_samples the covariance is
    rank-deficient, so ad_mahalanobis / ad_leverage are not identifiable and come back
    NaN. The frame must still satisfy the documented schema."""
    X, y = make_classification(n_samples=8, n_features=10, n_informative=4,
                               n_redundant=0, n_clusters_per_class=1, random_state=0)
    rm = aa.ReliabilityModel(random_state=0).fit(X, y)
    return rm, X[:4]


@pytest.fixture(scope="module")
def cpp_feat():
    """Freshly computed df_feat from CPP.run, one frame per supported split type.

    One run per split type rather than a single mixed run: CPP.run returns only the
    n_filter best features, so a mixed run could legitimately rank one split type out of
    the output and silently stop exercising its id format.
    """
    df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
    labels = df_seq["label"].to_list()
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq)
    df_scales = aa.load_scales().iloc[:, :15]
    dict_df_feat = {}
    for split_type in SPLIT_TYPES:
        split_kws = sf.get_split_kws(split_types=[split_type], n_split_min=1,
                                     n_split_max=2, steps_pattern=[3, 4], n_min=2,
                                     n_max=3, len_max=10, steps_periodicpattern=[3, 4])
        cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, split_kws=split_kws,
                     verbose=False, random_state=0)
        with warnings.catch_warnings():
            # Narrow on purpose: only the two advisories a tiny fixture raises by
            # construction (sparse candidate pool, empty Pattern bucket) are silenced,
            # so an unexpected warning still surfaces.
            warnings.filterwarnings("ignore", message="'n_filter'.*candidate features")
            warnings.filterwarnings("ignore", message="'Pattern' split config")
            dict_df_feat[split_type] = cpp.run(labels=labels, n_filter=10, n_jobs=1)
    return dict_df_feat, df_scales


# ------------------------------------------------------------------------ structure
class TestPredictionSchemaRegistered:
    @pytest.mark.parametrize("frame", ["df_pred", "df_rel", "df_eval_reliability"])
    def test_frame_registered(self, frame):
        assert frame in ut.DICT_DF_SCHEMAS
        assert ut.DICT_DF_SCHEMAS[frame]["columns"]

    def test_df_rel_schema_matches_constant_bundle(self):
        assert list(ut.DICT_DF_SCHEMAS["df_rel"]["columns"]) == list(ut.COLS_RELIABILITY)

    def test_df_eval_reliability_schema_matches_constant_bundle(self):
        assert (list(ut.DICT_DF_SCHEMAS["df_eval_reliability"]["columns"])
                == list(ut.COLS_EVAL_RELIABILITY))

    def test_frozen_names_match_constants(self):
        assert COLS_REL == list(ut.COLS_RELIABILITY)
        assert COLS_EVAL_REL == list(ut.COLS_EVAL_RELIABILITY)
        pred_cols = set(ut.DICT_DF_SCHEMAS["df_pred"]["columns"])
        frozen = COLS_PRED_SEQUENCE + COLS_PRED_DOMAIN + COLS_PRED_WINDOW
        assert set(frozen + ["predicted_label"]) == pred_cols

    def test_schema_dtypes_match_literal_maps(self):
        """The schema alone cannot drift: it must equal the literal maps in this file."""
        for frame, literal in [("df_rel", DTYPES_REL),
                               ("df_eval_reliability", DTYPES_EVAL_REL)]:
            schema = ut.DICT_DF_SCHEMAS[frame]["columns"]
            assert {c: r["dtype"] for c, r in schema.items()} == literal, frame
        pred = ut.DICT_DF_SCHEMAS["df_pred"]["columns"]
        merged = {**DTYPES_PRED_SEQUENCE, **DTYPES_PRED_DOMAIN, **DTYPES_PRED_WINDOW,
                  **DTYPES_PRED_LABEL}
        assert {c: r["dtype"] for c, r in pred.items()} == merged


# -------------------------------------------------------------------------- df_pred
class TestDfPredContract:
    def test_sequence_level(self, aap_fitted):
        aap, df_seq = aap_fitted
        df = aap.predict(df_seq, level="sequence")
        assert list(df.columns) == COLS_PRED_SEQUENCE
        assert len(df) == len(df_seq)
        _assert_literal_dtypes(df, DTYPES_PRED_SEQUENCE, "df_pred")
        _assert_conforms(df, "df_pred", scale=ut.STR_SCORE_RANGE_PROBA)
        assert df["entry"].is_unique

    def test_domain_level(self, aap_fitted):
        aap, df_seq = aap_fitted
        df = aap.predict(df_seq.head(3), level="domain", window=1)
        assert list(df.columns) == COLS_PRED_DOMAIN
        _assert_literal_dtypes(df, DTYPES_PRED_DOMAIN, "df_pred")
        _assert_conforms(df, "df_pred", scale=ut.STR_SCORE_RANGE_PROBA)
        assert (df.groupby("entry")["is_best"].sum() == 1).all()

    def test_window_level(self, aap_fitted):
        aap, df_seq = aap_fitted
        df = aap.predict(df_seq.head(2), level="window", tmd_len=20, step=10)
        assert list(df.columns) == COLS_PRED_WINDOW
        assert len(df) > 0
        _assert_literal_dtypes(df, DTYPES_PRED_WINDOW, "df_pred")
        _assert_conforms(df, "df_pred", scale=ut.STR_SCORE_RANGE_PROBA)

    def test_threshold_appends_predicted_label(self, aap_fitted):
        aap, df_seq = aap_fitted
        df = aap.predict(df_seq, level="sequence", threshold=0.5)
        assert list(df.columns) == COLS_PRED_SEQUENCE + ["predicted_label"]
        _assert_literal_dtypes(df, {**DTYPES_PRED_SEQUENCE, **DTYPES_PRED_LABEL},
                               "df_pred")
        _assert_conforms(df, "df_pred", scale=ut.STR_SCORE_RANGE_PROBA)

    def test_score_range_is_documented_per_scale(self):
        """Both score_range scales are contracted; neither is left undocumented."""
        rec = ut.DICT_DF_SCHEMAS["df_pred"]["columns"]["score"]
        assert "range" not in rec, "a single range would be false on one scale"
        assert set(rec["scale_ranges"]) == set(ut.LIST_SCORE_RANGES)
        assert rec["scale_ranges"][ut.STR_SCORE_RANGE_PROBA] == [0, 1]
        assert rec["scale_ranges"][ut.STR_SCORE_RANGE_PERCENT] == [0, 100]

    def test_percent_scale_conforms_to_its_documented_range(self, aap_fitted):
        """The percent output is checked by the generic helper, not exempted from it."""
        aap, df_seq = aap_fitted
        df_proba = aap.predict(df_seq, level="sequence")
        df_pct = aap.predict(df_seq, level="sequence", score_range="percent")
        assert list(df_pct.columns) == COLS_PRED_SEQUENCE
        _assert_literal_dtypes(df_pct, DTYPES_PRED_SEQUENCE, "df_pred")
        _assert_conforms(df_pct, "df_pred", scale=ut.STR_SCORE_RANGE_PERCENT)
        np.testing.assert_allclose(df_pct["score"].to_numpy(),
                                   df_proba["score"].to_numpy() * 100)
        np.testing.assert_allclose(df_pct["score_std"].to_numpy(),
                                   df_proba["score_std"].to_numpy() * 100)

    def test_percent_frame_fails_the_proba_contract(self, aap_fitted):
        """The per-scale range is a real gate: the scales are not interchangeable."""
        aap, df_seq = aap_fitted
        df_pct = aap.predict(df_seq, level="sequence", score_range="percent")
        assert df_pct["score"].max() > 1, "fixture too weak to tell the scales apart"
        with pytest.raises(AssertionError, match="above 1"):
            _assert_conforms(df_pct, "df_pred", scale=ut.STR_SCORE_RANGE_PROBA)

    def test_scale_dependent_range_needs_a_scale(self, aap_fitted):
        """Checking a scale-dependent frame without naming the scale is an error."""
        aap, df_seq = aap_fitted
        df = aap.predict(df_seq, level="sequence")
        with pytest.raises(AssertionError, match="needs an explicit scale"):
            _assert_conforms(df, "df_pred")


# ------------------------------------------------- df_rel / df_eval_reliability
class TestReliabilityOutputContract:
    def test_df_rel(self, rm_fitted):
        rm, X_test, _ = rm_fitted
        df = rm.predict(X_test)
        assert list(df.columns) == COLS_REL
        assert len(df) == len(X_test)
        _assert_literal_dtypes(df, DTYPES_REL, "df_rel")
        _assert_conforms(df, "df_rel")

    def test_df_rel_ood_sample_conforms(self, rm_fitted):
        rm, X_test, _ = rm_fitted
        X_ood = X_test[:1] + 50.0
        df = rm.predict(X_ood)
        _assert_conforms(df, "df_rel")
        assert not bool(df["in_domain"].iloc[0])

    def test_uncalibrated_ambiguity_uses_ensemble_score(self, rm_uncalibrated):
        rm, X_test = rm_uncalibrated
        df = rm.predict(X_test)
        assert df["score_calibrated"].isna().all()
        np.testing.assert_allclose(df["margin"].to_numpy(),
                                   np.abs(df["score"].to_numpy() - 0.5) * 2)

    def test_df_eval_reliability(self, rm_fitted):
        rm, X_test, y_test = rm_fitted
        df = rm.eval(X=X_test, labels=y_test, n_bins=4)
        assert list(df.columns) == COLS_EVAL_REL
        assert len(df) == 5 and df["bin"].iloc[-1] == ut.STR_BIN_SUMMARY
        _assert_literal_dtypes(df, DTYPES_EVAL_REL, "df_eval_reliability")
        _assert_conforms(df, "df_eval_reliability")
        assert int(df["n_samples"].iloc[-1]) == len(X_test)
        assert int(df["n_samples"].iloc[:-1].sum()) == len(X_test)

    def test_df_eval_reliability_with_calibration_metrics(self, rm_fitted):
        rm, X_test, y_test = rm_fitted
        df = rm.eval(X=X_test, labels=y_test, n_bins=4, use_calibrated=True,
                     add_metrics=True)
        assert list(df.columns) == COLS_EVAL_REL
        assert df["bin"].iloc[-3:].to_list() == [ut.STR_BIN_SUMMARY, ut.STR_BIN_BRIER,
                                                 ut.STR_BIN_ECE]
        _assert_literal_dtypes(df, DTYPES_EVAL_REL, "df_eval_reliability")
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

    def test_retyped_column_is_detected(self, rm_fitted):
        """A coordinated code+schema retype is caught by the literal map."""
        rm, X_test, _ = rm_fitted
        df = rm.predict(X_test)
        df["ad_nearest_train"] = df["ad_nearest_train"].astype(float)
        with pytest.raises(AssertionError, match="literal"):
            _assert_literal_dtypes(df, DTYPES_REL, "df_rel")


# ------------------------------------------------------- degenerate applicability domain
class TestReliabilityDegenerateDomain:
    def test_ad_distances_are_nan(self, rm_degenerate):
        rm, X_new = rm_degenerate
        df = rm.predict(X_new)
        assert df["ad_mahalanobis"].isna().all()
        assert df["ad_leverage"].isna().all()

    def test_schema_documents_them_as_nullable(self):
        cols = ut.DICT_DF_SCHEMAS["df_rel"]["columns"]
        for col in ["ad_mahalanobis", "ad_leverage"]:
            assert cols[col]["nullable"], f"{col} must be documented nullable"
            assert "degenerate" in cols[col]["description"]

    def test_degenerate_frame_still_conforms(self, rm_degenerate):
        """The documented contract must hold on the degenerate reference too."""
        rm, X_new = rm_degenerate
        df = rm.predict(X_new)
        assert list(df.columns) == COLS_REL
        _assert_literal_dtypes(df, DTYPES_REL, "df_rel")
        _assert_conforms(df, "df_rel")

    def test_other_columns_stay_populated(self, rm_degenerate):
        rm, X_new = rm_degenerate
        df = rm.predict(X_new)
        populated = [c for c in COLS_REL if c not in ("ad_mahalanobis", "ad_leverage")]
        assert df[populated].notna().all().all()


# ------------------------------------------------------------- CPP df_feat grammar
class TestCppFeatureIdContract:
    """The CPP output downstream tools consume, pinned by literals.

    ``test_df_feat_contract.py`` already guards the committed ``load_features`` frame
    against ``ut.DICT_DF_FEAT`` and parses ids with ``ut.split_feat_id``; what is added
    here is a *freshly computed* ``CPP.run`` -- for every supported split type -- checked
    against a hard-coded column list, a hard-coded dtype map and a raw regex, so a change
    to the production constants or parser cannot move the contract.
    """

    def test_every_split_type_is_produced(self, cpp_feat):
        """The fixture really exercises all three split types, not just Segment."""
        dict_df_feat, _ = cpp_feat
        assert set(dict_df_feat) == set(SPLIT_TYPES)
        for split_type, df_feat in dict_df_feat.items():
            assert len(df_feat) > 0, f"no features produced for {split_type}"

    @pytest.mark.parametrize("split_type", SPLIT_TYPES)
    def test_required_columns_present_in_literal_order(self, cpp_feat, split_type):
        df_feat = cpp_feat[0][split_type]
        assert list(df_feat.columns)[:len(COLS_FEAT_REQUIRED)] == COLS_FEAT_REQUIRED

    @pytest.mark.parametrize("split_type", SPLIT_TYPES)
    def test_contracted_dtypes_are_literal(self, cpp_feat, split_type):
        """Every contracted df_feat field has its dtype pinned, live and in the schema."""
        df_feat = cpp_feat[0][split_type]
        _assert_literal_dtypes(df_feat, DTYPES_FEAT_REQUIRED, "df_feat")

    @pytest.mark.parametrize("split_type", SPLIT_TYPES)
    def test_frame_conforms_to_the_schema(self, cpp_feat, split_type):
        df_feat = cpp_feat[0][split_type]
        _assert_conforms(df_feat, "df_feat")

    def test_production_constant_matches_literal(self, cpp_feat):
        """If LIST_COLS_FEAT is edited, this literal is the contract that must win."""
        assert list(ut.LIST_COLS_FEAT) == COLS_FEAT_REQUIRED

    def test_schema_dtypes_match_literal_map(self):
        """The df_feat schema alone cannot drift away from the literal dtype map."""
        schema = ut.DICT_DF_SCHEMAS["df_feat"]["columns"]
        assert {c: schema[c]["dtype"] for c in COLS_FEAT_REQUIRED} == DTYPES_FEAT_REQUIRED

    @pytest.mark.parametrize("split_type", SPLIT_TYPES)
    def test_feature_id_matches_raw_grammar(self, cpp_feat, split_type):
        df_feat = cpp_feat[0][split_type]
        for feat_id in df_feat["feature"]:
            assert RE_FEAT_ID.match(feat_id), f"id breaks PART-SPLIT-SCALE: {feat_id}"

    @pytest.mark.parametrize("split_type", SPLIT_TYPES)
    def test_feature_id_split_type_is_the_requested_one(self, cpp_feat, split_type):
        """The grammar resolves each split type exactly, PeriodicPattern included."""
        df_feat = cpp_feat[0][split_type]
        observed = {RE_FEAT_ID.match(f).group("split").split("(")[0]
                    for f in df_feat["feature"]}
        assert observed == {split_type}

    @pytest.mark.parametrize("split_type", SPLIT_TYPES)
    def test_feature_id_part_in_vocabulary(self, cpp_feat, split_type):
        df_feat = cpp_feat[0][split_type]
        for feat_id in df_feat["feature"]:
            part = RE_FEAT_ID.match(feat_id).group("part")
            assert part in PARTS_FIXTURE, f"unexpected part {part} in {feat_id}"
            assert part.lower() in ut.LIST_ALL_PARTS, f"{part} not a known part"

    @pytest.mark.parametrize("split_type", SPLIT_TYPES)
    def test_feature_id_scale_is_in_df_scales(self, cpp_feat, split_type):
        dict_df_feat, df_scales = cpp_feat
        valid = set(df_scales.columns)
        for feat_id in dict_df_feat[split_type]["feature"]:
            scale_id = RE_FEAT_ID.match(feat_id).group("scale")
            assert scale_id in valid, f"{scale_id} not a column of df_scales"

    @pytest.mark.parametrize("split_type", SPLIT_TYPES)
    def test_feature_ids_unique(self, cpp_feat, split_type):
        assert cpp_feat[0][split_type]["feature"].is_unique

    @pytest.mark.parametrize("split_type", SPLIT_TYPES)
    def test_positions_are_increasing_and_1_based(self, cpp_feat, split_type):
        """The per-residue half of the contract: 'positions' is a 1-based ascending list."""
        for positions in cpp_feat[0][split_type]["positions"]:
            pos = [int(p) for p in positions.split(",")]
            assert pos[0] >= 1, f"positions are 1-based: {positions}"
            assert all(b > a for a, b in zip(pos, pos[1:])), positions

    def test_segment_positions_are_contiguous(self, cpp_feat):
        """A Segment is a continuous sub-sequence, so its positions have no gap."""
        for positions in cpp_feat[0]["Segment"]["positions"]:
            pos = [int(p) for p in positions.split(",")]
            assert pos == list(range(pos[0], pos[-1] + 1)), positions

    @pytest.mark.parametrize("split_type", ["Pattern", "PeriodicPattern"])
    def test_pattern_positions_are_discontinuous(self, cpp_feat, split_type):
        """Pattern / PeriodicPattern are discontinuous, so some step exceeds 1."""
        for positions in cpp_feat[0][split_type]["positions"]:
            pos = [int(p) for p in positions.split(",")]
            assert max(b - a for a, b in zip(pos, pos[1:])) > 1, positions

    def test_broken_id_is_rejected_by_the_grammar(self):
        """The regex is a real gate: these malformed ids must not match."""
        for bad in ["TMD-Segment(1,2)", "tmd-Segment(1,2)-ARGP820103",
                    "TMD-Chunk(1,2)-ARGP820103", "TMD-Segment(1,2)-ARG-P820103",
                    "TMD-Pattern-ARGP820103", "TMD-PeriodicPattern(C,i+3/4,1)"]:
            assert RE_FEAT_ID.match(bad) is None, bad
