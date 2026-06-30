"""This is a script to test the built-in per-site predictor behind CPPStructurePlot.explore()."""
import numpy as np
import pandas as pd
import pytest

# Pro-gated: the per-site impact needs SHAP (ShapModel).
pytest.importorskip("shap")

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.feature_engineering_pro._backend.cpp_struct.predict import (
    build_builtin_predictor, resolve_estimators)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC


# --- fixtures / helpers ------------------------------------------------------
_AAS = "ACDEFGHIKLMNPQRSTVWY"


def _randseq(rng, n):
    return "".join(rng.choice(list(_AAS), size=n))


@pytest.fixture(scope="module")
def df_scales():
    return aa.load_scales(name="scales")


@pytest.fixture(scope="module")
def df_feat(df_scales):
    """A complete CPP-style df_feat over real scales/categories."""
    df_cat = aa.load_scales(name="scales_cat")
    df_cat = df_cat[df_cat[ut.COL_SCALE_ID].isin(df_scales.columns)].head(4).reset_index(drop=True)
    scale_ids = df_cat[ut.COL_SCALE_ID].tolist()
    splits = ["Segment(1,2)", "Segment(2,2)", "Segment(1,1)", "Pattern(C,1)"]
    parts = ["TMD", "TMD", "JMD_N", "TMD"]
    df = pd.DataFrame({
        ut.COL_FEATURE: [f"{parts[i]}-{splits[i]}-{scale_ids[i]}" for i in range(4)],
        ut.COL_CAT: df_cat[ut.COL_CAT], ut.COL_SUBCAT: df_cat[ut.COL_SUBCAT],
        ut.COL_SCALE_NAME: df_cat[ut.COL_SCALE_NAME],
        "abs_auc": [0.2, 0.15, 0.3, 0.1], "abs_mean_dif": [0.3, 0.2, 0.5, 0.4],
        "mean_dif": [0.3, -0.2, 0.5, -0.4], "std_test": 0.1, "std_ref": 0.1})
    return df


@pytest.fixture(scope="module")
def df_seq():
    rng = np.random.default_rng(0)
    rows = [{ut.COL_ENTRY: f"P{i:03d}", ut.COL_SEQ: _randseq(rng, 60),
             ut.COL_TMD_START: 21, ut.COL_TMD_STOP: 40} for i in range(16)]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def labels():
    return [1, 0] * 8


@pytest.fixture(scope="module")
def query_seq():
    return _randseq(np.random.default_rng(1), 120)


def _predictor(df_feat, df_seq, labels, df_scales, **kw):
    kw.setdefault("model", ut.MODEL_RF)
    kw.setdefault("random_state", 42)
    return build_builtin_predictor(df_feat=df_feat, df_seq=df_seq, labels=labels, tmd_len=20,
                                   jmd_n_len=10, jmd_c_len=10, df_scales=df_scales, **kw)


# --- normal cases ------------------------------------------------------------
class TestBuildBuiltinPredictor:
    """Normal behaviour of the per-site predictor."""

    def test_returns_callable(self, df_feat, df_seq, labels, df_scales):
        assert callable(_predictor(df_feat, df_seq, labels, df_scales))

    def test_col_imp_added(self, df_feat, df_seq, labels, df_scales, query_seq):
        out = _predictor(df_feat, df_seq, labels, df_scales)(query_seq, 50)
        assert ut.COL_FEAT_IMPACT in out.columns

    def test_one_row_per_feature(self, df_feat, df_seq, labels, df_scales, query_seq):
        out = _predictor(df_feat, df_seq, labels, df_scales)(query_seq, 50)
        assert len(out) == len(df_feat)

    def test_impact_normalized_to_100(self, df_feat, df_seq, labels, df_scales, query_seq):
        out = _predictor(df_feat, df_seq, labels, df_scales)(query_seq, 50)
        assert out[ut.COL_FEAT_IMPACT].abs().sum() == pytest.approx(100.0, abs=1e-6)

    def test_proba_in_attrs(self, df_feat, df_seq, labels, df_scales, query_seq):
        out = _predictor(df_feat, df_seq, labels, df_scales)(query_seq, 50)
        assert 0.0 <= out.attrs["proba"] <= 1.0

    def test_deterministic_with_seed(self, df_feat, df_seq, labels, df_scales, query_seq):
        pred = _predictor(df_feat, df_seq, labels, df_scales)
        a, b = pred(query_seq, 50), pred(query_seq, 50)
        assert np.allclose(a[ut.COL_FEAT_IMPACT], b[ut.COL_FEAT_IMPACT])
        assert a.attrs["proba"] == pytest.approx(b.attrs["proba"])

    def test_different_site_differs(self, df_feat, df_seq, labels, df_scales, query_seq):
        pred = _predictor(df_feat, df_seq, labels, df_scales)
        assert not np.allclose(pred(query_seq, 50)[ut.COL_FEAT_IMPACT],
                               pred(query_seq, 70)[ut.COL_FEAT_IMPACT])

    def test_does_not_mutate_template(self, df_feat, df_seq, labels, df_scales, query_seq):
        before = df_feat.copy()
        _predictor(df_feat, df_seq, labels, df_scales)(query_seq, 50)
        pd.testing.assert_frame_equal(df_feat, before)

    @pytest.mark.parametrize("model", [ut.MODEL_RF, ut.MODEL_SVM, ut.MODEL_LOG_REG,
                                       "extra_trees"])
    def test_model_names(self, df_feat, df_seq, labels, df_scales, query_seq, model):
        out = _predictor(df_feat, df_seq, labels, df_scales, model=model)(query_seq, 50)
        assert ut.COL_FEAT_IMPACT in out.columns and 0.0 <= out.attrs["proba"] <= 1.0

    def test_model_estimator_instance(self, df_feat, df_seq, labels, df_scales, query_seq):
        out = _predictor(df_feat, df_seq, labels, df_scales,
                         model=RandomForestClassifier())(query_seq, 50)
        assert 0.0 <= out.attrs["proba"] <= 1.0

    def test_model_list_averages(self, df_feat, df_seq, labels, df_scales, query_seq):
        out = _predictor(df_feat, df_seq, labels, df_scales,
                         model=[ut.MODEL_RF, ut.MODEL_SVM])(query_seq, 50)
        assert 0.0 <= out.attrs["proba"] <= 1.0

    def test_custom_col_imp(self, df_feat, df_seq, labels, df_scales, query_seq):
        out = _predictor(df_feat, df_seq, labels, df_scales, col_imp="my_imp")(query_seq, 50)
        assert "my_imp" in out.columns


# --- parity (KPI) ------------------------------------------------------------
class TestParity:
    """The per-site impact equals the explicit feature_matrix -> ShapModel chain (epic KPI)."""

    def test_matches_manual_chain(self, df_feat, df_seq, labels, df_scales, query_seq):
        from aaanalysis.feature_engineering import SequenceFeature
        from aaanalysis.explainable_ai_pro import ShapModel
        p1, tmd_len, jmd_n, jmd_c = 50, 20, 10, 10
        features = df_feat[ut.COL_FEATURE]
        list_parts = sorted({f.split("-")[0].lower() for f in features})
        sf = SequenceFeature(verbose=False)
        # Built-in result
        out = _predictor(df_feat, df_seq, labels, df_scales)(query_seq, p1)
        # Manual reference: same window geometry, same fuzzy refit
        X_train = np.asarray(sf.feature_matrix(
            features=features,
            df_parts=sf.get_df_parts(df_seq=df_seq, list_parts=list_parts,
                                     jmd_n_len=jmd_n, jmd_c_len=jmd_c),
            df_scales=df_scales))
        df_seq_q = pd.DataFrame({ut.COL_ENTRY: ["__QUERY__"], ut.COL_SEQ: [query_seq],
                                 ut.COL_TMD_START: [p1], ut.COL_TMD_STOP: [p1 + tmd_len - 1]})
        x_q = np.asarray(sf.feature_matrix(
            features=features,
            df_parts=sf.get_df_parts(df_seq=df_seq_q, list_parts=list_parts,
                                     jmd_n_len=jmd_n, jmd_c_len=jmd_c),
            df_scales=df_scales, n_jobs=1))
        from sklearn.ensemble import RandomForestClassifier as RF
        est = RF(random_state=42).fit(X_train, np.asarray(labels))
        proba = float(est.predict_proba(x_q)[0, list(est.classes_).index(1)])
        X_ext = np.vstack([X_train, x_q])
        labels_ext = np.append(np.asarray(labels), 1)
        df_seq_ext = pd.DataFrame({ut.COL_ENTRY: list(df_seq[ut.COL_ENTRY]) + ["__QUERY__"]})
        sm = ShapModel(random_state=42, verbose=False)
        sm.fit(X_ext, labels=labels_ext, label_target_class=1, fuzzy_labeling=True,
               fuzzy_aggregation="interpolate", n_rounds=1, df_seq=df_seq_ext,
               fuzzy_labels={"__QUERY__": proba})
        ref = sm.add_feat_impact(df_feat=df_feat.copy(), samples="__QUERY__", names="__QUERY__",
                                 df_seq=df_seq_ext, drop=True)
        assert out.attrs["proba"] == pytest.approx(proba)
        assert np.allclose(out[ut.COL_FEAT_IMPACT],
                           ref[f"{ut.COL_FEAT_IMPACT}___QUERY__"])


# --- resolve_estimators ------------------------------------------------------
class TestResolveEstimators:
    """The model= resolver (name / estimator / list)."""

    def test_name_rf(self):
        est = resolve_estimators(ut.MODEL_RF, random_state=0)
        assert len(est) == 1 and isinstance(est[0], RandomForestClassifier)

    def test_name_svm(self):
        assert isinstance(resolve_estimators(ut.MODEL_SVM)[0], SVC)

    def test_estimator_cloned(self):
        src = ExtraTreesClassifier()
        out = resolve_estimators(src)[0]
        assert isinstance(out, ExtraTreesClassifier) and out is not src

    def test_list_mixed(self):
        out = resolve_estimators([ut.MODEL_RF, ExtraTreesClassifier()])
        assert len(out) == 2

    def test_random_state_injected(self):
        assert resolve_estimators(ut.MODEL_RF, random_state=7)[0].random_state == 7

    # negative
    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            resolve_estimators("not_a_model")

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            resolve_estimators([])

    def test_bad_type_raises(self):
        with pytest.raises(ValueError):
            resolve_estimators(123)


# --- builder guards ----------------------------------------------------------
class TestBuilderGuards:
    """Build-time validation in build_builtin_predictor."""

    def test_label_target_class_absent(self, df_feat, df_seq, df_scales):
        with pytest.raises(ValueError, match="label_target_class"):
            build_builtin_predictor(df_feat=df_feat, df_seq=df_seq, labels=[0] * len(df_seq),
                                    tmd_len=20, jmd_n_len=10, jmd_c_len=10, df_scales=df_scales,
                                    label_target_class=1)

    def test_reserved_query_entry(self, df_feat, df_seq, labels, df_scales):
        bad = df_seq.copy()
        bad.loc[0, ut.COL_ENTRY] = "__QUERY__"
        with pytest.raises(ValueError, match="reserved"):
            build_builtin_predictor(df_feat=df_feat, df_seq=bad, labels=labels, tmd_len=20,
                                    jmd_n_len=10, jmd_c_len=10, df_scales=df_scales)

    def test_col_imp_collision_overwritten(self, df_feat, df_seq, labels, df_scales, query_seq):
        # A df_feat that already carries the col_imp column gets the fresh per-site impact, not a
        # stale value; the template itself is never mutated (the predictor returns a copy).
        templ = df_feat.copy()
        templ[ut.COL_FEAT_IMPACT] = 999.0
        out = _predictor(templ, df_seq, labels, df_scales)(query_seq, 50)
        assert (out[ut.COL_FEAT_IMPACT] != 999.0).all()
        assert (templ[ut.COL_FEAT_IMPACT] == 999.0).all()   # template untouched
