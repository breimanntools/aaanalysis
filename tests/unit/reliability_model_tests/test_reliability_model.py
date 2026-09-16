"""Unit tests for ReliabilityModel (prediction-reliability measures)."""
import inspect
import warnings
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
from scipy.stats import norm
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.prediction._backend.reliability.reliability import (
    apply_applicability_domain, comp_ad_status, comp_calibration_bins, comp_brier, comp_ece)

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

# The 14 columns predict returned before the applicability-domain columns were appended.
_COLS_LEGACY = ["score", "score_std", "ci_low", "ci_high", "ood_score", "in_domain", "ad_knn",
                "ad_mahalanobis", "ad_leverage", "score_calibrated", "margin", "entropy",
                "conformal_set", "reliable"]


def _data(n=120, n_features=8, seed=0):
    X, y = make_classification(n_samples=n, n_features=n_features, n_informative=5,
                               n_redundant=1, random_state=seed)
    return X[:90], y[:90], X[90:]


def _ood_point(X_train):
    return (X_train.mean(axis=0) + 20.0)[None, :]


def _mixed_new(Xtr, Xte):
    """Held-out rows plus stretched rows so all three finite statuses tend to occur."""
    return np.vstack([Xte, Xte * 1.3, Xte * 1.6, _ood_point(Xtr)])


def _hand_threshold(Xtr, k=5, percentile=95.0):
    Z = StandardScaler().fit_transform(Xtr)
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)
    return float(np.percentile(d[:, 1:].mean(axis=1), percentile))


def _status_from_score(ood, borderline):
    out = np.full(len(ood), "unknown", dtype=object)
    finite = np.isfinite(ood)
    out[finite & (ood <= 1)] = "inside"
    out[finite & (ood > 1) & (ood <= 1 + borderline)] = "borderline"
    out[finite & (ood > 1 + borderline)] = "outside"
    return out


def _degenerate_data(n_features=8, n_copies=20):
    distinct = np.array([[0.0] * n_features, [1.0] * n_features, [2.0] * n_features])
    Xtr = np.repeat(distinct, n_copies, axis=0)
    ytr = np.array([0, 1] * (len(Xtr) // 2))
    return Xtr, ytr


def _miscal_data(seed=0):
    """Redundant features make Gaussian naive Bayes strongly over-confident (mis-calibrated)."""
    X, y = make_classification(n_samples=600, n_features=20, n_informative=3, n_redundant=15,
                               class_sep=0.8, random_state=seed)
    return X[:400], y[:400], X[400:], y[400:]


def _failed_calibration_data():
    """Binary data whose positive class has ONE member, so the internal cv=2 calibration fails."""
    X, y = make_classification(n_samples=40, n_features=5, n_informative=3, random_state=0)
    y = np.zeros(len(y), dtype=int)
    y[0] = 1
    return X, y


def _metric(df_eval, name):
    return float(df_eval.loc[df_eval["bin"] == name, "mean_score"].iloc[0])


def _legacy_eval(rm, X, labels, n_bins=5):
    """Verbatim reference of the pre-1.2.0 eval body (raw score, no metric rows)."""
    y = (np.asarray(labels) == rm.label_pos_).astype(int)
    df = rm.predict(X)
    s = df["score"].to_numpy()
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for b in range(n_bins):
        m = (s >= edges[b]) & (s <= edges[b + 1] if b == n_bins - 1 else s < edges[b + 1])
        rows.append([f"{edges[b]:.2f}-{edges[b+1]:.2f}",
                     float(np.mean(s[m])) if m.any() else np.nan,
                     float(np.mean(y[m])) if m.any() else np.nan,
                     int(m.sum())])
    sets = df["conformal_set"].to_numpy()
    covered = (np.isin(sets, ["pos", "both"]) & (y == 1)) | (np.isin(sets, ["neg", "both"]) & (y == 0))
    rows.append(["summary", float(np.mean(df["in_domain"])), float(np.mean(covered)), len(X)])
    return pd.DataFrame(rows, columns=["bin", "mean_score", "empirical_pos", "n_samples"])


@pytest.fixture(scope="module")
def rm_miscal():
    Xtr, ytr, Xte, yte = _miscal_data()
    rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, model=GaussianNB(), n_bootstrap=0)
    return rm, Xte, yte


@pytest.fixture(scope="module")
def rm_uncal():
    Xtr, ytr, Xte, yte = _miscal_data()
    rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, model=GaussianNB(), n_bootstrap=0,
                                                calibrate=False)
    return rm, Xte, yte


@pytest.fixture(scope="module")
def rm_ad():
    """A default fit plus new rows spanning the inside / borderline / outside bands."""
    Xtr, ytr, Xte = _data()
    rm = aa.ReliabilityModel(random_state=0, verbose=False).fit(Xtr, ytr, n_bootstrap=3)
    return rm, Xtr, _mixed_new(Xtr, Xte)


# I __init__
class TestReliabilityModelInit:
    def test_returns_instance(self):
        assert isinstance(aa.ReliabilityModel(), aa.ReliabilityModel)

    def test_random_state_valid(self):
        assert isinstance(aa.ReliabilityModel(random_state=42), aa.ReliabilityModel)

    @pytest.mark.parametrize("rs", [-1, 1.5, "x"])
    def test_random_state_invalid(self, rs):
        with pytest.raises(ValueError, match="'random_state'"):
            aa.ReliabilityModel(random_state=rs)

    @pytest.mark.parametrize("v", [None, "yes", 3])
    def test_verbose_invalid(self, v):
        with pytest.raises(ValueError, match="'verbose'"):
            aa.ReliabilityModel(verbose=v)


# II fit
class TestFit:
    # X / labels / model / label_pos
    def test_fit_default_returns_self(self):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(random_state=0)
        assert rm.fit(Xtr, ytr) is rm
        assert rm.model_ is not None and rm.label_pos_ == 1

    def test_fit_with_fitted_estimator(self):
        Xtr, ytr, _ = _data()
        est = LogisticRegression(max_iter=500).fit(Xtr, ytr)
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, model=est)
        assert rm.model_ is est

    def test_fit_with_ensemble_list(self):
        Xtr, ytr, _ = _data()
        models = [RandomForestClassifier(n_estimators=20, random_state=i).fit(Xtr, ytr)
                  for i in range(3)]
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, model=models)
        assert isinstance(rm.model_, list) and len(rm.model_) == 3

    def test_fit_with_aapred_like(self):
        Xtr, ytr, _ = _data()

        class _Pred:                                         # duck-typed AAPred (has list_models_)
            list_models_ = [RandomForestClassifier(n_estimators=15, random_state=0).fit(Xtr, ytr)]
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, model=_Pred())
        assert isinstance(rm.model_, list)

    @pytest.mark.parametrize("label_pos", [0, 1])
    def test_label_pos_valid(self, label_pos):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, label_pos=label_pos, n_bootstrap=3)
        assert rm.label_pos_ == label_pos

    def test_label_pos_absent_raises(self):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="'label_pos'"):
            aa.ReliabilityModel().fit(Xtr, ytr, label_pos=7)

    def test_X_labels_mismatch_raises(self):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="n_samples does not match"):
            aa.ReliabilityModel().fit(Xtr, ytr[:-3])

    @pytest.mark.parametrize("X", ["abc", 5, [[1, 2], [3]]])
    def test_X_invalid(self, X):
        _, ytr, _ = _data()
        with pytest.raises(ValueError, match="'X'"):
            aa.ReliabilityModel().fit(X, ytr)

    @pytest.mark.parametrize("labels", [[1] * 90, ["a", "b"] * 45, 5])
    def test_labels_invalid(self, labels):
        Xtr, _, _ = _data()
        with pytest.raises(ValueError, match="labels"):
            aa.ReliabilityModel().fit(Xtr, labels)

    def test_empty_model_list_raises(self):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="'model'"):
            aa.ReliabilityModel().fit(Xtr, ytr, model=[])

    def test_non_binary_labels_raises(self):
        X, y = make_classification(n_samples=90, n_features=8, n_informative=5, n_classes=3,
                                   n_clusters_per_class=1, random_state=0)
        with pytest.raises(ValueError, match="binary labels"):
            aa.ReliabilityModel().fit(X, y)

    def test_model_without_predict_proba_raises(self):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="'model'"):          # SVC() has no predict_proba
            aa.ReliabilityModel().fit(Xtr, ytr, model=SVC().fit(Xtr, ytr))

    def test_unfitted_aapred_rejected(self):
        Xtr, ytr, _ = _data()

        class _Pred:
            list_models_ = None                              # unfitted AAPred
        with pytest.raises(ValueError, match="not fitted"):
            aa.ReliabilityModel().fit(Xtr, ytr, model=_Pred())

    # k
    @pytest.mark.parametrize("k", [1, 3, 10])
    def test_k_valid(self, k):
        Xtr, ytr, _ = _data()
        assert aa.ReliabilityModel().fit(Xtr, ytr, k=k, n_bootstrap=3) is not None

    @pytest.mark.parametrize("k", [0, -1, 2.5])
    def test_k_invalid(self, k):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="'k'"):
            aa.ReliabilityModel().fit(Xtr, ytr, k=k)

    # ad_percentile
    @pytest.mark.parametrize("p", [1, 50, 100])
    def test_ad_percentile_valid(self, p):
        Xtr, ytr, _ = _data()
        assert aa.ReliabilityModel().fit(Xtr, ytr, ad_percentile=p, n_bootstrap=3) is not None

    @pytest.mark.parametrize("p", [0, 101, -5])
    def test_ad_percentile_invalid(self, p):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="'ad_percentile'"):
            aa.ReliabilityModel().fit(Xtr, ytr, ad_percentile=p)

    # ad_borderline (+ the fitted ad_threshold_ / ad_method_ it bands)
    @settings(max_examples=5, deadline=None)
    @given(b=some.floats(min_value=0.0, max_value=3.0, allow_nan=False))
    def test_ad_borderline_valid_float(self, b):
        Xtr, ytr, Xte = _data()
        rm = aa.ReliabilityModel(random_state=0, verbose=False).fit(Xtr, ytr, ad_borderline=b,
                                                                   n_bootstrap=0, calibrate=False)
        df = rm.predict(_mixed_new(Xtr, Xte))
        assert set(df["ad_status"]) <= set(ut.LIST_AD_STATUS)

    @settings(max_examples=5, deadline=None)
    @given(b=some.integers(min_value=0, max_value=5))
    def test_ad_borderline_valid_int(self, b):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(verbose=False).fit(Xtr, ytr, ad_borderline=b, n_bootstrap=0,
                                                    calibrate=False)
        assert rm.ad_threshold_ > 0

    def test_ad_borderline_numpy_float(self):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(verbose=False).fit(Xtr, ytr, ad_borderline=np.float64(0.2),
                                                    n_bootstrap=0, calibrate=False)
        assert rm.ad_method_ == "knn"

    def test_ad_borderline_does_not_change_threshold(self):
        Xtr, ytr, _ = _data()
        kw = dict(n_bootstrap=0, calibrate=False)
        a = aa.ReliabilityModel(verbose=False).fit(Xtr, ytr, ad_borderline=0.0, **kw)
        b = aa.ReliabilityModel(verbose=False).fit(Xtr, ytr, ad_borderline=2.0, **kw)
        assert a.ad_threshold_ == b.ad_threshold_

    def test_attributes_none_before_fit(self):
        rm = aa.ReliabilityModel()
        assert rm.ad_threshold_ is None and rm.ad_method_ is None

    def test_attributes_set_after_fit(self, rm_ad):
        rm, _, _ = rm_ad
        assert isinstance(rm.ad_threshold_, float) and rm.ad_threshold_ > 0
        assert rm.ad_method_ == ut.STR_AD_METHOD_KNN == "knn"

    def test_ad_borderline_is_keyword_only(self):
        p = inspect.signature(aa.ReliabilityModel.fit).parameters["ad_borderline"]
        assert p.kind == inspect.Parameter.KEYWORD_ONLY and p.default == 0.1

    @pytest.mark.parametrize("b", [-0.1, -1, -1e-12])
    def test_ad_borderline_negative_raises(self, b):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="ad_borderline"):
            aa.ReliabilityModel().fit(Xtr, ytr, ad_borderline=b)

    @pytest.mark.parametrize("b", [None, "0.1", [0.1], {"b": 1}])
    def test_ad_borderline_wrong_type_raises(self, b):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="ad_borderline"):
            aa.ReliabilityModel().fit(Xtr, ytr, ad_borderline=b)

    @pytest.mark.parametrize("b", [True, False])
    def test_ad_borderline_bool_raises(self, b):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="ad_borderline"):
            aa.ReliabilityModel().fit(Xtr, ytr, ad_borderline=b)

    @pytest.mark.parametrize("b", [np.nan, np.inf, -np.inf])
    def test_ad_borderline_non_finite_raises(self, b):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="ad_borderline"):
            aa.ReliabilityModel().fit(Xtr, ytr, ad_borderline=b)

    def test_ad_borderline_non_finite_message(self):
        # House message format: the supplied value first, then the requirement.
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError,
                           match=r"'ad_borderline' \(inf\) should be a finite number >= 0"):
            aa.ReliabilityModel().fit(Xtr, ytr, ad_borderline=np.inf)

    def test_ad_borderline_bool_message(self):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError,
                           match=r"'ad_borderline' \(True\) should be a finite number >= 0"):
            aa.ReliabilityModel().fit(Xtr, ytr, ad_borderline=True)

    def test_invalid_ad_borderline_leaves_instance_unfitted(self):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel()
        with pytest.raises(ValueError, match="'ad_borderline'"):
            rm.fit(Xtr, ytr, ad_borderline=-1)
        assert rm.ad_threshold_ is None
        with pytest.raises(RuntimeError, match="Call 'fit' before 'predict'"):
            rm.predict(Xtr)

    # ci
    @settings(max_examples=5, deadline=None)
    @given(ci=some.floats(min_value=0.01, max_value=0.99))
    def test_ci_valid(self, ci):
        Xtr, ytr, _ = _data()
        assert aa.ReliabilityModel().fit(Xtr, ytr, ci=ci, n_bootstrap=3) is not None

    @pytest.mark.parametrize("ci", [0, 0.0, 1, 1.0, -0.1, 120, "0.9", None])
    def test_ci_invalid(self, ci):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="'ci'"):
            aa.ReliabilityModel().fit(Xtr, ytr, ci=ci)

    @pytest.mark.parametrize("ci", [50, 90, 90.0, 99])
    def test_ci_percent_rejected_with_hint(self, ci):
        # 'ci' is a fraction (like ModelEvaluator.run / comp_bootstrap_ci); a percent gets a hint.
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="should be a fraction"):
            aa.ReliabilityModel().fit(Xtr, ytr, ci=ci)

    def test_ci_default_is_fraction(self):
        assert inspect.signature(aa.ReliabilityModel.fit).parameters["ci"].default == 0.90

    def test_ci_default_equals_explicit(self):
        Xtr, ytr, Xte = _data()
        d_default = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=5).predict(Xte)
        d_explicit = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=5,
                                                             ci=0.90).predict(Xte)
        pd.testing.assert_frame_equal(d_default, d_explicit)

    def test_ci_wider_interval_for_larger_ci(self):
        Xtr, ytr, Xte = _data()
        narrow = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=5, ci=0.5).predict(Xte)
        wide = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=5, ci=0.99).predict(Xte)
        w_narrow = narrow["ci_high"] - narrow["ci_low"]
        w_wide = wide["ci_high"] - wide["ci_low"]
        assert (w_wide >= w_narrow - 1e-12).all() and (w_wide > w_narrow).any()

    @pytest.mark.parametrize("ci", [float("nan"), np.nan, np.float64("nan"), float("inf"),
                                    float("-inf"), np.float64("inf")])
    def test_ci_non_finite_raises(self, ci):
        # A non-finite value passes every range comparison ('nan < 0' and 'nan > 1' are each
        # False), so it would otherwise reach norm.ppf and yield NaN 'ci_low' / 'ci_high'.
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="finite"):
            aa.ReliabilityModel().fit(Xtr, ytr, ci=ci, n_bootstrap=3)

    @pytest.mark.parametrize("ci", [0.01, 0.5, 0.99])
    def test_ci_bounds_are_finite(self, ci):
        Xtr, ytr, Xte = _data()
        df = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=5, ci=ci).predict(Xte)
        assert np.isfinite(df[["ci_low", "ci_high", "score", "score_std"]].to_numpy()).all()

    @pytest.mark.parametrize("p", [float("nan"), np.float64("nan"), float("inf"), float("-inf")])
    def test_ad_percentile_non_finite_raises(self, p):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="finite"):
            aa.ReliabilityModel().fit(Xtr, ytr, ad_percentile=p)

    def test_ad_percentile_non_finite_message(self):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError,
                           match=r"'ad_percentile' \(inf\) should be a finite float or an integer"):
            aa.ReliabilityModel().fit(Xtr, ytr, ad_percentile=np.inf)

    @pytest.mark.parametrize("a", [float("nan"), np.float64("nan"), float("inf"), float("-inf")])
    def test_conformal_alpha_non_finite_raises(self, a):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="finite"):
            aa.ReliabilityModel().fit(Xtr, ytr, conformal_alpha=a)

    @settings(max_examples=5, deadline=None)
    @given(conformal_alpha=some.floats(min_value=0.01, max_value=0.5))
    def test_conformal_alpha_valid(self, conformal_alpha):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=3,
                                                     conformal_alpha=conformal_alpha)
        assert rm is not None

    @pytest.mark.parametrize("val", [float("nan"), float("inf")])
    def test_k_non_finite_raises(self, val):
        # Integer parameters reject a non-finite float by type, before any range comparison.
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="should be an integer"):
            aa.ReliabilityModel().fit(Xtr, ytr, k=val)

    @pytest.mark.parametrize("val", [float("nan"), float("inf")])
    def test_n_bootstrap_non_finite_raises(self, val):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="should be an integer"):
            aa.ReliabilityModel().fit(Xtr, ytr, n_bootstrap=val)
    # n_bootstrap
    @settings(max_examples=3, deadline=None)
    @given(nb=some.integers(min_value=0, max_value=5))
    def test_n_bootstrap_valid(self, nb):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(random_state=0, verbose=False).fit(Xtr, ytr, n_bootstrap=nb,
                                                                    calibrate=False)
        assert rm.model_ is not None

    @pytest.mark.parametrize("nb", [-1, 2.5])
    def test_n_bootstrap_invalid(self, nb):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="'n_bootstrap'"):
            aa.ReliabilityModel().fit(Xtr, ytr, n_bootstrap=nb)

    # calibrate / calibration_method
    @pytest.mark.parametrize("calibrate", [True, False])
    def test_calibrate_flag(self, calibrate):
        Xtr, ytr, Xte = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, calibrate=calibrate, n_bootstrap=3)
        df = rm.predict(Xte)
        assert df["score_calibrated"].notna().any() if calibrate else df["score_calibrated"].isna().all()

    @pytest.mark.parametrize("val", [None, "yes", 1, 0, [True]])
    def test_calibrate_invalid(self, val):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="'calibrate'"):
            aa.ReliabilityModel().fit(Xtr, ytr, calibrate=val)

    def test_calibration_method_invalid(self):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="calibration_method"):
            aa.ReliabilityModel().fit(Xtr, ytr, calibration_method="bogus")

    @pytest.mark.parametrize("m", ["isotonic", "sigmoid"])
    def test_calibration_method_valid(self, m):
        Xtr, ytr, _ = _data()
        assert aa.ReliabilityModel().fit(Xtr, ytr, calibration_method=m, n_bootstrap=3) is not None

    # conformal_alpha
    @settings(max_examples=3, deadline=None)
    @given(a=some.floats(min_value=0.02, max_value=0.5))
    def test_conformal_alpha_valid(self, a):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(random_state=0, verbose=False).fit(Xtr, ytr, conformal_alpha=a,
                                                                    n_bootstrap=0, calibrate=False)
        assert rm.predict(Xtr[:5])["conformal_set"].isin(["neg", "pos", "both", "none"]).all()

    @pytest.mark.parametrize("a", [-0.1, 1.5])
    def test_conformal_alpha_invalid(self, a):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="conformal_alpha"):
            aa.ReliabilityModel().fit(Xtr, ytr, conformal_alpha=a)


class TestFitComplex:
    """Fit parameters crossed with each other: the applicability-domain leakage
    contract and the calibration outcome."""

    def test_ad_borderline_zero_has_no_borderline(self, rm_ad):
        _, _, Xnew = rm_ad
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(random_state=0, verbose=False).fit(Xtr, ytr, ad_borderline=0,
                                                                    n_bootstrap=3)
        assert "borderline" not in set(rm.predict(Xnew)["ad_status"])

    def test_ad_borderline_default_equals_explicit(self, rm_ad):
        rm, _, Xnew = rm_ad
        Xtr, ytr, _ = _data()
        rm_exp = aa.ReliabilityModel(random_state=0, verbose=False).fit(Xtr, ytr, ad_borderline=0.1,
                                                                       n_bootstrap=3)
        pd.testing.assert_frame_equal(rm.predict(Xnew), rm_exp.predict(Xnew))

    @settings(max_examples=5, deadline=None)
    @given(b1=some.floats(min_value=0.0, max_value=1.0), extra=some.floats(min_value=0.0, max_value=1.0))
    def test_wider_band_moves_outside_to_borderline(self, b1, extra):
        Xtr, ytr, Xte = _data()
        Xnew = _mixed_new(Xtr, Xte)
        kw = dict(n_bootstrap=0, calibrate=False)
        s1 = aa.ReliabilityModel(verbose=False).fit(Xtr, ytr, ad_borderline=b1, **kw).predict(Xnew)["ad_status"]
        s2 = aa.ReliabilityModel(verbose=False).fit(Xtr, ytr, ad_borderline=b1 + extra, **kw).predict(Xnew)["ad_status"]
        assert (s2 == "outside").sum() <= (s1 == "outside").sum()
        assert (s2 == "borderline").sum() >= (s1 == "borderline").sum()
        assert ((s1 == "inside") == (s2 == "inside")).all()

    def test_leakage_fold_reference_differs_from_full(self):
        Xtr, ytr, _ = _data()
        kw = dict(n_bootstrap=0, calibrate=False)
        full = aa.ReliabilityModel(random_state=0, verbose=False).fit(Xtr, ytr, **kw)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for train_idx, test_idx in skf.split(Xtr, ytr):
            fold = aa.ReliabilityModel(random_state=0, verbose=False).fit(
                Xtr[train_idx], ytr[train_idx], **kw)
            assert fold.ad_threshold_ != full.ad_threshold_
            assert fold.ad_threshold_ == pytest.approx(_hand_threshold(Xtr[train_idx]), abs=1e-12)

    def test_leakage_test_rows_do_not_influence_reference(self):
        Xtr, ytr, _ = _data()
        kw = dict(n_bootstrap=0, calibrate=False)
        train_idx, test_idx = next(StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
                                   .split(Xtr, ytr))
        X_alt = Xtr.copy()
        X_alt[test_idx] = np.random.default_rng(0).normal(50, 10, size=X_alt[test_idx].shape)
        a = aa.ReliabilityModel(random_state=0, verbose=False).fit(Xtr[train_idx], ytr[train_idx], **kw)
        b = aa.ReliabilityModel(random_state=0, verbose=False).fit(X_alt[train_idx], ytr[train_idx], **kw)
        assert a.ad_threshold_ == b.ad_threshold_
        thr_before = a.ad_threshold_
        da, db = a.predict(Xtr[test_idx]), b.predict(Xtr[test_idx])
        pd.testing.assert_frame_equal(da[["ood_score", "ad_status", "ad_nearest_train"]],
                                      db[["ood_score", "ad_status", "ad_nearest_train"]])
        assert a.ad_threshold_ == thr_before                     # predicting does not refit
        assert da["ad_nearest_train"].between(0, len(train_idx) - 1).all()

    @pytest.mark.parametrize("p_lo, p_hi", [(50, 90), (80, 99), (5, 100)])
    def test_ad_percentile_orders_threshold_and_inside(self, p_lo, p_hi):
        Xtr, ytr, Xte = _data()
        Xnew = _mixed_new(Xtr, Xte)
        kw = dict(n_bootstrap=0, calibrate=False)
        lo = aa.ReliabilityModel(verbose=False).fit(Xtr, ytr, ad_percentile=p_lo, **kw)
        hi = aa.ReliabilityModel(verbose=False).fit(Xtr, ytr, ad_percentile=p_hi, **kw)
        assert lo.ad_threshold_ < hi.ad_threshold_
        assert (lo.predict(Xnew)["ad_status"] == "inside").sum() <= \
               (hi.predict(Xnew)["ad_status"] == "inside").sum()

    @pytest.mark.parametrize("k", [1, 3, 7])
    def test_k_changes_threshold_consistently(self, k):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(verbose=False).fit(Xtr, ytr, k=k, n_bootstrap=0, calibrate=False)
        assert rm.ad_threshold_ == pytest.approx(_hand_threshold(Xtr, k=k), abs=1e-12)

    def test_invalid_ad_borderline_with_invalid_k_raises(self):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match=r"'k'|'ad_borderline'"):
            aa.ReliabilityModel().fit(Xtr, ytr, k=0, ad_borderline=-1)

    def test_invalid_ad_borderline_with_valid_rest_raises(self):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="ad_borderline"):
            aa.ReliabilityModel().fit(Xtr, ytr, k=3, ad_percentile=90, ci=0.8, ad_borderline=-0.5)

    def test_invalid_ad_borderline_with_ensemble_raises(self):
        Xtr, ytr, _ = _data()
        models = [LogisticRegression(max_iter=300).fit(Xtr, ytr)]
        with pytest.raises(ValueError, match="ad_borderline"):
            aa.ReliabilityModel().fit(Xtr, ytr, model=models, ad_borderline=np.nan)

    def test_percent_ci_with_valid_ad_borderline_raises(self):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="should be a fraction"):
            aa.ReliabilityModel().fit(Xtr, ytr, ci=90, ad_borderline=0.2)

    def test_non_binary_labels_with_ad_params_raises(self):
        X, y = make_classification(n_samples=90, n_features=8, n_informative=5, n_classes=3,
                                   n_clusters_per_class=1, random_state=0)
        with pytest.raises(ValueError, match="binary labels"):
            aa.ReliabilityModel().fit(X, y, ad_percentile=90, ad_borderline=0.3)

    def test_invalid_calibrate_with_ensemble_raises(self):
        Xtr, ytr, _ = _data()
        models = [LogisticRegression(max_iter=300).fit(Xtr, ytr)]
        with pytest.raises(ValueError, match="'calibrate'"):
            aa.ReliabilityModel().fit(Xtr, ytr, model=models, calibrate="yes")

    # Calibration outcome at fit time (the eval-time message is TestEvalComplex's job)
    def test_failed_calibration_warns_at_fit(self):
        X, y = _failed_calibration_data()
        with pytest.warns(UserWarning, match=r"'calibrate' \(True\) could not be applied"):
            rm = aa.ReliabilityModel(random_state=0).fit(X, y, n_bootstrap=0)
        assert rm._calibrator is None
        assert rm._calibrate_requested is True
        assert "2-fold" in rm._calibration_error

    def test_failed_calibration_leaves_score_calibrated_nan(self):
        X, y = _failed_calibration_data()
        with pytest.warns(UserWarning):
            rm = aa.ReliabilityModel(random_state=0).fit(X, y, n_bootstrap=0)
        assert rm.predict(X)["score_calibrated"].isna().all()

    def test_successful_calibration_is_silent(self):
        Xtr, ytr, _ = _data()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=3)
        assert rm._calibrator is not None and rm._calibration_error is None

    def test_calibrate_false_records_no_error(self, rm_uncal):
        rm, _, _ = rm_uncal
        assert rm._calibrate_requested is False and rm._calibration_error is None

    # Positive interactions
    @pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
    def test_ensemble_is_calibrated_with_either_method(self, method):
        Xtr, ytr, Xte = _data()
        models = [LogisticRegression(max_iter=500).fit(Xtr, ytr),
                  RandomForestClassifier(n_estimators=20, random_state=0).fit(Xtr, ytr)]
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, model=models, calibrate=True,
                                                     calibration_method=method)
        assert rm._calibrator is not None and rm._calibration_error is None
        assert rm.predict(Xte)["score_calibrated"].notna().all()

    def test_calibration_method_is_inert_when_calibrate_is_false(self):
        Xtr, ytr, Xte = _data()
        kw = dict(n_bootstrap=3, calibrate=False)
        iso = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, calibration_method="isotonic", **kw)
        sig = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, calibration_method="sigmoid", **kw)
        assert iso._calibrator is None and sig._calibrator is None
        pd.testing.assert_frame_equal(iso.predict(Xte), sig.predict(Xte))

    @pytest.mark.parametrize("label_pos", [0, 1])
    def test_label_pos_crossed_with_calibration_scores_that_class(self, label_pos):
        Xtr, ytr, Xte = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, label_pos=label_pos,
                                                     n_bootstrap=3, calibrate=True)
        df = rm.predict(Xte)
        assert rm.label_pos_ == label_pos
        assert df["score_calibrated"].between(0, 1).all()

    @settings(max_examples=3, deadline=None)
    @given(nb=some.integers(min_value=2, max_value=6))
    def test_bootstrap_and_calibration_combine(self, nb):
        Xtr, ytr, Xte = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=nb, calibrate=True,
                                                     ci=0.8)
        df = rm.predict(Xte)
        assert rm._calibrator is not None
        assert (df["score_std"] >= 0).all() and (df["ci_high"] >= df["ci_low"]).all()

    def test_ci_widens_while_calibrated_score_is_unchanged(self):
        Xtr, ytr, Xte = _data()
        kw = dict(n_bootstrap=5, calibrate=True)
        narrow = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, ci=0.5, **kw).predict(Xte)
        wide = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, ci=0.99, **kw).predict(Xte)
        w_narrow = narrow["ci_high"] - narrow["ci_low"]
        w_wide = wide["ci_high"] - wide["ci_low"]
        assert (w_wide >= w_narrow - 1e-12).all()
        np.testing.assert_allclose(narrow["score_calibrated"], wide["score_calibrated"], atol=1e-12)

    # Negative interactions
    def test_invalid_calibration_method_with_valid_calibrate_raises(self):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="calibration_method"):
            aa.ReliabilityModel().fit(Xtr, ytr, calibrate=True, calibration_method="platt")

    @pytest.mark.parametrize("val", [None, "yes", 1, [True]])
    def test_non_bool_calibrate_with_valid_method_raises(self, val):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="'calibrate'"):
            aa.ReliabilityModel().fit(Xtr, ytr, calibrate=val, calibration_method="sigmoid")

    def test_empty_ensemble_with_calibration_raises(self):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="'model'"):
            aa.ReliabilityModel().fit(Xtr, ytr, model=[], calibrate=True,
                                      calibration_method="sigmoid")

    def test_member_without_predict_proba_with_calibration_raises(self):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="predict_proba"):
            aa.ReliabilityModel().fit(Xtr, ytr, model=[SVC().fit(Xtr, ytr)], calibrate=True)

    def test_unfitted_aapred_with_calibration_raises(self):
        Xtr, ytr, _ = _data()

        class _Pred:
            list_models_ = None                              # unfitted AAPred
        with pytest.raises(ValueError, match="not fitted"):
            aa.ReliabilityModel().fit(Xtr, ytr, model=_Pred(), calibrate=True, n_bootstrap=0)

    def test_label_pos_absent_with_calibration_raises(self):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="'label_pos'"):
            aa.ReliabilityModel().fit(Xtr, ytr, label_pos=2, calibrate=True,
                                      calibration_method="sigmoid")

    @pytest.mark.parametrize("a", [-0.1, 1.5])
    def test_conformal_alpha_out_of_range_with_calibration_raises(self, a):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError, match="'conformal_alpha'"):
            aa.ReliabilityModel().fit(Xtr, ytr, conformal_alpha=a, calibrate=True,
                                      calibration_method="isotonic")


class TestFitGoldenValues:
    """Hand-computed applicability-domain reference learned by fit."""

    def test_threshold_equals_hand_percentile(self):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(verbose=False).fit(Xtr, ytr, k=5, ad_percentile=95.0,
                                                    n_bootstrap=0, calibrate=False)
        assert rm.ad_threshold_ == pytest.approx(_hand_threshold(Xtr, k=5, percentile=95.0), abs=1e-12)

    @pytest.mark.parametrize("k", [1, 5])
    @pytest.mark.parametrize("percentile", [50.0, 95.0, 100.0])
    def test_threshold_grid_equals_hand_percentile(self, k, percentile):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(verbose=False).fit(Xtr, ytr, k=k, ad_percentile=percentile,
                                                    n_bootstrap=0, calibrate=False)
        assert rm.ad_threshold_ == pytest.approx(_hand_threshold(Xtr, k=k, percentile=percentile),
                                                 abs=1e-12)

    def test_threshold_of_diagonal_line(self):
        # Training points (i, i) for i in 0..7 with k=1: after standardization every
        # nearest-neighbour distance is sqrt(2)/std, so ad_threshold_ = sqrt(2)/std.
        line = np.arange(8, dtype=float)
        Xtr = np.column_stack([line, line])
        ytr = np.array([0, 1] * 4)
        rm = aa.ReliabilityModel(random_state=0, verbose=False).fit(
            Xtr, ytr, k=1, n_bootstrap=0, calibrate=False)
        assert rm.ad_threshold_ == pytest.approx(np.sqrt(2) / np.std(line), abs=1e-12)

    def test_degenerate_reference_threshold_is_not_positive(self):
        Xtr, ytr = _degenerate_data()
        rm = aa.ReliabilityModel(random_state=0, verbose=False).fit(Xtr, ytr, n_bootstrap=0,
                                                                    calibrate=False)
        assert rm.ad_threshold_ <= 0                         # no usable spread in the reference


# III predict
class TestPredict:
    def test_columns(self):
        Xtr, ytr, Xte = _data()
        df = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=5).predict(Xte)
        assert isinstance(df, pd.DataFrame) and len(df) == len(Xte)
        for c in ["score", "score_std", "ci_low", "ci_high", "ood_score", "in_domain",
                  "ad_knn", "ad_mahalanobis", "ad_leverage", "score_calibrated",
                  "margin", "entropy", "conformal_set", "reliable",
                  "ad_status", "ad_nearest_train"]:
            assert c in df.columns
        assert df["in_domain"].dtype == bool and df["reliable"].dtype == bool

    def test_columns_order_matches_constant_bundle(self, rm_ad):
        rm, _, Xnew = rm_ad
        df = rm.predict(Xnew)
        assert list(df.columns) == ut.COLS_RELIABILITY
        assert list(df.columns[:14]) == _COLS_LEGACY
        assert list(df.columns[14:]) == ["ad_status", "ad_nearest_train"]

    def test_before_fit_raises(self):
        _, _, Xte = _data()
        with pytest.raises(RuntimeError, match="Call 'fit' before 'predict'"):
            aa.ReliabilityModel().predict(Xte)

    def test_feature_mismatch_raises(self):
        Xtr, ytr, Xte = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=3)
        with pytest.raises(ValueError, match=r"'X' has 4 features"):
            rm.predict(Xte[:, :4])

    @pytest.mark.parametrize("X", [None, "abc", 5, [[1, 2], [3]]])
    def test_X_invalid(self, X):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=3)
        with pytest.raises(ValueError, match="'X'"):
            rm.predict(X)

    def test_ood_point_flagged(self):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=5)
        df = rm.predict(_ood_point(Xtr))
        assert not bool(df["in_domain"].iloc[0])
        assert df["ood_score"].iloc[0] > 1.0
        assert not bool(df["reliable"].iloc[0])

    def test_in_distribution_mostly_in_domain(self):
        Xtr, ytr, Xte = _data()
        df = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=5).predict(Xte)
        assert df["in_domain"].mean() > 0.7

    def test_deterministic(self):
        Xtr, ytr, Xte = _data()
        Xnew = np.vstack([Xte, _ood_point(Xtr)])
        a = aa.ReliabilityModel(random_state=42).fit(Xtr, ytr, n_bootstrap=8).predict(Xnew)
        b = aa.ReliabilityModel(random_state=42).fit(Xtr, ytr, n_bootstrap=8).predict(Xnew)
        assert np.allclose(a["score"], b["score"])
        assert (a["conformal_set"].values == b["conformal_set"].values).all()

    def test_ensemble_std_positive(self):
        Xtr, ytr, Xte = _data()
        models = [RandomForestClassifier(n_estimators=25, random_state=i).fit(Xtr, ytr)
                  for i in range(4)]
        df = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, model=models).predict(Xte)
        assert (df["score_std"] > 0).any()

    def test_no_bootstrap_zero_std(self):
        Xtr, ytr, Xte = _data()
        est = LogisticRegression(max_iter=500).fit(Xtr, ytr)
        df = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, model=est, n_bootstrap=0).predict(Xte)
        assert np.allclose(df["score_std"], 0.0)

    def test_margin_entropy_ranges(self):
        Xtr, ytr, Xte = _data()
        df = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=5).predict(Xte)
        assert df["margin"].between(0, 1).all()
        assert df["entropy"].between(0, 1.0001).all()
        assert df["conformal_set"].isin(["neg", "pos", "both", "none"]).all()

    # ad_status
    def test_ad_status_values_never_null(self, rm_ad):
        rm, _, Xnew = rm_ad
        s = rm.predict(Xnew)["ad_status"]
        assert s.notna().all()
        assert set(s) <= {"inside", "borderline", "outside", "unknown"}
        assert ut.LIST_AD_STATUS == ["inside", "borderline", "outside", "unknown"]

    def test_all_finite_statuses_occur(self, rm_ad):
        rm, _, Xnew = rm_ad
        s = set(rm.predict(Xnew)["ad_status"])
        assert {"inside", "borderline", "outside"} <= s and "unknown" not in s

    @settings(max_examples=5, deadline=None)
    @given(seed=some.integers(min_value=0, max_value=50),
           b=some.floats(min_value=0.0, max_value=1.0))
    def test_in_domain_equals_inside(self, seed, b):
        Xtr, ytr, Xte = _data(seed=seed)
        rm = aa.ReliabilityModel(verbose=False).fit(Xtr, ytr, ad_borderline=b, n_bootstrap=0,
                                                    calibrate=False)
        df = rm.predict(_mixed_new(Xtr, Xte))
        assert (df["in_domain"] == (df["ad_status"] == "inside")).all()

    def test_ood_point_is_outside(self, rm_ad):
        rm, Xtr, _ = rm_ad
        assert rm.predict(_ood_point(Xtr))["ad_status"].iloc[0] == "outside"

    # ad_nearest_train
    def test_nearest_train_dtype_and_range(self, rm_ad):
        rm, Xtr, Xnew = rm_ad
        nn = rm.predict(Xnew)["ad_nearest_train"]
        assert np.issubdtype(nn.dtype, np.integer)
        assert nn.between(0, len(Xtr) - 1).all()

    def test_nearest_train_of_training_rows_is_itself(self, rm_ad):
        rm, Xtr, _ = rm_ad
        nn = rm.predict(Xtr)["ad_nearest_train"].to_numpy()
        np.testing.assert_array_equal(nn, np.arange(len(Xtr)))

    def test_nearest_train_of_perturbed_copy(self, rm_ad):
        rm, Xtr, _ = rm_ad
        idx = [3, 17, 42]
        nn = rm.predict(Xtr[idx] + 1e-6)["ad_nearest_train"].tolist()
        assert nn == idx

    # the legacy columns stay exactly as they were
    @settings(max_examples=3, deadline=None)
    @given(b=some.floats(min_value=0.0, max_value=2.0))
    def test_legacy_columns_unaffected_by_ad_borderline(self, b):
        Xtr, ytr, Xte = _data()
        Xnew = _mixed_new(Xtr, Xte)
        kw = dict(n_bootstrap=3)
        a = aa.ReliabilityModel(random_state=0, verbose=False).fit(Xtr, ytr, **kw).predict(Xnew)
        c = aa.ReliabilityModel(random_state=0, verbose=False).fit(Xtr, ytr, ad_borderline=b,
                                                                  **kw).predict(Xnew)
        pd.testing.assert_frame_equal(a[_COLS_LEGACY], c[_COLS_LEGACY])

    def test_legacy_ad_columns_match_backend(self, rm_ad):
        rm, _, Xnew = rm_ad
        ad = apply_applicability_domain(rm._ad_state, Xnew)
        df = rm.predict(Xnew)
        np.testing.assert_array_equal(df["ood_score"].to_numpy(), ad["ood_score"])
        np.testing.assert_array_equal(df["in_domain"].to_numpy(), ad["in_domain"])
        np.testing.assert_array_equal(df["ad_knn"].to_numpy(), ad["knn"])

    def test_degenerate_reference_is_not_silently_in_domain(self):
        """A training reference with no spread must not wave every sample through.

        A few distinct rows, each heavily duplicated, passes the unique-samples guard but
        still drives every training kNN distance to 0, so the domain boundary collapses.
        Nothing can be established as inside it, and the honest report is 'not in domain'
        with an undefined score -- never a blanket ``in_domain=True``, which is precisely
        the failure an applicability domain exists to prevent.
        """
        n_features, n_copies = 8, 20
        distinct = np.array([[0.0] * n_features, [1.0] * n_features, [2.0] * n_features])
        Xtr = np.repeat(distinct, n_copies, axis=0)
        ytr = np.array([0, 1] * (len(Xtr) // 2))
        Xfar = np.full((3, n_features), 99.0)
        df = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=3).predict(Xfar)
        assert not df["in_domain"].any()
        assert df["ood_score"].isna().all()
        assert not df["reliable"].any()

    def test_degenerate_reference_status_unknown(self):
        Xtr, ytr = _degenerate_data()
        rm = aa.ReliabilityModel(random_state=0, verbose=False).fit(Xtr, ytr, n_bootstrap=3)
        df = rm.predict(np.vstack([np.full((3, 8), 99.0), Xtr[:2]]))
        assert (df["ad_status"] == "unknown").all()
        assert df["ood_score"].isna().all() and not df["in_domain"].any()
        assert (df["in_domain"] == (df["ad_status"] == "inside")).all()
        assert rm.ad_threshold_ <= 0
        assert df["ad_nearest_train"].between(0, len(Xtr) - 1).all()


class TestPredictComplex:
    """The appended columns crossed with the fit parameters and degenerate inputs."""

    def test_ensemble_model_with_ad_borderline(self):
        Xtr, ytr, Xte = _data()
        models = [RandomForestClassifier(n_estimators=10, random_state=i).fit(Xtr, ytr) for i in range(3)]
        df = aa.ReliabilityModel(random_state=0, verbose=False).fit(
            Xtr, ytr, model=models, ad_borderline=0.5).predict(_mixed_new(Xtr, Xte))
        assert list(df.columns) == ut.COLS_RELIABILITY
        assert (df["in_domain"] == (df["ad_status"] == "inside")).all()

    def test_features_exceed_samples_status_is_known(self):
        X, y = make_classification(n_samples=40, n_features=30, n_informative=12, random_state=0)
        rm = aa.ReliabilityModel(random_state=0, verbose=False).fit(X[:14], y[:14], n_bootstrap=3)
        df = rm.predict(X[14:])
        assert df["ad_mahalanobis"].isna().all()
        assert "unknown" not in set(df["ad_status"])

    def test_reliable_implies_inside(self, rm_ad):
        rm, _, Xnew = rm_ad
        df = rm.predict(Xnew)
        assert (df.loc[df["reliable"], "ad_status"] == "inside").all()

    def test_degenerate_with_borderline_band_still_unknown(self):
        Xtr, ytr = _degenerate_data()
        rm = aa.ReliabilityModel(random_state=0, verbose=False).fit(Xtr, ytr, n_bootstrap=0,
                                                                    ad_borderline=100.0)
        assert (rm.predict(Xtr[:4])["ad_status"] == "unknown").all()

    def test_single_row_keeps_full_schema(self, rm_ad):
        rm, Xtr, _ = rm_ad
        df = rm.predict(Xtr[:1])
        assert len(df) == 1 and list(df.columns) == ut.COLS_RELIABILITY
        assert df["ad_status"].iloc[0] in ut.LIST_AD_STATUS

    def test_repeated_predict_is_identical(self, rm_ad):
        rm, _, Xnew = rm_ad
        pd.testing.assert_frame_equal(rm.predict(Xnew), rm.predict(Xnew))

    def test_feature_mismatch_still_raises(self, rm_ad):
        rm, _, Xnew = rm_ad
        with pytest.raises(ValueError, match=r"'X' has 3 features"):
            rm.predict(Xnew[:, :3])

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="Call 'fit' before 'predict'"):
            aa.ReliabilityModel().predict(np.zeros((2, 3)))

    def test_predict_none_raises(self, rm_ad):
        rm, _, _ = rm_ad
        with pytest.raises(ValueError, match="'X'"):
            rm.predict(None)

    def test_predict_with_nan_raises(self, rm_ad):
        rm, _, Xnew = rm_ad
        X_nan = Xnew.copy()
        X_nan[0, 0] = np.nan
        with pytest.raises(ValueError, match="'X'"):
            rm.predict(X_nan)

    def test_predict_with_inf_raises(self, rm_ad):
        rm, _, Xnew = rm_ad
        X_inf = Xnew.copy()
        X_inf[0, 0] = np.inf
        with pytest.raises(ValueError, match="'X'"):
            rm.predict(X_inf)


class TestPredictGoldenValues:
    """Hand-computed band edges, the ood_score identity, and the nearest-neighbour index."""

    def test_ood_score_identity(self, rm_ad):
        rm, _, Xnew = rm_ad
        df = rm.predict(Xnew)
        np.testing.assert_allclose(df["ood_score"], df["ad_knn"] / rm.ad_threshold_, rtol=0, atol=1e-9)

    @pytest.mark.parametrize("b", [0.0, 0.1, 0.25, 1.0])
    def test_status_matches_score_bands(self, rm_ad, b):
        _, _, Xnew = rm_ad
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(random_state=0, verbose=False).fit(Xtr, ytr, ad_borderline=b, n_bootstrap=3)
        df = rm.predict(Xnew)
        np.testing.assert_array_equal(df["ad_status"].to_numpy(),
                                      _status_from_score(df["ood_score"].to_numpy(), b))

    def test_nearest_train_matches_hand_neighbors(self, rm_ad):
        rm, Xtr, Xnew = rm_ad
        sc = StandardScaler().fit(Xtr)
        nn = NearestNeighbors(n_neighbors=5).fit(sc.transform(Xtr))
        expected = nn.kneighbors(sc.transform(Xnew))[1][:, 0]
        np.testing.assert_array_equal(rm.predict(Xnew)["ad_nearest_train"].to_numpy(), expected)

    def test_band_edges_backend(self):
        ood = np.array([0.5, 1.0, 1.0 + 1e-12, 1.1, 1.1 + 1e-12, 5.0, np.nan])
        status = comp_ad_status(ood, ood <= 1, borderline=0.1)
        assert status.tolist() == ["inside", "inside", "borderline", "borderline", "outside",
                                   "outside", "unknown"]

    def test_band_edges_zero_width(self):
        ood = np.array([1.0, 1.0 + 1e-12, np.inf, np.nan])
        status = comp_ad_status(ood, ood <= 1, borderline=0.0)
        assert status.tolist() == ["inside", "outside", "unknown", "unknown"]

    def test_hand_computed_diagonal_line(self):
        # Training points (i, i) for i in 0..7 (k=1): after standardization every nearest-neighbour
        # distance is sqrt(2)/std, so ad_threshold_ = sqrt(2)/std and ood_score equals the raw gap
        # to the nearest training point along the line.
        line = np.arange(8, dtype=float)
        Xtr = np.column_stack([line, line])
        ytr = np.array([0, 1] * 4)
        rm = aa.ReliabilityModel(random_state=0, verbose=False).fit(
            Xtr, ytr, k=1, ad_borderline=0.1, n_bootstrap=0, calibrate=False)
        assert rm.ad_threshold_ == pytest.approx(np.sqrt(2) / np.std(line), abs=1e-12)
        q = np.array([3.5, 8.05, 8.2, -1.05, 7.0])
        df = rm.predict(np.column_stack([q, q]))
        np.testing.assert_allclose(df["ood_score"], [0.5, 1.05, 1.2, 1.05, 0.0], atol=1e-9)
        assert df["ad_status"].tolist() == ["inside", "borderline", "outside", "borderline", "inside"]
        assert df["ad_nearest_train"].tolist()[1:] == [7, 7, 0, 7]
        assert df["ad_nearest_train"].iloc[0] in (3, 4)


class TestDistinctions:
    """Pin the mental model: score != trust; OOD overrides a high score."""

    def test_high_score_but_ood_is_not_reliable(self):
        # The key distinction: a confident-looking score on an OOD input must NOT be trusted.
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=5)
        # push a training positive far out of distribution -> stays "positive-looking" but OOD
        pos = Xtr[ytr == 1][0]
        ood = (pos + 25.0)[None, :]
        row = rm.predict(ood).iloc[0]
        assert not bool(row["in_domain"])
        assert not bool(row["reliable"])                     # untrustworthy regardless of score

    def test_confident_in_domain_can_be_reliable(self):
        Xtr, ytr, Xte = _data()
        df = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=5).predict(Xte)
        # at least some clearly-classifiable in-domain points come back reliable + in-domain
        assert (df["reliable"] & df["in_domain"]).any()
        assert df.loc[df["reliable"], "in_domain"].all()     # reliable implies in_domain

    def test_predict_is_idempotent_no_refit(self):
        # fit-once refactor: repeated predict calls are identical (no re-splitting/re-bootstrapping)
        Xtr, ytr, Xte = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=8)
        a, b = rm.predict(Xte), rm.predict(Xte)
        assert np.allclose(a["score"], b["score"]) and np.allclose(a["score_std"], b["score_std"])
        assert (a["conformal_set"].values == b["conformal_set"].values).all()

    def test_conformal_coverage_near_target(self):
        # empirical coverage of the conformal sets should be >= 1 - alpha (minus slack) on held-out
        X, y = make_classification(n_samples=300, n_features=8, n_informative=5, random_state=1)
        rm = aa.ReliabilityModel(random_state=1).fit(X[:200], y[:200], conformal_alpha=0.1)
        ev = rm.eval(X=X[200:], labels=y[200:])
        coverage = ev.iloc[-1]["empirical_pos"]
        assert coverage >= 0.8                               # target 0.9, allow finite-sample slack

    def test_score_is_centre_of_its_interval(self):
        # score is the member mean, so it never falls outside [ci_low, ci_high] (even with an
        # overfitting base model, which previously produced score-outside-CI rows)
        Xtr, ytr, Xte = _data(n=180)
        rm = aa.ReliabilityModel(random_state=3).fit(
            Xtr, ytr, model=GradientBoostingClassifier().fit(Xtr, ytr), n_bootstrap=25)
        df = rm.predict(Xte)
        assert (df["score"] >= df["ci_low"] - 1e-9).all()
        assert (df["score"] <= df["ci_high"] + 1e-9).all()

    def test_ad_distances_nan_when_features_exceed_samples(self):
        # p >= n makes Mahalanobis/leverage rank-deficient -> reported as NaN, not a constant;
        # the robust kNN-based in_domain still works
        X, y = make_classification(n_samples=40, n_features=30, n_informative=12, random_state=0)
        rm = aa.ReliabilityModel(random_state=0).fit(X[:14], y[:14], n_bootstrap=3)
        df = rm.predict(X[14:])
        assert df["ad_mahalanobis"].isna().all() and df["ad_leverage"].isna().all()
        assert df["in_domain"].notna().all()


# IV eval
class TestEval:
    def test_eval_default(self):
        Xtr, ytr, _ = _data()
        ev = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=5).eval()
        assert isinstance(ev, pd.DataFrame)
        assert list(ev.columns) == ["bin", "mean_score", "empirical_pos", "n_samples"]
        assert (ev["bin"] == "summary").any()

    def test_eval_columns_match_constant_bundle(self):
        Xtr, ytr, _ = _data()
        ev = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=3).eval()
        assert list(ev.columns) == ut.COLS_EVAL_RELIABILITY
        assert "n" not in ev.columns

    def test_eval_custom_bins(self):
        Xtr, ytr, _ = _data()
        ev = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=5).eval(n_bins=3)
        assert len(ev) == 3 + 1                              # bins + summary

    def test_eval_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="Call 'fit' before 'eval'"):
            aa.ReliabilityModel().eval()

    @pytest.mark.parametrize("nb", [1, 0, -2])
    def test_eval_n_bins_invalid(self, nb):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=3)
        with pytest.raises(ValueError, match="'n_bins'"):
            rm.eval(n_bins=nb)

    # X / labels: positive
    @settings(max_examples=5, deadline=None)
    @given(n=some.integers(min_value=20, max_value=40))
    def test_eval_X_subset(self, n):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=3)
        ev = rm.eval(X=Xtr[:n], labels=ytr[:n])
        assert ev["n_samples"].iloc[-1] == n
        assert int(ev["n_samples"].iloc[:-1].sum()) == n

    def test_eval_labels_drive_the_empirical_rate(self):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=3)
        ev = rm.eval(X=Xtr, labels=ytr)
        ev_flipped = rm.eval(X=Xtr, labels=1 - np.asarray(ytr))
        assert not ev["empirical_pos"].equals(ev_flipped["empirical_pos"])
        assert ev["mean_score"].equals(ev_flipped["mean_score"])      # scoring is label-free

    # X / labels: negative
    @pytest.mark.parametrize("X", ["abc", 5, [[1, 2], [3]]])
    def test_eval_X_invalid(self, X):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=3)
        with pytest.raises(ValueError, match="'X'"):
            rm.eval(X=X, labels=ytr)

    def test_eval_X_without_labels_raises(self):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=3)
        with pytest.raises(ValueError, match=r"'labels' \(None\) should be the evaluation"):
            rm.eval(X=Xtr)

    def test_eval_labels_without_X_uses_training_features(self):
        # Labels given without features are scored against the training matrix; they are not
        # replaced by the training labels (see test_eval_uses_labels_with_default_training_features).
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=3)
        ev = rm.eval(labels=ytr, n_bins=3)
        assert int(ev[ev["bin"] == "summary"]["n_samples"].iloc[0]) == len(Xtr)

    def test_eval_labels_without_X_length_mismatch_raises(self):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=3)
        with pytest.raises(ValueError, match="n_samples does not match"):
            rm.eval(labels=ytr[:-5])

    @pytest.mark.parametrize("labels", [[1] * 90, ["a", "b"] * 45, 5])
    def test_eval_labels_invalid(self, labels):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=3)
        with pytest.raises(ValueError, match="labels"):
            rm.eval(X=Xtr, labels=labels)

    def test_eval_labels_length_mismatch_raises(self):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=3)
        with pytest.raises(ValueError, match="n_samples does not match"):
            rm.eval(X=Xtr, labels=ytr[:-5])

    # use_calibrated: positive
    def test_use_calibrated_false_equals_default(self, rm_miscal):
        rm, Xte, yte = rm_miscal
        pd.testing.assert_frame_equal(rm.eval(X=Xte, labels=yte),
                                      rm.eval(X=Xte, labels=yte, use_calibrated=False))

    def test_use_calibrated_bins_calibrated_column(self, rm_miscal):
        rm, Xte, yte = rm_miscal
        ev = rm.eval(X=Xte, labels=yte, use_calibrated=True, n_bins=4)
        s_cal = rm.predict(Xte)["score_calibrated"].to_numpy()
        last = ev.iloc[3]                                    # closed [0.75, 1.00] bin
        m = s_cal >= 0.75
        assert last["n_samples"] == int(m.sum())
        if m.any():
            assert last["mean_score"] == pytest.approx(float(np.mean(s_cal[m])), abs=1e-12)

    @settings(max_examples=5, deadline=None)
    @given(n_bins=some.integers(min_value=2, max_value=12))
    def test_use_calibrated_shape(self, rm_miscal, n_bins):
        rm, Xte, yte = rm_miscal
        ev = rm.eval(X=Xte, labels=yte, n_bins=n_bins, use_calibrated=True)
        assert len(ev) == n_bins + 1
        assert int(ev["n_samples"].iloc[:n_bins].sum()) == len(Xte)

    def test_use_calibrated_keeps_summary_row(self, rm_miscal):
        rm, Xte, yte = rm_miscal
        raw = rm.eval(X=Xte, labels=yte).iloc[-1]
        cal = rm.eval(X=Xte, labels=yte, use_calibrated=True).iloc[-1]
        assert raw.equals(cal)

    def test_use_calibrated_changes_curve(self, rm_miscal):
        rm, Xte, yte = rm_miscal
        raw = rm.eval(X=Xte, labels=yte)
        cal = rm.eval(X=Xte, labels=yte, use_calibrated=True)
        assert not raw.equals(cal)

    @pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
    def test_use_calibrated_both_methods(self, method):
        Xtr, ytr, Xte, yte = _miscal_data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, model=GaussianNB(), n_bootstrap=0,
                                                    calibration_method=method)
        ev = rm.eval(X=Xte, labels=yte, use_calibrated=True, add_metrics=True)
        assert 0.0 <= _metric(ev, "brier") <= 1.0 and 0.0 <= _metric(ev, "ece") <= 1.0

    # use_calibrated: negative
    @pytest.mark.parametrize("val", [None, "yes", 1, 0, [True]])
    def test_use_calibrated_invalid_type(self, rm_miscal, val):
        rm, Xte, yte = rm_miscal
        with pytest.raises(ValueError, match="use_calibrated"):
            rm.eval(X=Xte, labels=yte, use_calibrated=val)

    def test_use_calibrated_without_calibrator_raises(self, rm_uncal):
        rm, Xte, yte = rm_uncal
        with pytest.raises(ValueError, match=r"'use_calibrated' \(True\) should"):
            rm.eval(X=Xte, labels=yte, use_calibrated=True)

    def test_use_calibrated_without_calibrator_on_training_data_raises(self, rm_uncal):
        rm, _, _ = rm_uncal
        with pytest.raises(ValueError, match=r"fitted with 'calibrate=False'"):
            rm.eval(use_calibrated=True)

    def test_use_calibrated_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="Call 'fit' before 'eval'"):
            aa.ReliabilityModel().eval(use_calibrated=True)

    # add_metrics: positive
    def test_add_metrics_false_has_no_metric_rows(self, rm_miscal):
        rm, Xte, yte = rm_miscal
        ev = rm.eval(X=Xte, labels=yte, add_metrics=False)
        assert not ev["bin"].isin(ut.LIST_BIN_METRICS).any()

    @settings(max_examples=5, deadline=None)
    @given(n_bins=some.integers(min_value=2, max_value=12))
    def test_add_metrics_rows(self, rm_miscal, n_bins):
        rm, Xte, yte = rm_miscal
        ev = rm.eval(X=Xte, labels=yte, n_bins=n_bins, add_metrics=True)
        assert len(ev) == n_bins + 3
        assert list(ev["bin"].iloc[-3:]) == [ut.STR_BIN_SUMMARY, ut.STR_BIN_BRIER, ut.STR_BIN_ECE]
        tail = ev.iloc[-2:]
        assert tail["empirical_pos"].isna().all()
        assert (tail["n_samples"] == len(Xte)).all()
        assert list(ev.columns) == ut.COLS_EVAL_RELIABILITY

    @settings(max_examples=5, deadline=None)
    @given(n_bins=some.integers(min_value=2, max_value=12))
    def test_add_metrics_ranges(self, rm_miscal, n_bins):
        rm, Xte, yte = rm_miscal
        for use_cal in (False, True):
            ev = rm.eval(X=Xte, labels=yte, n_bins=n_bins, add_metrics=True, use_calibrated=use_cal)
            assert 0.0 <= _metric(ev, "brier") <= 1.0
            assert 0.0 <= _metric(ev, "ece") <= 1.0

    def test_add_metrics_prefix_equals_default(self, rm_miscal):
        rm, Xte, yte = rm_miscal
        base = rm.eval(X=Xte, labels=yte)
        ext = rm.eval(X=Xte, labels=yte, add_metrics=True)
        pd.testing.assert_frame_equal(ext.iloc[:len(base)], base, check_exact=True)

    def test_add_metrics_without_calibrator(self, rm_uncal):
        rm, Xte, yte = rm_uncal
        ev = rm.eval(X=Xte, labels=yte, add_metrics=True)
        assert _metric(ev, "brier") > 0

    def test_add_metrics_deterministic(self, rm_miscal):
        rm, Xte, yte = rm_miscal
        a = rm.eval(X=Xte, labels=yte, add_metrics=True, use_calibrated=True)
        b = rm.eval(X=Xte, labels=yte, add_metrics=True, use_calibrated=True)
        pd.testing.assert_frame_equal(a, b, check_exact=True)

    # add_metrics: negative
    @pytest.mark.parametrize("val", [None, "no", 1, 0.0, {}])
    def test_add_metrics_invalid_type(self, rm_miscal, val):
        rm, Xte, yte = rm_miscal
        with pytest.raises(ValueError, match="add_metrics"):
            rm.eval(X=Xte, labels=yte, add_metrics=val)

    def test_add_metrics_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="Call 'fit' before 'eval'"):
            aa.ReliabilityModel().eval(add_metrics=True)

    def test_eval_uses_labels_with_default_training_features(self):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=3)
        # These valid but deliberately unbalanced labels must not be overwritten with ytr merely
        # because X defaults to the training feature matrix.
        labels_eval = np.array([1] * (len(ytr) - 10) + [0] * 10)
        ev = rm.eval(labels=labels_eval, n_bins=3)
        bins = ev[ev["bin"] != "summary"]
        bins = bins[bins["n_samples"] > 0]
        weighted_rate = np.average(bins["empirical_pos"], weights=bins["n_samples"])
        assert weighted_rate == pytest.approx(labels_eval.mean())

    def test_eval_unknown_labels_raise(self):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=3)
        with pytest.raises(ValueError, match="labels observed"):
            rm.eval(labels=np.array([2] * (len(ytr) - 1) + [0]))


class TestEvalComplex:
    """Acceptance criteria and cross-parameter interactions for calibrated evaluation."""

    @pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_calibration_strictly_lowers_brier_and_ece(self, method, seed):
        Xtr, ytr, Xte, yte = _miscal_data(seed=seed)
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, model=GaussianNB(), n_bootstrap=0,
                                                    calibration_method=method)
        raw = rm.eval(X=Xte, labels=yte, add_metrics=True)
        cal = rm.eval(X=Xte, labels=yte, add_metrics=True, use_calibrated=True)
        assert _metric(cal, "brier") < _metric(raw, "brier")
        assert _metric(cal, "ece") < _metric(raw, "ece")

    @pytest.mark.parametrize("n_bins", [2, 4, 5, 10])
    def test_default_output_byte_identical_to_legacy(self, n_bins):
        Xtr, ytr, Xte = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=5)
        yte = make_classification(n_samples=120, n_features=8, n_informative=5, n_redundant=1,
                                  random_state=0)[1][90:]
        for X, labels in [(Xte, yte), (Xtr, ytr)]:
            new = rm.eval(X=X, labels=labels, n_bins=n_bins)
            ref = _legacy_eval(rm, X, labels, n_bins=n_bins)
            pd.testing.assert_frame_equal(new, ref, check_exact=True)
            assert new.to_csv(float_format="%.17g") == ref.to_csv(float_format="%.17g")
            assert [type(v) for v in new.iloc[0]] == [type(v) for v in ref.iloc[0]]

    def test_default_on_training_data_byte_identical(self):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=3)
        pd.testing.assert_frame_equal(rm.eval(), _legacy_eval(rm, Xtr, ytr), check_exact=True)

    def test_eval_unchanged_by_new_columns(self, rm_ad):
        rm, _, _ = rm_ad
        ev = rm.eval()
        assert list(ev.columns) == ut.COLS_EVAL_RELIABILITY

    def test_calibrated_metrics_with_custom_bins(self, rm_miscal):
        rm, Xte, yte = rm_miscal
        e5 = rm.eval(X=Xte, labels=yte, n_bins=5, add_metrics=True, use_calibrated=True)
        e10 = rm.eval(X=Xte, labels=yte, n_bins=10, add_metrics=True, use_calibrated=True)
        assert _metric(e5, "brier") == _metric(e10, "brier")         # Brier is bin-free
        assert len(e10) == 13 and len(e5) == 8

    def test_use_calibrated_on_training_data(self, rm_miscal):
        rm, _, _ = rm_miscal
        ev = rm.eval(use_calibrated=True, add_metrics=True)
        assert ev["n_samples"].iloc[-1] == 400

    def test_uncalibrated_raises_even_with_metrics(self, rm_uncal):
        rm, Xte, yte = rm_uncal
        with pytest.raises(ValueError, match="use_calibrated"):
            rm.eval(X=Xte, labels=yte, use_calibrated=True, add_metrics=True, n_bins=3)

    @pytest.mark.parametrize("nb", [1, 0, -1])
    def test_invalid_n_bins_with_metrics_raises(self, rm_miscal, nb):
        rm, Xte, yte = rm_miscal
        with pytest.raises(ValueError, match="'n_bins'"):
            rm.eval(X=Xte, labels=yte, n_bins=nb, add_metrics=True, use_calibrated=True)

    def test_labels_mismatch_with_calibrated_raises(self, rm_miscal):
        rm, Xte, yte = rm_miscal
        with pytest.raises(ValueError, match="n_samples does not match"):
            rm.eval(X=Xte, labels=yte[:-5], use_calibrated=True, add_metrics=True)

    def test_feature_mismatch_with_calibrated_raises(self, rm_miscal):
        rm, Xte, yte = rm_miscal
        with pytest.raises(ValueError, match="features but the model was fit on"):
            rm.eval(X=Xte[:, :5], labels=yte, use_calibrated=True)

    def test_bad_flag_combination_types_raise(self, rm_miscal):
        rm, Xte, yte = rm_miscal
        with pytest.raises(ValueError, match="'use_calibrated'"):
            rm.eval(X=Xte, labels=yte, use_calibrated="True", add_metrics="True")

    # Unavailable calibrator: the eval-time message names the real reason
    def test_failed_calibration_eval_names_the_real_reason(self):
        X, y = _failed_calibration_data()
        with pytest.warns(UserWarning):
            rm = aa.ReliabilityModel(random_state=0).fit(X, y, n_bootstrap=0)
        with pytest.raises(ValueError, match="the calibrator could not be fitted") as e:
            rm.eval(use_calibrated=True)
        msg = str(e.value)
        assert msg.startswith("'use_calibrated' (True) should be False for this model")
        assert "'calibrate=False'" not in msg          # never blame a flag the user did not pass

    def test_calibrate_false_eval_names_that_flag(self, rm_uncal):
        rm, Xte, yte = rm_uncal
        with pytest.raises(ValueError, match=r"fitted with 'calibrate=False'") as e:
            rm.eval(X=Xte, labels=yte, use_calibrated=True)
        assert "could not be fitted" not in str(e.value)


class TestEvalGoldenValues:
    """Hand-computed Brier / ECE values and the sklearn reference."""

    @pytest.mark.parametrize("use_cal", [False, True])
    def test_brier_matches_sklearn(self, rm_miscal, use_cal):
        rm, Xte, yte = rm_miscal
        col = "score_calibrated" if use_cal else "score"
        s = rm.predict(Xte)[col].to_numpy()
        ev = rm.eval(X=Xte, labels=yte, add_metrics=True, use_calibrated=use_cal)
        assert abs(_metric(ev, "brier") - brier_score_loss(yte, s)) <= 1e-12

    def test_brier_matches_sklearn_default_model(self):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=5)
        ev = rm.eval(add_metrics=True, use_calibrated=True)
        s = rm.predict(Xtr)["score_calibrated"].to_numpy()
        assert abs(_metric(ev, "brier") - brier_score_loss(ytr, s)) <= 1e-12

    def test_ece_recomputed_from_bins(self, rm_miscal):
        rm, Xte, yte = rm_miscal
        ev = rm.eval(X=Xte, labels=yte, n_bins=5, add_metrics=True, use_calibrated=True)
        bins = ev.iloc[:5]
        bins = bins[bins["n_samples"] > 0]
        expected = float((bins["n_samples"] / len(Xte)
                          * (bins["mean_score"] - bins["empirical_pos"]).abs()).sum())
        assert _metric(ev, "ece") == pytest.approx(expected, abs=1e-12)

    def test_hand_computed_brier_and_ece(self):
        s = np.array([0.1, 0.2, 0.9, 0.8])
        y = np.array([0, 1, 1, 1])
        rows = comp_calibration_bins(s, y, n_bins=2)
        # bin [0, .5): mean .15 vs rate .5 -> .35 * 2/4; bin [.5, 1]: mean .85 vs rate 1 -> .15 * 2/4
        assert comp_ece(rows, n_samples=4) == pytest.approx(0.25, abs=1e-12)
        assert comp_brier(s, y) == pytest.approx((0.01 + 0.64 + 0.01 + 0.04) / 4, abs=1e-12)

    def test_perfectly_calibrated_bins_give_zero_ece(self):
        s = np.array([0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75])
        y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
        assert comp_ece(comp_calibration_bins(s, y, n_bins=2), n_samples=8) == pytest.approx(0.0, abs=1e-12)

    def test_empty_bins_ignored_in_ece(self):
        s = np.array([0.95, 0.95])
        y = np.array([1, 1])
        rows = comp_calibration_bins(s, y, n_bins=10)
        assert sum(r[3] for r in rows) == 2
        assert comp_ece(rows, n_samples=2) == pytest.approx(0.05, abs=1e-12)


class TestReliabilityModelGoldenValues:
    """Hand-checkable numbers for the renamed / re-unit-ed outputs."""

    def test_ci_fraction_gives_wald_z_interval(self):
        # ci=0.90 -> z = norm.ppf(0.95); unclipped rows satisfy ci_high - score == z * score_std.
        Xtr, ytr, Xte = _data()
        df = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=10, ci=0.90).predict(Xte)
        z = norm.ppf(0.95)
        inner = (df["ci_high"] < 1.0) & (df["ci_low"] > 0.0)
        assert inner.any()
        d = df[inner]
        np.testing.assert_allclose(d["ci_high"] - d["score"], z * d["score_std"], atol=1e-12)
        np.testing.assert_allclose(d["score"] - d["ci_low"], z * d["score_std"], atol=1e-12)

    def test_eval_n_samples_counts(self):
        Xtr, ytr, _ = _data()
        ev = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=3).eval(n_bins=4)
        bins = ev[ev["bin"] != "summary"]
        summary = ev[ev["bin"] == "summary"].iloc[0]
        assert summary["n_samples"] == len(Xtr) == 90
        assert int(bins["n_samples"].sum()) == 90
        assert list(bins["bin"]) == ["0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"]

    def test_ad_knn_column_name(self):
        Xtr, ytr, Xte = _data()
        df = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=3).predict(Xte)
        assert ut.COL_AD_KNN == "ad_knn" and "ad_knn" in df.columns
        assert "ad_knn_dist" not in df.columns
        assert (df["ad_knn"] >= 0).all()
