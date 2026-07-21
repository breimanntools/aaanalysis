"""Unit tests for ReliabilityModel (prediction-reliability measures)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import aaanalysis as aa


def _data(n=120, n_features=8, seed=0):
    X, y = make_classification(n_samples=n, n_features=n_features, n_informative=5,
                               n_redundant=1, random_state=seed)
    return X[:90], y[:90], X[90:]


def _ood_point(X_train):
    return (X_train.mean(axis=0) + 20.0)[None, :]


# I __init__
class TestReliabilityModelInit:
    def test_returns_instance(self):
        assert isinstance(aa.ReliabilityModel(), aa.ReliabilityModel)

    def test_random_state_valid(self):
        assert isinstance(aa.ReliabilityModel(random_state=42), aa.ReliabilityModel)

    @pytest.mark.parametrize("rs", [-1, 1.5, "x"])
    def test_random_state_invalid(self, rs):
        with pytest.raises(ValueError):
            aa.ReliabilityModel(random_state=rs)

    @pytest.mark.parametrize("v", [None, "yes", 3])
    def test_verbose_invalid(self, v):
        with pytest.raises(ValueError):
            aa.ReliabilityModel(verbose=v)


# II fit
class TestFit:
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

    def test_label_pos_absent_raises(self):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError):
            aa.ReliabilityModel().fit(Xtr, ytr, label_pos=7)

    def test_X_labels_mismatch_raises(self):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError):
            aa.ReliabilityModel().fit(Xtr, ytr[:-3])

    def test_empty_model_list_raises(self):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError):
            aa.ReliabilityModel().fit(Xtr, ytr, model=[])

    def test_non_binary_labels_raises(self):
        X, y = make_classification(n_samples=90, n_features=8, n_informative=5, n_classes=3,
                                   n_clusters_per_class=1, random_state=0)
        with pytest.raises(ValueError):
            aa.ReliabilityModel().fit(X, y)

    def test_model_without_predict_proba_raises(self):
        from sklearn.svm import SVC
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError):                      # SVC() has no predict_proba by default
            aa.ReliabilityModel().fit(Xtr, ytr, model=SVC().fit(Xtr, ytr))

    def test_unfitted_aapred_rejected(self):
        Xtr, ytr, _ = _data()

        class _Pred:
            list_models_ = None                              # unfitted AAPred
        with pytest.raises(ValueError):
            aa.ReliabilityModel().fit(Xtr, ytr, model=_Pred())

    @pytest.mark.parametrize("k", [1, 3, 10])
    def test_k_valid(self, k):
        Xtr, ytr, _ = _data()
        assert aa.ReliabilityModel().fit(Xtr, ytr, k=k, n_bootstrap=3) is not None

    @pytest.mark.parametrize("k", [0, -1, 2.5])
    def test_k_invalid(self, k):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError):
            aa.ReliabilityModel().fit(Xtr, ytr, k=k)

    @pytest.mark.parametrize("p", [1, 50, 100])
    def test_ad_percentile_valid(self, p):
        Xtr, ytr, _ = _data()
        assert aa.ReliabilityModel().fit(Xtr, ytr, ad_percentile=p, n_bootstrap=3) is not None

    @pytest.mark.parametrize("p", [0, 101, -5])
    def test_ad_percentile_invalid(self, p):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError):
            aa.ReliabilityModel().fit(Xtr, ytr, ad_percentile=p)

    @pytest.mark.parametrize("ci", [50, 90, 99])
    def test_ci_valid(self, ci):
        Xtr, ytr, _ = _data()
        assert aa.ReliabilityModel().fit(Xtr, ytr, ci=ci, n_bootstrap=3) is not None

    @pytest.mark.parametrize("ci", [0, 100, 120])
    def test_ci_invalid(self, ci):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError):
            aa.ReliabilityModel().fit(Xtr, ytr, ci=ci)

    @pytest.mark.parametrize("nb", [-1, 2.5])
    def test_n_bootstrap_invalid(self, nb):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError):
            aa.ReliabilityModel().fit(Xtr, ytr, n_bootstrap=nb)

    def test_calibration_method_invalid(self):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError):
            aa.ReliabilityModel().fit(Xtr, ytr, calibration_method="bogus")

    @pytest.mark.parametrize("m", ["isotonic", "sigmoid"])
    def test_calibration_method_valid(self, m):
        Xtr, ytr, _ = _data()
        assert aa.ReliabilityModel().fit(Xtr, ytr, calibration_method=m, n_bootstrap=3) is not None

    @pytest.mark.parametrize("calibrate", [True, False])
    def test_calibrate_flag(self, calibrate):
        Xtr, ytr, Xte = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, calibrate=calibrate, n_bootstrap=3)
        df = rm.predict(Xte)
        assert df["score_calibrated"].notna().any() if calibrate else df["score_calibrated"].isna().all()

    @pytest.mark.parametrize("a", [-0.1, 1.5])
    def test_conformal_alpha_invalid(self, a):
        Xtr, ytr, _ = _data()
        with pytest.raises(ValueError):
            aa.ReliabilityModel().fit(Xtr, ytr, conformal_alpha=a)


# III predict
class TestPredict:
    def test_columns(self):
        Xtr, ytr, Xte = _data()
        df = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=5).predict(Xte)
        assert isinstance(df, pd.DataFrame) and len(df) == len(Xte)
        for c in ["score", "score_std", "ci_low", "ci_high", "ood_score", "in_domain",
                  "ad_knn_dist", "ad_mahalanobis", "ad_leverage", "score_calibrated",
                  "margin", "entropy", "conformal_set", "reliable"]:
            assert c in df.columns
        assert df["in_domain"].dtype == bool and df["reliable"].dtype == bool

    def test_before_fit_raises(self):
        _, _, Xte = _data()
        with pytest.raises(RuntimeError):
            aa.ReliabilityModel().predict(Xte)

    def test_feature_mismatch_raises(self):
        Xtr, ytr, Xte = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=3)
        with pytest.raises(ValueError):
            rm.predict(Xte[:, :4])

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
        from sklearn.ensemble import GradientBoostingClassifier
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
        assert {"bin", "mean_score", "empirical_pos", "n"}.issubset(ev.columns)
        assert (ev["bin"] == "summary").any()

    def test_eval_custom_bins(self):
        Xtr, ytr, _ = _data()
        ev = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=5).eval(n_bins=3)
        assert len(ev) == 3 + 1                              # bins + summary

    def test_eval_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            aa.ReliabilityModel().eval()

    @pytest.mark.parametrize("nb", [1, 0, -2])
    def test_eval_n_bins_invalid(self, nb):
        Xtr, ytr, _ = _data()
        rm = aa.ReliabilityModel(random_state=0).fit(Xtr, ytr, n_bootstrap=3)
        with pytest.raises(ValueError):
            rm.eval(n_bins=nb)
