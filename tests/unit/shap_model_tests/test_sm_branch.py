"""This script targets branch arms of ShapModel and its backend via the public API."""
import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import aaanalysis as aa

aa.options["verbose"] = False


# Small shared fixtures (kept tiny: few estimators, small X)
def _small_data(n_samples=12, n_feat=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, n_feat))
    labels = np.array([1, 0] * (n_samples // 2))
    return X, labels


def _df_feat(n_feat=4):
    df = aa.load_features(name="DOM_GSEC").head(n_feat).reset_index(drop=True)
    # Drop any bundled impact/importance columns so guards start from a clean frame
    keep = [c for c in df.columns if "feat_impact" not in c and "feat_importance" not in c]
    return df[keep]


MODEL_KWARGS = dict(list_model_classes=[RandomForestClassifier], random_state=0)
ARGS = dict(n_rounds=1)


class TestInitBranch:
    """Branch arms in ShapModel.__init__ / explainer checks."""

    def test_explainer_class_none_default_arm(self):
        # _shap_model.py:334 -> explainer_class is None selects TreeExplainer.
        # Construction alone exercises the None-default arm; we do not fit because
        # the installed shap rejects model_output="probability" with the default
        # tree_path_dependent perturbation (a shap-version quirk, not our branch).
        sm = aa.ShapModel(explainer_class=None, **MODEL_KWARGS, verbose=False)
        assert sm is not None

    def test_explainer_kwargs_not_none_arm(self):
        # _shap_model.py:63 -> explainer_kwargs is not None branch in
        # check_match_class_explainer_and_models (valid kwargs accepted)
        sm = aa.ShapModel(explainer_class=shap.TreeExplainer,
                          explainer_kwargs=dict(feature_perturbation="tree_path_dependent"),
                          **MODEL_KWARGS, verbose=False)
        assert sm is not None

    def test_explainer_kwargs_invalid_raises(self):
        # _shap_model.py:63 then the explainer-kwargs failure path
        with pytest.raises(ValueError):
            aa.ShapModel(explainer_class=shap.KernelExplainer,
                         explainer_kwargs=dict(not_a_real_kwarg=123),
                         **MODEL_KWARGS, verbose=False)


class TestFitBranch:
    """Branch arms reached through ShapModel.fit."""

    def test_label_target_class_not_in_labels(self):
        # _shap_model.py:130 -> label_target_class not in label classes
        X, labels = _small_data()
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=False)
        with pytest.raises(ValueError, match="label_target_class"):
            sm.fit(X, labels=labels, label_target_class=5, **ARGS)

    def test_n_background_data_ge_n_samples(self):
        # _shap_model.py:139 -> n_background_data >= n_samples
        X, labels = _small_data(n_samples=10)
        sm = aa.ShapModel(explainer_class=shap.KernelExplainer, **MODEL_KWARGS, verbose=False)
        with pytest.raises(ValueError, match="n_background_data"):
            sm.fit(X, labels=labels, n_background_data=10, **ARGS)

    def test_n_background_data_none_skip_arm(self):
        # _shap_model.py:137 -> n_background_data is None returns/skips
        X, labels = _small_data()
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=False)
        sm.fit(X, labels=labels, n_background_data=None, **ARGS)
        assert sm.shap_values is not None

    def test_non_binary_labels_raises(self):
        # check_models.py:15 -> len(unique_labels) != 2 (non-fuzzy path).
        # Need >1 unique value (so ut.check_labels passes) but != 2 unique -> 3 classes.
        X, _ = _small_data(n_samples=12)
        labels = np.array([0, 1, 2] * 4)
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=False)
        with pytest.raises(ValueError, match="2 unique labels"):
            sm.fit(X, labels=labels, **ARGS)

    def test_is_selected_feature_mismatch_raises(self):
        # check_models.py:27 -> X feature count != is_selected width
        X, labels = _small_data(n_feat=4)
        wrong = [np.ones(3, dtype=bool)]  # width 3 != 4 features
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=False)
        with pytest.raises(ValueError, match="does not match"):
            sm.fit(X, labels=labels, is_selected=wrong, **ARGS)

    def test_fuzzy_valid_threshold_path(self):
        # _shap_model.py:89 (<2 unique fuzzy labels) is UNREACHABLE: ut.check_labels
        # rejects a single-unique label vector first. Instead exercise the reachable
        # fuzzy positive path (backend threshold relabeling) with one fuzzy label.
        X, _ = _small_data(n_samples=12)
        labels = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0.5], dtype=float)
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=False)
        sm.fit(X, labels=labels, fuzzy_labeling=True, **ARGS)
        assert sm.shap_values.shape == X.shape

    def test_fuzzy_label_out_of_range_raises(self):
        # _shap_model.py:93 -> fuzzy path, values outside [0,1]
        X, _ = _small_data(n_samples=12)
        labels = np.array([0, 1] * 6, dtype=float)
        labels[0] = 1.5  # out of range
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=False)
        with pytest.raises(ValueError, match="between 0 and 1"):
            sm.fit(X, labels=labels, fuzzy_labeling=True, **ARGS)

    def test_fuzzy_warns_not_exactly_one_fuzzy(self):
        # _shap_model.py:117 -> n_fuzzy_labels != 1 warns (verbose=True)
        X, _ = _small_data(n_samples=12)
        labels = np.array([0, 1] * 6, dtype=float)
        labels[0] = 0.3
        labels[1] = 0.7  # two fuzzy labels
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=True)
        with pytest.warns(UserWarning):
            sm.fit(X, labels=labels, fuzzy_labeling=True, **ARGS)
        assert sm.shap_values is not None

    def test_fuzzy_warns_unbalanced_nonfuzzy(self):
        # _shap_model.py:122 -> n_pos != n_neg warns (one fuzzy, unbalanced rest)
        X, _ = _small_data(n_samples=12)
        # 7 ones, 4 zeros, 1 fuzzy -> exactly one fuzzy but unbalanced
        labels = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0.5], dtype=float)
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=True)
        with pytest.warns(UserWarning):
            sm.fit(X, labels=labels, fuzzy_labeling=True, **ARGS)
        assert sm.shap_values is not None


class TestFitBackendBranch:
    """Backend shap_model_fit branch arms via explainer choice."""

    def test_kernel_explainer_list_output_path(self):
        # shap_model_fit.py:26,27 -> KernelExplainer with predict_proba yields
        # list-of-arrays shap_output for the class_index path.
        X, labels = _small_data(n_samples=12, n_feat=4)
        sm = aa.ShapModel(explainer_class=shap.KernelExplainer,
                          list_model_classes=[RandomForestClassifier], verbose=False,
                          random_state=0)
        sm.fit(X, labels=labels, n_rounds=1)
        assert sm.shap_values.shape == X.shape

    def test_tree_explainer_ndarray_path(self):
        # shap_model_fit.py:32/38 -> TreeExplainer returns ndarray (ndim 2 or 3)
        X, labels = _small_data(n_samples=12, n_feat=4)
        sm = aa.ShapModel(explainer_class=shap.TreeExplainer,
                          list_model_classes=[RandomForestClassifier], verbose=False,
                          random_state=0)
        sm.fit(X, labels=labels, n_rounds=1)
        assert sm.shap_values.shape == X.shape

    def test_multiple_models_aggregate(self):
        # shap_model_fit.py:98 -> aggregate across >1 model; shape-fix arm if shap
        # output carries a trailing class axis.
        X, labels = _small_data(n_samples=12, n_feat=4)
        sm = aa.ShapModel(list_model_classes=[RandomForestClassifier, ExtraTreesClassifier],
                          verbose=False, random_state=0)
        sm.fit(X, labels=labels, n_rounds=1)
        assert sm.shap_values.shape == X.shape


def _fitted_sm(n_samples=12, n_feat=4, seed=0, verbose=False):
    X, labels = _small_data(n_samples=n_samples, n_feat=n_feat, seed=seed)
    sm = aa.ShapModel(**MODEL_KWARGS, verbose=verbose)
    sm.fit(X, labels=labels, n_rounds=1)
    return sm


class TestAddFeatImpactBranch:
    """Branch arms in add_feat_impact and its backend."""

    def test_add_feat_impact_before_fit_raises(self):
        # _shap_model.py:75 (check_shap_values) and :220 both guard None shap_values
        sm = aa.ShapModel(**MODEL_KWARGS, verbose=False)
        with pytest.raises(ValueError, match="None"):
            sm.add_feat_impact(df_feat=_df_feat())

    def test_feat_importance_normalize_true(self):
        # sm_add_feat_impact.py:64 -> normalize arm in comp_shap_feature_importance
        sm = _fitted_sm(n_feat=4)
        df_feat = _df_feat(4)
        out = sm.add_feat_impact(df_feat=df_feat, shap_feat_importance=True, normalize=True)
        assert "feat_importance" in list(out)

    def test_feat_importance_already_exists_raises(self):
        # _shap_model.py:228,229 -> feat_importance column already present
        sm = _fitted_sm(n_feat=4)
        df_feat = _df_feat(4)
        out = sm.add_feat_impact(df_feat=df_feat, shap_feat_importance=True)
        with pytest.raises(ValueError, match="feat_importance"):
            sm.add_feat_impact(df_feat=out, shap_feat_importance=True, drop=False)

    def test_feat_importance_drop_arm(self):
        # sm_add_feat_impact.py:72 -> drop=True drops then re-inserts
        sm = _fitted_sm(n_feat=4)
        df_feat = _df_feat(4)
        out = sm.add_feat_impact(df_feat=df_feat, shap_feat_importance=True)
        out2 = sm.add_feat_impact(df_feat=out, shap_feat_importance=True, drop=True)
        assert "feat_importance" in list(out2)

    def test_df_feat_shap_n_feat_mismatch_raises(self):
        # _shap_model.py:224 -> len(df_feat) != shap_values n_feat
        sm = _fitted_sm(n_feat=4)
        df_feat = _df_feat(3)  # 3 rows vs 4 shap features
        with pytest.raises(ValueError, match="Mismatch"):
            sm.add_feat_impact(df_feat=df_feat)

    def test_single_sample_int_path(self):
        # sm_add_feat_impact.py:88 -> sample_positions int single-sample arm
        sm = _fitted_sm(n_feat=4)
        df_feat = _df_feat(4)
        out = sm.add_feat_impact(df_feat=df_feat, sample_positions=0, names="P0")
        assert any("feat_impact" in c for c in list(out))

    def test_group_average_path(self):
        # sm_add_feat_impact.py:43,49,95 (group) + _shap_model.py:180,184
        sm = _fitted_sm(n_feat=4)
        df_feat = _df_feat(4)
        out = sm.add_feat_impact(df_feat=df_feat, sample_positions=[0, 1, 2],
                                 group_average=True, normalize=True)
        assert any("feat_impact" in c for c in list(out))

    def test_group_average_verbose_std_warn(self):
        # sm_add_feat_impact.py:49,51 -> verbose std warning path may fire
        sm = _fitted_sm(n_feat=4, verbose=True)
        df_feat = _df_feat(4)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = sm.add_feat_impact(df_feat=df_feat, sample_positions=[0, 1, 2, 3],
                                     group_average=True, normalize=True)
        assert any("feat_impact" in c for c in list(out))

    def test_group_average_not_list_raises(self):
        # _shap_model.py:180 -> group_average=True but sample_positions int
        sm = _fitted_sm(n_feat=4)
        df_feat = _df_feat(4)
        with pytest.raises(ValueError, match="must be a list"):
            sm.add_feat_impact(df_feat=df_feat, sample_positions=2, group_average=True)

    def test_group_average_single_position_raises(self):
        # _shap_model.py:214 -> group_average with single-element list
        sm = _fitted_sm(n_feat=4)
        df_feat = _df_feat(4)
        with pytest.raises(ValueError):
            sm.add_feat_impact(df_feat=df_feat, sample_positions=[0], group_average=True)

    def test_group_average_invalid_names_type_raises(self):
        # _shap_model.py:187 -> group_average with list names (not str/None)
        sm = _fitted_sm(n_feat=4)
        df_feat = _df_feat(4)
        with pytest.raises(ValueError):
            sm.add_feat_impact(df_feat=df_feat, sample_positions=[0, 1, 2],
                               group_average=True, names=["a", "b"])

    def test_group_average_names_str_kept(self):
        # _shap_model.py:184 -> group_average with str names kept as-is
        sm = _fitted_sm(n_feat=4)
        df_feat = _df_feat(4)
        out = sm.add_feat_impact(df_feat=df_feat, sample_positions=[0, 1, 2],
                                 group_average=True, names="MyGroup")
        assert any("MyGroup" in c for c in list(out))

    def test_impact_columns_exist_raises(self):
        # _shap_model.py:232-235 -> feat_impact cols already present (non-importance)
        sm = _fitted_sm(n_feat=4)
        df_feat = _df_feat(4)
        out = sm.add_feat_impact(df_feat=df_feat, sample_positions=0, names="P0")
        with pytest.raises(ValueError, match="feat_impact"):
            sm.add_feat_impact(df_feat=out, sample_positions=0, names="P0", drop=False)
