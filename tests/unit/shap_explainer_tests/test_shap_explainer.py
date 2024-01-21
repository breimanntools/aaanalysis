"""This script tests the ShapExplainer class initialization and its parameters."""
import pytest
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
import shap

import aaanalysis as aa
aa.options["verbose"] = "off"

ARGS = dict(explainer_class=shap.TreeExplainer)

class TestShapExplainer:
    """Test ShapExplainer class individual parameters."""

    # Positive tests
    def test_explainer_class_parameter(self):
        se = aa.ShapExplainer(explainer_class=shap.TreeExplainer)
        assert se._explainer_class == shap.TreeExplainer
        se = aa.ShapExplainer(explainer_class=shap.LinearExplainer, list_model_classes=[LogisticRegression, LinearRegression])
        assert se._explainer_class == shap.LinearExplainer
        se = aa.ShapExplainer(explainer_class=shap.KernelExplainer,
                              list_model_classes=[LogisticRegression, LinearRegression, SVR, SVR])
        assert se._explainer_class == shap.KernelExplainer

    def test_explainer_kwargs_parameter(self):
        list_explainer_kwargs = [dict(feature_perturbation="interventional", model_output="probability"),
                                 dict(feature_perturbation="interventional", model_output="raw")]
        for explainer_kwargs in list_explainer_kwargs:
            se = aa.ShapExplainer(explainer_kwargs=explainer_kwargs)
            assert se._explainer_kwargs == explainer_kwargs

    def test_list_model_classes_parameter(self):
        list_model_classes = [RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier]
        for i in range(1, len(list_model_classes)):
            se = aa.ShapExplainer(list_model_classes=list_model_classes[:i])
            assert se._list_model_classes == list_model_classes[:i]

    def test_list_model_kwargs_parameter(self):
        list_model_kwargs = [{"n_estimators": 10, "max_depth": 5},
                             {"n_estimators": 50, "max_depth": 10},
                             {"n_estimators": 100, "max_depth": None},
                             {"n_estimators": 150, "min_samples_split": 2},
                             {"n_estimators": 200, "min_samples_leaf": 1}]
        for model_kwargs in list_model_kwargs:
            se = aa.ShapExplainer(list_model_kwargs=[model_kwargs], list_model_classes=[RandomForestClassifier])
            assert se._list_model_kwargs == [model_kwargs]

    def test_verbose_parameter(self):
        aa.options["verbose"] = "off"
        for verbose in [True, False]:
            se = aa.ShapExplainer(verbose=verbose)
            assert se._verbose == verbose

    def test_random_state_parameter(self):
        """Test the 'random_state' parameter."""
        for random_state in [None, 0, 1, 5, 42]:
            explainer = aa.ShapExplainer(random_state=random_state)
            assert explainer._random_state == random_state


    # Negative tests for explainer_class
    def test_invalid_explainer_class(self):
        """Test invalid type for 'explainer_class' parameter."""
        with pytest.raises(ValueError):
            aa.ShapExplainer(explainer_class="Not a valid explainer class")
        with pytest.raises(ValueError):
            aa.ShapExplainer(explainer_class=shap.Explainer)
        with pytest.raises(ValueError):
            aa.ShapExplainer(explainer_class=RandomForestClassifier)
        with pytest.raises(ValueError):
            aa.ShapExplainer(explainer_class="srt")
        with pytest.raises(ValueError):
            aa.ShapExplainer(explainer_class=[shap.TreeExplainer])


    def test_invalid_explainer_kwargs(self):

        for explainer_kwargs in [123, "asdf", [], dict(tree=1)]:
            with pytest.raises(ValueError):
                aa.ShapExplainer(explainer_kwargs=explainer_kwargs)
        list_explainer_kwargs = [dict(models=RandomForestClassifier),
                                 dict(feature_perturbation="interventional", model_output="invalid_probability"),
                                 dict(model=RandomForestClassifier, feature_perturbation_invalid="interventional")]
        for explainer_kwargs in list_explainer_kwargs:
            with pytest.raises(ValueError):
                aa.ShapExplainer(explainer_kwargs=explainer_kwargs)


    def test_invalid_list_model_classes(self):
        with pytest.raises(ValueError):
            aa.ShapExplainer(list_model_classes=[])
        with pytest.raises(ValueError):
            aa.ShapExplainer(list_model_classes=[123])
        with pytest.raises(ValueError):
            aa.ShapExplainer(list_model_classes="str")
        with pytest.raises(ValueError):
            aa.ShapExplainer(list_model_classes=[SVC, LinearRegression])
        with pytest.raises(ValueError):
            aa.ShapExplainer(list_model_classes=[RandomForestClassifier, 123])


    def test_invalid_list_model_kwargs(self):
        with pytest.raises(ValueError):
            aa.ShapExplainer(list_model_kwargs=[])
        with pytest.raises(ValueError):
            aa.ShapExplainer(list_model_kwargs=[234])
        with pytest.raises(ValueError):
            aa.ShapExplainer(list_model_kwargs=123)
        with pytest.raises(ValueError):
            aa.ShapExplainer(list_model_kwargs="invalid")
        # 2 instead of 3
        with pytest.raises(ValueError):
            aa.ShapExplainer(list_model_kwargs=[{"n_estimators": 10, "max_depth": 5},
                                                {"n_estimators": 10, "max_depth": 5}])
        # Wrong parameter
        with pytest.raises(ValueError):
            aa.ShapExplainer(
                list_model_kwargs=[{"n_estimators": 10, "max_depth": 5},
                                   {"n_estimators": 10, "max_depth": 5},
                                   {"N_estimators": 10, "max_depth": 5}])

    def test_invalid_verbose(self):
        with pytest.raises(ValueError):
            aa.ShapExplainer(verbose=1)
        with pytest.raises(ValueError):
            aa.ShapExplainer(verbose=None)
        with pytest.raises(ValueError):
            aa.ShapExplainer(verbose=[True, False])

    def test_invalid_random_state(self):
        aa.options["random_state"] = "off"
        for random_state in [-1, "Invalid", [1], dict()]:
            with pytest.raises(ValueError):
                aa.ShapExplainer(random_state=random_state)


class TestComplexShapExplainer:
    """Complex test class"""

    # Positive test
    def test_valid_combination_of_parameters(self):
        explainer_kwargs = dict(feature_perturbation="interventional", model_output="probability")
        list_model_classes = [RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier]
        list_model_kwargs = [{"n_estimators": 10, "max_depth": 5},
                             {"n_estimators": 50, "max_depth": 10},
                             {"n_estimators": 100, "max_depth": None}]

        se = aa.ShapExplainer(explainer_class=shap.TreeExplainer,
                              explainer_kwargs=explainer_kwargs,
                              list_model_classes=list_model_classes,
                              list_model_kwargs=list_model_kwargs,
                              verbose=True,
                              random_state=42)

        assert se._explainer_class == shap.TreeExplainer
        assert se._explainer_kwargs == explainer_kwargs
        assert se._list_model_classes == list_model_classes
        assert se._list_model_kwargs == list_model_kwargs
        assert se._verbose is True
        assert se._random_state == 42


    def test_invalid_combination_of_parameters(self):
        """Test an invalid combination of various parameters."""
        explainer_kwargs = dict(feature_perturbation="interventional", model_output="invalid_output")
        list_model_classes = [RandomForestClassifier, SVC]  # SVC is not compatible here
        list_model_kwargs = [{"n_estimators": 10, "max_depth": 5},
                             {"C": 1.0, "kernel": "linear"}]  # Incorrect kwargs for RandomForestClassifier

        with pytest.raises(ValueError):
            aa.ShapExplainer(explainer_class=shap.TreeExplainer,
                             explainer_kwargs=explainer_kwargs,
                             list_model_classes=list_model_classes,
                             list_model_kwargs=list_model_kwargs,
                             verbose="not_boolean",  # Invalid verbose
                             random_state="not_an_int")  # Invalid random_state

