"""This script tests the ShapExplainer class initialization and its parameters."""
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
import shap

import aaanalysis as aa


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
        list_explainer_kwargs = [dict(model=RandomForestClassifier),
                                 dict(model=RandomForestClassifier, data=[[3,4], [3, 4]]),
                                 dict(feature_perturbation="interventional", model_output="probability"),
                                 dict(model=RandomForestClassifier, feature_perturbation="interventional", model_output="probability")]
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
        """
        # TODO
        se = aa.ShapExplainer(list_model_kwargs=list_model_kwargs)
        assert se._list_model_kwargs == list_model_kwargs
        """

    def test_verbose_parameter(self):
        for verbose in [True, False]:
            se = aa.ShapExplainer(verbose=verbose)
            assert se._verbose == verbose


    def test_random_state_parameter(self):
        """Test the 'random_state' parameter."""
        for random_state in [None, 0, 1, 5, 42]:
            explainer = aa.ShapExplainer(random_state=random_state)
            assert explainer._random_state == random_state


    # Negative tests