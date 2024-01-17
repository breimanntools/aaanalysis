"""This script tests the ShapExplainer class initialization and its parameters."""
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
import aaanalysis as aa
import shap


ARGS = dict(explainer_class=shap.TreeExplainer)

class TestShapExplainer:
    """Test ShapExplainer class individual parameters."""

    # Positive tests
    @settings(deadline=1000)
    @given(explainer_class=some.sampled_from([shap.TreeExplainer, shap.LinearExplainer, shap.DeepExplainer]))
    def test_explainer_class_parameter(self, explainer_class):
        # TODO check that only tree Explainer are accepted ...
        explainer = aa.ShapExplainer(explainer_class=explainer_class)
        assert explainer._explainer_class == explainer_class

    def test_explainer_kwargs_parameter(self):
        list_explainer_kwargs = [dict(model=RandomForestClassifier),
                                 dict(model=RandomForestClassifier, data=[[3,4], [3,4 ]]),
                                 dict(feature_perturbation="interventional", model_output="probability"),
                                 dict(model=RandomForestClassifier, feature_perturbation="interventional", model_output="probability")]
        for explainer_kwargs in list_explainer_kwargs:
            explainer = aa.ShapExplainer(explainer_kwargs=explainer_kwargs)
            assert explainer._explainer_kwargs == explainer_kwargs

    @settings(deadline=1000)
    @given(list_model_classes=some.lists(some.sampled_from([RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier]), min_size=1))
    def test_list_model_classes_parameter(self, list_model_classes):
        explainer = aa.ShapExplainer(list_model_classes=list_model_classes)
        assert explainer._list_model_classes == list_model_classes

    @settings(deadline=1000)
    @given(list_model_kwargs=some.lists(some.dictionaries(keys=some.text(), values=some.integers()), min_size=1))
    def test_list_model_kwargs_parameter(self, list_model_kwargs):
        explainer = aa.ShapExplainer(list_model_kwargs=list_model_kwargs)
        assert explainer._list_model_kwargs == list_model_kwargs

    def test_verbose_parameter(self):
        for verbose in [True, False]:
            explainer = aa.ShapExplainer(verbose=verbose)
            assert explainer._verbose == verbose


    def test_random_state_parameter(self):
        """Test the 'random_state' parameter."""
        for random_state in [None, 0, 1, 5, 42]:
            explainer = aa.ShapExplainer(random_state=random_state)
            assert explainer._random_state == random_state


# To run the tests, use a test runner like pytest. Ensure that hypothesis and pytest are installed.
