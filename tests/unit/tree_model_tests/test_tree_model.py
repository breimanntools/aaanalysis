"""This script tests the TreeModel class initialization and its parameters."""
import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as some
import pytest
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import aaanalysis as aa

aa.options["verbose"] = "off"

# Mock data for testing
mock_classes = [RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier]
mock_kwargs = [{'n_estimators': 10}, {'min_samples_split': 2}, {'learning_rate': 0.1}]
mock_preselected = [True, False, True, False]

class TestTreeModel:
    """Test TreeModel class individual parameters"""

    # Positive tests
    def test_list_model_kwargs_parameter(self):
        """Test the 'list_model_kwargs' parameter."""
        tree_model = aa.TreeModel(list_model_kwargs=mock_kwargs)
        assert tree_model._list_model_kwargs == mock_kwargs
        tree_model = aa.TreeModel(list_model_kwargs=mock_kwargs[0:1], list_model_classes=mock_classes[0:1])
        assert tree_model._list_model_kwargs == mock_kwargs[0:1]
        tree_model = aa.TreeModel(list_model_kwargs=mock_kwargs[0:2], list_model_classes=mock_classes[0:2])
        assert tree_model._list_model_kwargs == mock_kwargs[0:2]

    @settings(deadline=1000)
    @given(is_preselected=some.lists(some.booleans(), min_size=2, max_size=5))
    def test_is_preselected_parameter(self, is_preselected):
        """Test the 'is_preselected' parameter."""
        if sum(is_preselected) > 1:

            tree_model = aa.TreeModel(is_preselected=is_preselected)
            assert np.array_equal(tree_model._is_preselected, is_preselected)
            tree_model = aa.TreeModel(is_preselected=mock_preselected)
            assert np.array_equal(tree_model._is_preselected, mock_preselected)

    @settings(deadline=1000)
    @given(verbose=some.booleans())
    def test_verbose_parameter(self, verbose):
        """Test the 'verbose' parameter."""
        aa.options["verbose"] = "off"
        tree_model = aa.TreeModel(verbose=verbose)
        assert tree_model._verbose == verbose

    @settings(deadline=1000)
    @given(random_state=some.integers() | some.none())
    def test_random_state_parameter(self, random_state):
        """Test the 'random_state' parameter."""
        if random_state is None or random_state >= 0:
            tree_model = aa.TreeModel(random_state=random_state)
            assert tree_model._random_state == random_state

    # Negative tests
    def test_invalid_list_model_classes_type(self):
        """Test invalid type for 'list_model_classes' parameter."""
        with pytest.raises(ValueError):
            aa.TreeModel(list_model_classes="Not a list")

    def test_invalid_list_model_classes_content(self):
        """Test invalid content in 'list_model_classes' parameter."""
        with pytest.raises(ValueError):
            aa.TreeModel(list_model_classes=[RandomForestClassifier, "Not a model class"])

    def test_empty_list_model_classes(self):
        """Test empty list for 'list_model_classes' parameter."""
        with pytest.raises(ValueError):
            aa.TreeModel(list_model_classes=[])
        with pytest.raises(ValueError):
            aa.TreeModel(list_model_classes=[KNeighborsClassifier])

    def test_invalid_list_model_kwargs_type(self):
        """Test invalid type for 'list_model_kwargs' parameter."""
        with pytest.raises(ValueError):
            aa.TreeModel(list_model_kwargs="Not a list")

    def test_invalid_list_model_kwargs_content(self):
        """Test invalid content in 'list_model_kwargs' parameter."""
        with pytest.raises(ValueError):
            aa.TreeModel(list_model_kwargs=[{"n_estimators": 10}, "Not a dict"])

    def test_empty_list_model_kwargs(self):
        """Test empty list for 'list_model_kwargs' parameter."""
        with pytest.raises(ValueError):
            aa.TreeModel(list_model_kwargs=[])

    def test_invalid_is_preselected_type(self):
        """Test invalid type for 'is_preselected' parameter."""
        with pytest.raises(ValueError):
            aa.TreeModel(is_preselected="Not a list")

    def test_invalid_is_preselected_content(self):
        """Test invalid content in 'is_preselected' parameter."""
        with pytest.raises(ValueError):
            aa.TreeModel(is_preselected=[True, "Not a boolean"])

    def test_invalid_verbose_type(self):
        """Test invalid type for 'verbose' parameter."""
        with pytest.raises(ValueError):
            aa.TreeModel(verbose="Not a boolean")

    def test_invalid_random_state_type(self):
        """Test invalid type for 'random_state' parameter."""
        with pytest.raises(ValueError):
            aa.TreeModel(random_state="Not an integer or None")


class TestTreeModelComplex:
    """Test TreeModel class with complex scenarios"""


    @settings(deadline=1000, max_examples=10)
    @given(verbose=some.booleans(), random_state=some.integers())
    def test_verbose_and_random_state(self, verbose, random_state):
        """Test 'verbose' and 'random_state' parameters together."""
        aa.options["random_state"] = "off"
        if random_state > -1:
            tree_model = aa.TreeModel(verbose=verbose, random_state=random_state)
            assert tree_model._verbose == verbose
            assert tree_model._random_state == random_state

    @settings(deadline=1000, max_examples=10)
    @given(is_preselected=some.lists(some.booleans(), min_size=2, max_size=5),
           random_state=some.integers() | some.none())
    def test_is_preselected_and_random_state(self, is_preselected, random_state):
        """Test 'is_preselected' and 'random_state' parameters together."""
        if sum(is_preselected) > 1 and (random_state >= 0 or random_state is None):
            tree_model = aa.TreeModel(is_preselected=is_preselected, random_state=random_state)
            assert np.array_equal(tree_model._is_preselected, is_preselected)
            assert tree_model._random_state == random_state

    @settings(deadline=1000, max_examples=10)
    @given(list_model_classes=some.lists(some.sampled_from(mock_classes), min_size=1, max_size=3),
           is_preselected=some.lists(some.booleans(), min_size=1, max_size=5))
    def test_model_classes_and_is_preselected(self, list_model_classes, is_preselected):
        """Test 'list_model_classes' and 'is_preselected' parameters together."""
        try:
            tree_model = aa.TreeModel(list_model_classes=list_model_classes, is_preselected=is_preselected)
            assert tree_model._list_model_classes == list_model_classes
            assert np.array_equal(tree_model._is_preselected, is_preselected)
        except Exception as e:
            with pytest.raises(type(e)):
                aa.TreeModel(list_model_classes=list_model_classes, is_preselected=is_preselected)

    # Negative tests
    def test_invalid_combination_model_classes_and_kwargs(self):
        """Test invalid combination of 'list_model_classes' and 'list_model_kwargs' parameters."""
        with pytest.raises(ValueError):
            invalid_kwargs = [{'invalid_param': 10}] * 3
            aa.TreeModel(list_model_classes=mock_classes, list_model_kwargs=invalid_kwargs)

    def test_mismatch_length_model_classes_and_kwargs(self):
        """Test mismatch in length of 'list_model_classes' and 'list_model_kwargs'."""
        with pytest.raises(ValueError):
            mismatched_kwargs = mock_kwargs[:2]  # Shorter than mock_classes
            aa.TreeModel(list_model_classes=mock_classes, list_model_kwargs=mismatched_kwargs)
