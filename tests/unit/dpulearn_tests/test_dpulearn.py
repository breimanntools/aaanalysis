"""
This is a script for testing the dPULearn class.
"""
import pytest
import hypothesis.strategies as some
from hypothesis import settings, given
import inspect
from sklearn.decomposition import PCA
import aaanalysis as aa


valid_pca_params = {
    'n_components': [None, 1, 5, 'mle', 0.95],
    'copy': [True, False],
    'whiten': [True, False],
    'svd_solver': ['auto', 'full', 'arpack', 'randomized'],
    'tol': [0.0, 0.01, 0.1],
    'iterated_power': ['auto', 2, 3, 5],
    'random_state': [None, 42],
}


class TestdPULearn:
    """Test dPULearn class individual parameters"""

    # Positive tests
    @settings(deadline=1000, max_examples=2)
    @given(verbose=some.booleans())
    def test_verbose_parameter(self, verbose):
        """Test the 'verbose' parameter."""
        aa.options["verbose"] = "off"
        model = aa.dPULearn(verbose=verbose)
        assert model._verbose == verbose

    @settings(deadline=1000, max_examples=10)
    @given(data=some.data())
    def test_pca_kwargs_various_combinations(self, data):
        """Test the 'model_kwargs' parameter with various valid combinations."""
        # Generate a model_kwargs dictionary with a combination of parameters
        model_kwargs = {param: data.draw(some.sampled_from(values)) for param, values in valid_pca_params.items()}
        valid_args = list(inspect.signature(PCA).parameters.keys())
        model_kwargs = {key: model_kwargs[key] for key in model_kwargs if key in valid_args}
        # Create the dPULearn model with generated model_kwargs
        model = aa.dPULearn(model_kwargs=model_kwargs)
        # Assert that the model_kwargs in the model match the generated ones
        if "random_state" not in model_kwargs:
            model_kwargs.update(dict(random_state=None))
        assert model._model_kwargs == model_kwargs


    # Negative tests
    @settings(deadline=1000, max_examples=10)
    @given(pca_kwargs=some.dictionaries(keys=some.text(), values=some.just("invalid_value")))
    def test_pca_kwargs_negative(self, pca_kwargs):
        """Test the 'pca_kwargs' parameter with invalid PCA arguments."""
        valid_args = list(inspect.signature(PCA).parameters.keys())
        pca_kwargs = {key: pca_kwargs[key] for key in pca_kwargs if key not in valid_args}
        pca_kwargs["invalid_param"] = "invalid_value"   # At least one invalid
        with pytest.raises(ValueError):
            aa.dPULearn(model_kwargs=pca_kwargs)


class TestdPULearnComplex:
    """Test dPULearn class with complex scenarios"""

    # Positive tests
    @settings(deadline=1000, max_examples=10)
    @given(verbose=some.booleans(), model_kwargs=some.dictionaries(keys=some.text(), values=some.integers()))
    def test_verbose_and_pca_kwargs(self, verbose, model_kwargs):
        """Test 'verbose' and 'pca_kwargs' parameters together."""
        valid_args = list(inspect.signature(PCA).parameters.keys())
        model_kwargs = {key: model_kwargs[key] for key in model_kwargs if key in valid_args}
        model = aa.dPULearn(verbose=verbose, model_kwargs=model_kwargs)
        if "random_state" not in model_kwargs:
            model_kwargs.update(dict(random_state=None))
        assert model._verbose == verbose
        assert model._model_kwargs == model_kwargs


    # Negative tests
    @settings(deadline=1000, max_examples=10)
    @given(verbose=some.booleans(), model_kwargs=some.dictionaries(keys=some.text(), values=some.just("invalid_value")))
    def test_valid_verbose_and_invalid_pca_and_kwargs(self, verbose, model_kwargs):
        """Test combining valid 'verbose' with invalid 'pca_kwargs'."""
        valid_args = list(inspect.signature(PCA).parameters.keys())
        model_kwargs = {key: model_kwargs[key] for key in model_kwargs if key not in valid_args}
        model_kwargs["invalid_param"] = "invalid_value"   # At least one invalid
        with pytest.raises(ValueError):
            aa.dPULearn(verbose=verbose, model_kwargs=model_kwargs)
