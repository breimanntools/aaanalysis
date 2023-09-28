"""
This is a script for testing the AAclust class functions.
"""
from hypothesis import given, example
import hypothesis.strategies as some
import aaanalysis.utils as ut
import aaanalysis as aa
import pytest


class TestAAclust:
    """Test aa.AAclust class individual parameters"""

    @given(model=some.none() | some.just('KMeans'))
    def test_model_parameter(self, model):
        """Test the 'model' parameter."""
        if model:
            aaclust = aa.AAclust(model=model)
            assert isinstance(aaclust.model, type(model))
        else:
            with pytest.raises(ValueError):
                aa.AAclust(model=model)

    @given(model_kwargs=some.dictionaries(keys=some.text(), values=some.integers()))
    def test_model_kwargs_parameter(self, model_kwargs):
        """Test the 'model_kwargs' parameter."""
        try:
            aaclust = aa.AAclust(model_kwargs=model_kwargs)
            assert aaclust._model_kwargs == model_kwargs
        except Exception as e:
            with pytest.raises(type(e)):
                aa.AAclust(model_kwargs=model_kwargs)

    @given(verbose=some.booleans())
    def test_verbose_parameter(self, verbose):
        """Test the 'verbose' parameter."""
        aaclust = aa.AAclust(verbose=verbose)
        assert aaclust._verbose == verbose


class TestAAclustComplex:
    """Test aa.AAclust class with complex scenarios"""

    @given(model=some.none() | some.just('KMeans'),
           model_kwargs=some.dictionaries(keys=some.text(), values=some.integers()))
    def test_model_and_model_kwargs(self, model, model_kwargs):
        """Test 'model' and 'model_kwargs' parameters together."""
        try:
            aaclust = aa.AAclust(model=model, model_kwargs=model_kwargs)
            assert isinstance(aaclust.model, type(model))
            assert aaclust._model_kwargs == model_kwargs
        except Exception as e:
            with pytest.raises(type(e)):
                aa.AAclust(model=model, model_kwargs=model_kwargs)

    @given(model_kwargs=some.dictionaries(keys=some.text(), values=some.integers()),
           verbose=some.booleans())
    def test_model_kwargs_and_verbose(self, model_kwargs, verbose):
        """Test 'model_kwargs' and 'verbose' parameters together."""
        try:
            aaclust = aa.AAclust(model_kwargs=model_kwargs, verbose=verbose)
            assert aaclust._model_kwargs == model_kwargs
            assert aaclust._verbose == verbose
        except Exception as e:
            with pytest.raises(type(e)):
                aa.AAclust(model_kwargs=model_kwargs, verbose=verbose)

    @given(model=some.none() | some.just('KMeans'),
           verbose=some.booleans())
    def test_model_and_verbose(self, model, verbose):
        """Test 'model' and 'verbose' parameters together."""
        try:
            aaclust = aa.AAclust(model=model, verbose=verbose)
            assert isinstance(aaclust.model, type(model))
            assert aaclust._verbose == verbose
        except Exception as e:
            with pytest.raises(type(e)):
                aa.AAclust(model=model, verbose=verbose)

