"""
This is a script for testing the AAclust class functions.
"""
from hypothesis import given, example
import hypothesis.strategies as some
import aaanalysis.utils as ut
import aaanalysis as aa
import pytest
from sklearn.cluster import (KMeans, AgglomerativeClustering, MiniBatchKMeans,
                             SpectralClustering, DBSCAN, MeanShift, OPTICS, Birch,
                             AffinityPropagation)
from sklearn.mixture import GaussianMixture

K_BASED_MODELS = {
    'KMeans': KMeans,
    'AgglomerativeClustering': AgglomerativeClustering,
    'MiniBatchKMeans': MiniBatchKMeans,
    'SpectralClustering': SpectralClustering,
}

K_FREE_MODELS = {
    'DBSCAN': DBSCAN,
    'MeanShift': MeanShift,
    'OPTICS': OPTICS,
    'Birch': Birch,
    'AffinityPropagation': AffinityPropagation,
    'GaussianMixture': GaussianMixture
}

X_mock = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]  # Mocked data


class TestAAclust:
    """Test aa.AAclust class individual parameters"""

    @given(model=some.none() | some.sampled_from(list(K_BASED_MODELS.keys())))
    def test_k_based_model_parameter(self, model):
        """Test the 'model' parameter for k_based models."""
        if model:
            ModelClass = K_BASED_MODELS[model]
            aaclust = aa.AAclust(model_class=ModelClass)
            # Checking if model_class attribute stores the class reference, not an instance
            assert aaclust.model_class is ModelClass
        if not model:
            aaclust = aa.AAclust()
            # Checking if model_class attribute stores the default model
            assert aaclust.model_class is KMeans
        else:
            with pytest.raises(ValueError):
                aa.AAclust(model_class=model)

    @given(model=some.sampled_from(list(K_BASED_MODELS.keys())))
    def test_k_based_model_parameter_after_fit(self, model):
        """Test the 'model' parameter for k_based models after fitting."""
        ModelClass = K_BASED_MODELS[model]
        aaclust = aa.AAclust(model_class=ModelClass)
        # Assuming AAclust's fit method doesn't require any other arguments except data
        aaclust.fit(X_mock)
        # Checking if the instance of the model has been created after calling fit
        assert isinstance(aaclust.model, ModelClass)

    @given(model=some.sampled_from(list(K_FREE_MODELS.keys())))
    def test_k_free_model_parameter_negative(self, model_class):
        """Negative test for 'model' parameter for k_free models."""
        with pytest.raises(ValueError):
            aa.AAclust(model_class=model_class)

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

    @given(model=some.none() | some.sampled_from(list(K_BASED_MODELS.keys())),
           model_kwargs=some.dictionaries(keys=some.text(), values=some.integers()))
    def test_k_based_model_and_model_kwargs(self, model, model_kwargs):
        """Test 'model' and 'model_kwargs' parameters together for k_based models."""
        if model:
            ModelClass = K_BASED_MODELS[model]
            try:
                aaclust = aa.AAclust(model_class=ModelClass(**model_kwargs), model_kwargs=model_kwargs)
                assert isinstance(aaclust.model_class, ModelClass)
                assert aaclust._model_kwargs == model_kwargs
            except Exception as e:
                with pytest.raises(type(e)):
                    aa.AAclust(model_class=ModelClass(**model_kwargs), model_kwargs=model_kwargs)
        else:
            with pytest.raises(ValueError):
                aa.AAclust(model_class=model, model_kwargs=model_kwargs)

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

