"""
This is a script for testing the AAclust class.
"""
from hypothesis import given, example, settings
import hypothesis.strategies as some
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

    @given(model_class_name=some.none() | some.sampled_from(list(K_BASED_MODELS.keys())))
    def test_k_based_model_parameter(self, model_class_name):
        """Test the 'model' parameter for k_based models."""
        if model_class_name:
            ModelClass = K_BASED_MODELS[model_class_name]
            aaclust = aa.AAclust(model_class=ModelClass)
            assert aaclust.model_class is ModelClass
        elif model_class_name is None:
            aaclust = aa.AAclust()
            assert aaclust.model_class is KMeans
        else:
            with pytest.raises(ValueError):
                aa.AAclust(model_class=model_class_name)


    @given(model_class_name=some.sampled_from(list(K_BASED_MODELS.keys())))
    def test_k_based_model_parameter_after_fit(self, model_class_name):
        """Test the 'model' parameter for k_based models after fitting."""
        ModelClass = K_BASED_MODELS[model_class_name]
        aaclust = aa.AAclust(model_class=ModelClass)
        aaclust.fit(X_mock)
        assert isinstance(aaclust.model, ModelClass)

    @given(model_class_name=some.sampled_from(list(K_FREE_MODELS.keys())))
    def test_k_free_model_parameter_negative(self, model_class_name):
        """Negative test for 'model' parameter for k_free models."""
        with pytest.raises(ValueError):
            aa.AAclust(model_class=model_class_name)

    @settings(deadline=1000, max_examples=10)
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

    @given(model_class_name=some.none() | some.sampled_from(list(K_BASED_MODELS.keys())),
           model_kwargs=some.dictionaries(keys=some.text(), values=some.integers()))
    def test_k_based_model_and_model_kwargs(self, model_class_name, model_kwargs):
        """Test 'model' and 'model_kwargs' parameters together for k_based models."""
        if model_class_name:
            ModelClass = K_BASED_MODELS[model_class_name]
            try:
                aaclust = aa.AAclust(model_class=ModelClass(**model_kwargs), model_kwargs=model_kwargs)
                assert isinstance(aaclust.model_class, ModelClass)
                assert aaclust._model_kwargs == model_kwargs
            except Exception as e:
                with pytest.raises(type(e)):
                    aa.AAclust(model_class=ModelClass(**model_kwargs), model_kwargs=model_kwargs)
        else:
            with pytest.raises(ValueError):
                aa.AAclust(model_class=model_class_name, model_kwargs=model_kwargs)

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

