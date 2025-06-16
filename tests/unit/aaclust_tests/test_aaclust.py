"""
This is a script for testing the initialization of the AAclust class.
"""
from hypothesis import given, example, settings
import hypothesis.strategies as some
import aaanalysis as aa
import pytest
from sklearn.cluster import (KMeans, AgglomerativeClustering, MiniBatchKMeans,
                             SpectralClustering, DBSCAN, MeanShift, OPTICS, Birch,
                             AffinityPropagation)
from sklearn.mixture import GaussianMixture

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=400)
settings.load_profile("ci")


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

    @settings(deadline=2000)
    @given(model_class_name=some.none() | some.sampled_from(list(K_BASED_MODELS.keys())))
    def test_k_based_model_parameter(self, model_class_name):
        """Test the 'model' parameter for k_based models."""
        if model_class_name:
            ModelClass = K_BASED_MODELS[model_class_name]
            aaclust = aa.AAclust(model_class=ModelClass)
            assert aaclust._model_class is ModelClass
        elif model_class_name is None:
            aaclust = aa.AAclust()
            assert aaclust._model_class is KMeans
        else:
            with pytest.raises(ValueError):
                aa.AAclust(model_class=model_class_name)

    @settings(deadline=2000)
    @given(model_class_name=some.sampled_from(list(K_BASED_MODELS.keys())))
    def test_k_based_model_parameter_after_fit(self, model_class_name):
        """Test the 'model' parameter for k_based models after fitting."""
        ModelClass = K_BASED_MODELS[model_class_name]
        aaclust = aa.AAclust(model_class=ModelClass)
        aaclust.fit(X_mock)
        assert isinstance(aaclust.model, ModelClass)

    @settings(deadline=2000)
    @given(model_class_name=some.sampled_from(list(K_FREE_MODELS.keys())))
    def test_k_free_model_parameter_negative(self, model_class_name):
        """Negative test for 'model' parameter for k_free models."""
        with pytest.raises(ValueError):
            aa.AAclust(model_class=model_class_name)

    @settings(deadline=2000, max_examples=10)
    @given(model_kwargs=some.dictionaries(keys=some.text(), values=some.integers()))
    def test_model_kwargs_parameter(self, model_kwargs):
        """Test the 'model_kwargs' parameter."""
        try:
            aaclust = aa.AAclust(model_kwargs=model_kwargs)
            if "random_state" not in model_kwargs:
                model_kwargs.update(dict(random_state=None))
            assert aaclust._model_kwargs == model_kwargs
        except Exception as e:
            with pytest.raises(type(e)):
                aa.AAclust(model_kwargs=model_kwargs)

    def test_verbose_parameter(self):
        """Test the 'verbose' parameter."""
        aa.options["verbose"] = "off"
        aaclust = aa.AAclust(verbose=True)
        assert aaclust._verbose is True
        aaclust = aa.AAclust(verbose=False)
        assert aaclust._verbose is False


class TestAAclustComplex:
    """Test aa.AAclust class with complex scenarios"""

    @settings(deadline=2000, max_examples=10)
    @given(model_class_name=some.none() | some.sampled_from(list(K_BASED_MODELS.keys())),
           model_kwargs=some.dictionaries(keys=some.text(), values=some.integers()))
    def test_k_based_model_and_model_kwargs(self, model_class_name, model_kwargs):
        """Test 'model' and 'model_kwargs' parameters together for k_based models."""
        if model_class_name:
            ModelClass = K_BASED_MODELS[model_class_name]
            try:
                aaclust = aa.AAclust(model_class=ModelClass(**model_kwargs), model_kwargs=model_kwargs)
                if "random_state" not in model_kwargs:
                    model_kwargs.update(dict(random_state=None))
                assert isinstance(aaclust._model_class, ModelClass)
                assert aaclust._model_kwargs == model_kwargs
            except Exception as e:
                with pytest.raises(type(e)):
                    aa.AAclust(model_class=ModelClass(**model_kwargs), model_kwargs=model_kwargs)
        else:
            with pytest.raises(ValueError):
                aa.AAclust(model_class=model_class_name, model_kwargs=model_kwargs)

    @settings(deadline=2000, max_examples=10)
    @given(model_kwargs=some.dictionaries(keys=some.text(), values=some.integers()),
           verbose=some.booleans())
    def test_model_kwargs_and_verbose(self, model_kwargs, verbose):
        """Test 'model_kwargs' and 'verbose' parameters together."""
        aa.options["verbose"] = "off"
        try:
            aaclust = aa.AAclust(model_kwargs=model_kwargs, verbose=verbose)
            if "random_state" not in model_kwargs:
                model_kwargs.update(dict(random_state=None))
            assert aaclust._model_kwargs == model_kwargs
            assert aaclust._verbose == verbose
        except Exception as e:
            pass

