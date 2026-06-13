"""Equivalence test for the Batch-6 vectorization of ``bic_score_`` (#186).

The per-cluster sum-of-squared-distance was computed with
``distance.cdist(X[mask], [center], 'euclidean') ** 2`` (allocating a full
(n_in_cluster x 1) distance matrix per cluster). The new path uses
``((X[mask] - center) ** 2).sum()``. These are mathematically identical; the
BIC value must match to ``atol=1e-10`` (bit-identical in practice). The original
cdist formulation is reproduced inline as the reference.
"""
import numpy as np
import pytest
from collections import OrderedDict
from scipy.spatial import distance

from aaanalysis._utils.metrics import bic_score_


def _ref_bic_cdist(X, labels):
    """Original BIC using the per-cluster cdist(...)**2 formulation."""
    epsilon = 1e-10
    n_classes = len(set(labels))
    n_samples, n_features = X.shape
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    labels = inverse
    labels_centers = list(OrderedDict.fromkeys(labels))
    list_masks = [[i == label for i in labels] for label in labels_centers]
    centers = np.concatenate([X[m].mean(axis=0)[np.newaxis, :] for m in list_masks]).round(3)
    center_labels = np.array(labels_centers)
    size_clusters = np.bincount(labels)
    masks = [labels == label for label in center_labels]
    sum_squared_dist = sum([sum(distance.cdist(X[m], [c], 'euclidean') ** 2)
                            for m, c in zip(masks, centers)])
    denominator = max((n_samples - n_classes) * n_features, epsilon)
    bet_clu_var = max((1.0 / denominator) * sum_squared_dist, epsilon)
    const_term = 0.5 * n_classes * np.log(n_samples) * (n_features + 1)
    log_size_clusters = np.log(size_clusters + epsilon)
    log_n_samples = np.log(n_samples + epsilon)
    log_bcv = np.log(2 * np.pi * bet_clu_var)
    bic_components = (size_clusters * (log_size_clusters - log_n_samples)
                      - 0.5 * size_clusters * n_features * log_bcv
                      - 0.5 * (size_clusters - 1) * n_features)
    return np.sum(bic_components) - const_term


class TestBicScoreEquivalence:
    """Vectorized bic_score_ == cdist reference within atol=1e-10."""

    @pytest.mark.parametrize("n,d,k,seed", [
        (600, 40, 12, 0),
        (200, 8, 3, 1),
        (50, 5, 2, 7),
        (1000, 64, 25, 42),
    ])
    def test_random_clusters_allclose(self, n, d, k, seed):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, d))
        labels = rng.integers(0, k, size=n)
        got = bic_score_(X, labels=labels)
        ref = _ref_bic_cdist(X, labels)
        assert np.allclose(got, ref, atol=1e-10)

    def test_non_contiguous_string_labels(self):
        """Arbitrary (string / non-0-based) labels: same value as cdist path."""
        rng = np.random.default_rng(3)
        X = rng.standard_normal((120, 10))
        raw = rng.integers(0, 4, size=120)
        labels = np.array(["c%d" % v for v in raw])
        got = bic_score_(X, labels=labels)
        ref = _ref_bic_cdist(X, labels)
        assert np.allclose(got, ref, atol=1e-10)

    def test_singleton_clusters_allclose(self):
        """Clusters of size 1 (zero within-cluster spread) stay identical."""
        rng = np.random.default_rng(5)
        X = rng.standard_normal((30, 6))
        labels = np.arange(15).repeat(2)  # 15 clusters of size 2
        got = bic_score_(X, labels=labels)
        ref = _ref_bic_cdist(X, labels)
        assert np.allclose(got, ref, atol=1e-10)
