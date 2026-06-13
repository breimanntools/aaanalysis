"""Test that the vectorized ``_dist_to_medoids`` correlation branch matches a per-pair
reference (``1 - np.corrcoef``) and that the non-correlation metric branch is unchanged."""
import numpy as np
import pytest
from sklearn.metrics import pairwise_distances

import aaanalysis as aa
from aaanalysis.feature_engineering._backend.aaclust._utils_aaclust import _dist_to_medoids

aa.options["verbose"] = False


def _ref_dist_to_medoids(X, labels, medoid_ind, labels_medoids, metric="correlation"):
    """Reference per-pair implementation (the original loop)."""
    labels = np.asarray(labels)
    label_to_medoid = {lab: int(medoid_ind[j]) for j, lab in enumerate(labels_medoids)}
    dist = np.zeros(len(labels), dtype=float)
    for i in range(len(labels)):
        m = label_to_medoid[labels[i]]
        if i == m:
            continue
        if metric == "correlation":
            dist[i] = 1.0 - np.corrcoef(X[i], X[m])[0, 1]
        else:
            dist[i] = pairwise_distances(X[m].reshape(1, -1), X[i].reshape(1, -1), metric=metric)[0, 0]
    return dist


def _make(seed, n=400, F=40, k=8):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, F))
    labels = rng.integers(0, k, size=n)
    labels_medoids = np.arange(k)
    medoid_ind = [int(np.where(labels == c)[0][0]) for c in range(k)]
    return X, labels, medoid_ind, labels_medoids


class TestDistToMedoids:
    """The vectorized correlation branch must match the per-pair reference."""

    @pytest.mark.parametrize("seed", [0, 1, 5, 42])
    def test_correlation_matches_reference(self, seed):
        X, labels, medoid_ind, labels_medoids = _make(seed)
        new = _dist_to_medoids(X, labels=labels, medoid_ind=medoid_ind,
                               labels_medoids=labels_medoids, metric="correlation")
        ref = _ref_dist_to_medoids(X, labels, medoid_ind, labels_medoids, metric="correlation")
        np.testing.assert_allclose(new, ref, rtol=0, atol=1e-10, equal_nan=True)

    def test_medoids_have_zero_distance(self):
        X, labels, medoid_ind, labels_medoids = _make(0)
        new = _dist_to_medoids(X, labels=labels, medoid_ind=medoid_ind,
                               labels_medoids=labels_medoids, metric="correlation")
        for mi in medoid_ind:
            assert new[mi] == 0.0

    @pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
    def test_non_correlation_metric_unchanged(self, metric):
        X, labels, medoid_ind, labels_medoids = _make(3)
        new = _dist_to_medoids(X, labels=labels, medoid_ind=medoid_ind,
                               labels_medoids=labels_medoids, metric=metric)
        ref = _ref_dist_to_medoids(X, labels, medoid_ind, labels_medoids, metric=metric)
        np.testing.assert_allclose(new, ref, rtol=0, atol=1e-12)
