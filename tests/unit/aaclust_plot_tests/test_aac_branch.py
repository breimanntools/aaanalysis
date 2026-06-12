"""
Branch-coverage tests for AAclustPlot public API (frontend + backend plot arcs
reached only through ``aa.AAclustPlot``). All access is via the public class.
"""
import warnings
import numpy as np
import pandas as pd
import pytest
import hypothesis.strategies as some
from hypothesis import given, settings
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.decomposition import FastICA
import aaanalysis as aa

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


def _fit_X_labels(n=20):
    """Real scale matrix + AAclust labels."""
    X = aa.load_scales().to_numpy()[:n]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = aa.AAclust(verbose=False, random_state=42).fit(X).labels_
    return X, labels


class TestCentersNonPCA:
    """centers(...) with a non-PCA decomposition model (aaclust_plot.py L40->43)."""

    def test_non_pca_model_column_names(self):
        X, labels = _fit_X_labels(n=20)
        aacp = aa.AAclustPlot(model_class=FastICA, model_kwargs=dict(max_iter=10),
                              verbose=False, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax, df_components = aacp.centers(X, labels=labels)
        # Non-PCA branch: columns keep model name, no "PC (..%)" rewrite.
        assert all(col.startswith("FastICA") for col in df_components.columns)
        assert "%" not in "".join(df_components.columns)
        plt.close()


class TestCorrelationCbarKwarg:
    """correlation(...) with 'cbar' in kwargs_heatmap (aaclust_plot.py L148->154)."""

    def test_cbar_in_kwargs_skips_cbar_styling(self):
        df_corr = pd.DataFrame(np.random.RandomState(1).rand(4, 4))
        labels = [0, 1, 0, 1]
        aacp = aa.AAclustPlot(verbose=False, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax = aacp.correlation(df_corr=df_corr, labels=labels,
                                  kwargs_heatmap={"cbar": False})
        assert isinstance(ax, plt.Axes)
        plt.close()

    def test_default_cbar_styling(self):
        df_corr = pd.DataFrame(np.random.RandomState(2).rand(4, 4))
        labels = [0, 1, 0, 1]
        aacp = aa.AAclustPlot(verbose=False, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax = aacp.correlation(df_corr=df_corr, labels=labels)
        assert isinstance(ax, plt.Axes)
        plt.close()


class TestCorrelationLabelsRef:
    """correlation(...) labels_ref validation arcs (_aaclust_plot.py L49, L62)."""

    def test_string_columns_require_labels_ref(self):
        # L49: non-pairwise df_corr with string columns and labels_ref=None -> raise.
        df_corr = pd.DataFrame(np.random.RandomState(0).rand(4, 2),
                               columns=["cA", "cB"])
        labels = [0, 1, 0, 1]
        aacp = aa.AAclustPlot(verbose=False, random_state=42)
        with pytest.raises(ValueError, match="'labels_ref' must be provided"):
            aacp.correlation(df_corr=df_corr, labels=labels, labels_ref=None)

    def test_labels_ref_length_mismatch(self):
        # L62: labels_ref length != n_clusters in df_corr -> raise.
        df_corr = pd.DataFrame(np.random.RandomState(0).rand(4, 2))
        labels = [0, 1, 0, 1]
        aacp = aa.AAclustPlot(verbose=False, random_state=42)
        with pytest.raises(ValueError, match="must match with n_clusters"):
            aacp.correlation(df_corr=df_corr, labels=labels, labels_ref=[0, 1, 2])

    @given(seed=some.integers(min_value=0, max_value=50))
    @settings(max_examples=5, deadline=None)
    def test_labels_ref_valid_int_columns(self, seed):
        # Non-pairwise df_corr with integer columns + matching labels_ref passes.
        df_corr = pd.DataFrame(np.random.RandomState(seed).rand(4, 2))
        labels = [0, 1, 0, 1]
        aacp = aa.AAclustPlot(verbose=False, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax = aacp.correlation(df_corr=df_corr, labels=labels, labels_ref=[0, 1])
        assert isinstance(ax, plt.Axes)
        plt.close()
