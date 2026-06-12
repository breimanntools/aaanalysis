"""
Branch-coverage tests for AAclust public API (frontend + backend arcs reached
only through ``aa.AAclust``). All access is via the public class; no private
backend function is called directly.
"""
import warnings
import numpy as np
import pytest
import hypothesis.strategies as some
from hypothesis import given, settings
import matplotlib
matplotlib.use("Agg")
import aaanalysis as aa

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


def _load_X(n=20):
    """Real scale matrix as input."""
    df_scales = aa.load_scales()
    return df_scales.to_numpy()[:n]


class TestFitNClustersMatch:
    """fit(n_clusters=...) -> check_match_X_n_clusters arcs (_aaclust.py L42, L44)."""

    def test_n_clusters_exceeds_n_samples(self):
        # L42: n_samples < n_clusters
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        aac = aa.AAclust(verbose=False, random_state=42)
        with pytest.raises(ValueError, match="should be >= 'n_clusters'"):
            aac.fit(X, n_clusters=5)

    def test_n_clusters_exceeds_n_unique_samples(self):
        # L44: n_samples >= n_clusters but n_unique_samples < n_clusters.
        # 5 samples (>= 3 unique to pass uniqueness check) but only 4 unique rows,
        # n_clusters=5 -> L42 false (5>=5), L44 true (4<5).
        X = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0],
                      [5.0, 6.0], [7.0, 8.0]])
        aac = aa.AAclust(verbose=False, random_state=42)
        with pytest.raises(ValueError, match="should be >= n_unique_samples"):
            aac.fit(X, n_clusters=5)

    @given(n_clusters=some.integers(min_value=2, max_value=6))
    @settings(max_examples=5, deadline=None)
    def test_n_clusters_valid_positive(self, n_clusters):
        X = _load_X(n=20)
        aac = aa.AAclust(verbose=False, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aac.fit(X, n_clusters=n_clusters)
        assert aac.n_clusters >= 1


class TestEvalWarnCH:
    """eval(...) warn_ch arc (_aaclust.py L395->396; aaclust_eval.py L41-42)."""

    def test_ch_nan_sets_zero_and_warns_when_verbose(self):
        # Overflow inside calinski_harabasz_score -> NaN -> CH set to 0 + warn.
        v = 1e200
        X = np.array([[v, v], [-v, -v], [v * 2, v * 2], [-v * 2, -v * 2]])
        labels = [0, 1, 0, 1]
        aac = aa.AAclust(verbose=True, random_state=42)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            df_eval = aac.eval(X, list_labels=[labels])
        assert (df_eval["CH"] == 0).all()
        assert any("CH was set to 0" in str(w.message) for w in rec)

    def test_ch_nan_silent_when_not_verbose(self):
        # Same degenerate input, verbose=False -> CH set to 0 but no warning.
        v = 1e200
        X = np.array([[v, v], [-v, -v], [v * 2, v * 2], [-v * 2, -v * 2]])
        labels = [0, 1, 0, 1]
        aac = aa.AAclust(verbose=False, random_state=42)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            df_eval = aac.eval(X, list_labels=[labels])
        assert (df_eval["CH"] == 0).all()
        assert not any("CH was set to 0" in str(w.message) for w in rec)


class TestNameClustersComma:
    """name_clusters(...) remove_2nd_info comma/paren arc (aaclust_methods.py L26-28)."""

    def test_name_with_comma_and_paren(self):
        # Names contain ", " inside parentheses -> split on comma + re-close paren.
        X = np.array([[1.0, 2.0], [1.1, 2.1], [5.0, 6.0], [5.1, 6.1]])
        labels = [0, 0, 1, 1]
        names = ["Alpha (helix, prop)", "Alpha (helix, x)", "Beta", "Beta"]
        out = aa.AAclust.name_clusters(X, labels=labels, names=names)
        assert out[0].startswith("Alpha") and "," not in out[0]
        assert out[2] == "Beta"
