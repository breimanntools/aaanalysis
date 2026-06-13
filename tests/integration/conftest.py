"""Fixtures for the integration tier.

Wraps the shared seeded builders in ``tests/_pipeline.py`` as session-scoped
fixtures so the (relatively expensive) ``CPP.run`` spine is built once and
reused across the seam tests. Treat the returned artifacts as read-only.
"""
import pytest

from tests import _pipeline


@pytest.fixture(scope="session")
def pipeline():
    """The shared load -> parts -> CPP -> feature-matrix artifacts (built once)."""
    return _pipeline.build_pipeline()


@pytest.fixture(scope="session")
def small_scales():
    """A small, fixed ``df_scales`` (rows = amino acids, cols = scales)."""
    return _pipeline.small_scales()
