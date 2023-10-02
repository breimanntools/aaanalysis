"""This is a script to test the comp_coverage function."""
from hypothesis import given, settings
import hypothesis.strategies as some
from hypothesis.extra import pandas as pdst
import pandas as pd
import numpy as np
import aaanalysis as aa
import pytest


class TestCompCoverage:
    """Test comp_coverage function of the TARGET FUNCTION"""

    @given(names=some.lists(elements=some.text(), min_size=1))
    def test_valid_names(self, names):
        """Test a valid 'names' parameter."""
        names_ref = names + ['additional_name']
        result = aa.AAclust().comp_coverage(names, names_ref)  # Please replace your_class with actual class name
        assert isinstance(result, float)

    @given(names=some.lists(elements=some.text(), min_size=2))
    def test_invalid_names(self, names):
        """Test an invalid 'names' parameter."""
        names = list(set(names))
        names_ref = names[:-1]  # names is a superset of names_ref
        with pytest.raises(ValueError):
            aa.AAclust().comp_coverage(names, names_ref)

    @given(names_ref=some.lists(elements=some.text(), unique=True, min_size=1))
    def test_valid_names_ref(self, names_ref):
        """Test a valid 'names_ref' parameter."""
        names = names_ref[:-1]  # names is a subset of names_ref
        result = aa.AAclust().comp_coverage(names, names_ref)
        assert isinstance(result, float)

    @given(names_ref=some.lists(elements=some.text(), unique=True, min_size=1))
    def test_invalid_names_ref(self, names_ref):
        """Test an invalid 'names_ref' parameter."""
        names = names_ref + ['additional_name']  # names_ref is a subset of names
        with pytest.raises(ValueError):
            aa.AAclust().comp_coverage(names, names_ref)

    def test_names_none(self):
        """Test 'names' parameter set to None."""
        names_ref = ['name']
        with pytest.raises(ValueError):
            aa.AAclust().comp_coverage(None, names_ref)

    def test_names_ref_none(self):
        """Test 'names_ref' parameter set to None."""
        names = ['name']
        with pytest.raises(ValueError):
            aa.AAclust().comp_coverage(names, None)


class TestCompCoverageComplex:
    """Test comp_coverage function of the TARGET FUNCTION for Complex Cases"""

    @settings(deadline=1000, max_examples=5)
    @given(names=some.lists(elements=some.text(), min_size=1),
           additional=some.lists(elements=some.text(), min_size=1))
    def test_combination_valid_parameters(self, names, additional):
        """Test combination of valid parameters."""
        names_ref = names + additional
        result = aa.AAclust().comp_coverage(names, names_ref)
        assert isinstance(result, float)

