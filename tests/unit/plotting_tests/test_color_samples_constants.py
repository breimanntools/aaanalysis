"""
This script tests the named sample-color constants exposed at top level (issue #308).

COLOR_SAMPLES_POS / NEG / UNL / REL_NEG are public, named aliases for the canonical sample
colors. They must equal today's ``plot_get_cdict("DICT_COLOR")["SAMPLES_*"]`` values exactly,
so users can reference a named constant instead of indexing the color dict by string key.
"""
import pytest

import aaanalysis as aa


# Golden equivalence test
class TestColorSamplesConstants:
    """Named constants must equal the plot_get_cdict values (golden KPI #308)."""

    def test_constants_exist_at_top_level(self):
        for name in ("COLOR_SAMPLES_POS", "COLOR_SAMPLES_NEG",
                     "COLOR_SAMPLES_UNL", "COLOR_SAMPLES_REL_NEG"):
            assert hasattr(aa, name)
            assert name in aa.__all__

    @pytest.mark.parametrize("const_name,dict_key", [
        ("COLOR_SAMPLES_POS", "SAMPLES_POS"),
        ("COLOR_SAMPLES_NEG", "SAMPLES_NEG"),
        ("COLOR_SAMPLES_UNL", "SAMPLES_UNL"),
        ("COLOR_SAMPLES_REL_NEG", "SAMPLES_REL_NEG"),
    ])
    def test_constant_equals_cdict_value(self, const_name, dict_key):
        dict_color = aa.plot_get_cdict(name="DICT_COLOR")
        assert getattr(aa, const_name) == dict_color[dict_key]

    def test_constants_are_strings(self):
        for name in ("COLOR_SAMPLES_POS", "COLOR_SAMPLES_NEG",
                     "COLOR_SAMPLES_UNL", "COLOR_SAMPLES_REL_NEG"):
            assert isinstance(getattr(aa, name), str)
