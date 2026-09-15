"""This is a script to test the SequenceFeature().get_df_parts() method ."""
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
import aaanalysis as aa

# Set default deadline from 200 to 400
settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


aa.options["verbose"] = False


class TestGetSplitKws:
    """Test the get_split_kws static method."""

    @settings(max_examples=10)
    @given(split_types=st.sampled_from(
        [None, "Segment", "Pattern", "PeriodicPattern", ["Segment", "Pattern"], ["Pattern", "PeriodicPattern"],
         ["Segment", "PeriodicPattern"]]))
    def test_split_types(self, split_types):
        """Test different 'split_types'."""
        sf = aa.SequenceFeature()
        result = sf.get_split_kws(split_types=split_types)
        assert isinstance(result, dict)

    @settings(max_examples=10)
    @given(n_split_min=st.integers(min_value=1, max_value=14))
    def test_n_split_min(self, n_split_min):
        """Test 'n_split_min' within valid range."""
        sf = aa.SequenceFeature()
        result = sf.get_split_kws(n_split_min=n_split_min)
        assert isinstance(result, dict)

    @settings(max_examples=10)
    @given(n_split_max=st.integers(min_value=2, max_value=15))
    def test_n_split_max(self, n_split_max):
        """Test 'n_split_max' within valid range."""
        sf = aa.SequenceFeature()
        result = sf.get_split_kws(n_split_max=n_split_max)
        assert isinstance(result, dict)

    @settings(max_examples=10)
    @given(steps_pattern=st.lists(st.integers(min_value=1), min_size=1, max_size=8))
    def test_steps_pattern(self, steps_pattern):
        """Test 'steps_pattern' with various list sizes."""
        sf = aa.SequenceFeature()
        if len(steps_pattern) > 0:
            # n_min=1 so a single shortest step fits within len_max (avoids the
            # empty-Pattern-bucket warning when len_max barely exceeds a step)
            result = sf.get_split_kws(steps_pattern=steps_pattern, len_max=steps_pattern[0]+1, n_min=1)
            assert isinstance(result, dict)
        result = sf.get_split_kws(steps_pattern=[9, 15], len_max=10, n_min=1)
        assert isinstance(result, dict)


    @settings(max_examples=10)
    @given(n_min=st.integers(min_value=1, max_value=4))
    def test_n_min(self, n_min):
        """Test 'n_min' within valid range."""
        sf = aa.SequenceFeature()
        result = sf.get_split_kws(n_min=n_min)
        assert isinstance(result, dict)

    @settings(max_examples=10)
    @given(n_max=st.integers(min_value=2, max_value=4))
    def test_n_max(self, n_max):
        """Test 'n_max' within valid range."""
        sf = aa.SequenceFeature()
        result = sf.get_split_kws(n_max=n_max)
        assert isinstance(result, dict)

    @settings(max_examples=10)
    @given(len_max=st.integers(min_value=4, max_value=15))
    def test_len_max(self, len_max):
        """Test 'len_max' within valid range."""
        sf = aa.SequenceFeature()
        # n_min=1 so the default steps_pattern fits even at the smallest len_max
        # (default n_min=2 empties the Pattern bucket when 2*min(steps) > len_max)
        result = sf.get_split_kws(len_max=len_max, n_min=1)
        assert isinstance(result, dict)

    @settings(max_examples=10)
    @given(steps_periodicpattern=st.lists(st.integers(min_value=1), min_size=2, max_size=2))
    def test_steps_periodicpattern(self, steps_periodicpattern):
        """Test 'steps_periodicpattern' with various list sizes."""
        sf = aa.SequenceFeature()
        if len(steps_periodicpattern) == 2:
            result = sf.get_split_kws(steps_periodicpattern=steps_periodicpattern)
            assert isinstance(result, dict)

    @settings(max_examples=5, deadline=None)
    @given(strategy=st.sampled_from([None, "compositional", "positional"]))
    def test_strategy(self, strategy):
        """Test valid 'strategy' values."""
        sf = aa.SequenceFeature()
        result = sf.get_split_kws(strategy=strategy)
        assert isinstance(result, dict)
        if strategy == "compositional":
            assert list(result) == ["Segment"]
        else:
            assert list(result) == ["Segment", "Pattern", "PeriodicPattern"]

    def test_strategy_compositional_single_segment(self):
        """'compositional' yields exactly one whole-part Segment."""
        result = aa.SequenceFeature.get_split_kws(strategy="compositional")
        assert result["Segment"] == {"n_split_min": 1, "n_split_max": 1}

    def test_strategy_positional_excludes_whole_part(self):
        """'positional' starts sub-segments at 2 (no whole-part Segment)."""
        result = aa.SequenceFeature.get_split_kws(strategy="positional")
        assert result["Segment"]["n_split_min"] == 2
        assert result["Segment"]["n_split_max"] > 1

    def test_strategy_explicit_defaults_accepted(self):
        """Passing the split args at their defaults alongside 'strategy' does not raise."""
        sf = aa.SequenceFeature()
        expected = sf.get_split_kws(strategy="positional")
        assert sf.get_split_kws(strategy="positional", split_types=None, n_split_min=1, n_split_max=15) == expected

    # Negative tests for each parameter
    def test_invalid_strategy(self):
        """Test invalid 'strategy' values."""
        sf = aa.SequenceFeature()
        for strategy in ["Compositional", "POSITIONAL", "composition", "both", "", "Segment", 1, 1.5, True,
                         ["compositional"], {"positional": 1}]:
            with pytest.raises(ValueError, match="strategy"):
                sf.get_split_kws(strategy=strategy)

    def test_invalid_strategy_with_split_types(self):
        """'strategy' combined with explicit 'split_types' raises."""
        sf = aa.SequenceFeature()
        for split_types in ["Segment", ["Segment"], ["Pattern", "PeriodicPattern"]]:
            for strategy in ["compositional", "positional"]:
                with pytest.raises(ValueError, match="'split_types'"):
                    sf.get_split_kws(strategy=strategy, split_types=split_types)

    @settings(max_examples=5, deadline=None)
    @given(n_split_min=st.integers(min_value=2, max_value=15))
    def test_invalid_strategy_with_n_split_min(self, n_split_min):
        """'strategy' combined with a non-default 'n_split_min' raises."""
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError, match="'n_split_min'"):
            sf.get_split_kws(strategy="compositional", n_split_min=n_split_min)

    @settings(max_examples=5, deadline=None)
    @given(n_split_max=st.integers(min_value=1, max_value=14))
    def test_invalid_strategy_with_n_split_max(self, n_split_max):
        """'strategy' combined with a non-default 'n_split_max' raises."""
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError, match="'n_split_max'"):
            sf.get_split_kws(strategy="positional", n_split_max=n_split_max)

    def test_invalid_split_types(self):
        """Test invalid 'split_types' values."""
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_split_kws(split_types="InvalidType")
        with pytest.raises(ValueError):
            sf.get_split_kws(split_types=["Segment", "InvalidType"])

    def test_invalid_n_split_min(self):
        """Test invalid 'n_split_min' values."""
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_split_kws(n_split_min=0)

    def test_invalid_n_split_max(self):
        """Test invalid 'n_split_max' values."""
        sf = aa.SequenceFeature()
        # Joint constraint: the message names the offending 'n_split_min'/'n_split_max'
        with pytest.raises(ValueError, match="n_split_min"):
            sf.get_split_kws(n_split_max=1, n_split_min=2)
        with pytest.raises(ValueError):
            sf.get_split_kws(n_split_max=0)

    def test_invalid_steps_pattern(self):
        """Test invalid 'steps_pattern' values."""
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_pattern=-1)
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_pattern=["a", "b", "c"])
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_pattern=[0])
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_pattern=[-4, 10])
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_pattern=[])
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_pattern=[3, None])

    def test_invalid_n_min_max(self):
        """Test invalid 'n_min' and 'n_max' values."""
        sf = aa.SequenceFeature()
        # Joint constraint: the message names the offending 'n_min'/'n_max'
        with pytest.raises(ValueError, match="n_min"):
            sf.get_split_kws(n_min=5, n_max=4)
        with pytest.raises(ValueError):
            sf.get_split_kws(n_min=0, n_max=3)

    def test_invalid_len_max(self):
        """Test invalid 'len_max' values."""
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_split_kws(len_max=0)
        # Joint constraint: the message names the offending 'len_max'
        with pytest.raises(ValueError, match="len_max"):
            sf.get_split_kws(len_max=3, steps_pattern=[4, 5])

    def test_invalid_steps_periodicpattern(self):
        """Test invalid 'steps_periodicpattern' values."""
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_periodicpattern=-1)
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_periodicpattern=["a", "b", "c"])
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_periodicpattern=[0])
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_periodicpattern=[-4, 10])
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_periodicpattern=[3, 4, 5])
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_periodicpattern=[])
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_periodicpattern=[3, None])


class TestGetSplitKwsComplex:
    """Test complex combinations of parameters in get_split_kws."""

    @settings(max_examples=5, deadline=None)
    @given(
        split_types=st.sampled_from([None, "Segment", "Pattern", "PeriodicPattern", ["Segment", "Pattern"], ["Pattern", "PeriodicPattern"], ["Segment", "PeriodicPattern"]]),
        n_split_min=st.integers(min_value=1, max_value=14),
        n_split_max=st.integers(min_value=2, max_value=15),
        steps_pattern=st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=8),
        n_min=st.integers(min_value=1, max_value=4),
        n_max=st.integers(min_value=2, max_value=4),
        len_max=st.integers(min_value=4, max_value=15),
        steps_periodicpattern=st.lists(st.integers(min_value=1), min_size=2, max_size=2)
    )
    def test_valid_combinations(self, split_types, n_split_min, n_split_max, steps_pattern, n_min, n_max, len_max, steps_periodicpattern):
        """Test valid combinations of parameters."""
        sf = aa.SequenceFeature()
        if n_split_min > n_split_max:
            n_split_min, n_split_max = n_split_max, n_split_min  # Ensure min <= max
        if n_min > n_max:
            n_min, n_max = n_max, n_min  # Ensure n_min <= n_max
        if steps_pattern:
            lo = min(steps_pattern)
            # len_max must exceed the smallest step (else ValueError) and admit the
            # shortest pattern span n_min*lo (else the Pattern bucket is empty -> warns)
            len_max = max(len_max, lo + 1, n_min * lo)
        if len(steps_pattern) > 1 and len(steps_periodicpattern) == 2:
            result = sf.get_split_kws(split_types=split_types, n_split_min=n_split_min, n_split_max=n_split_max,
                                      steps_pattern=steps_pattern, n_min=n_min, n_max=n_max,
                                      len_max=len_max, steps_periodicpattern=steps_periodicpattern)
            assert isinstance(result, dict)

    @settings(max_examples=5, deadline=None)
    @given(
        split_types=st.sampled_from(["Segment", ["Pattern", "PeriodicPattern"]]),
        n_split_min=st.integers(min_value=1, max_value=3),
        n_split_max=st.integers(min_value=10, max_value=15),
        steps_pattern=st.just([3, 4, 5]),
        n_min=st.integers(min_value=1, max_value=2),
        n_max=st.integers(min_value=3, max_value=4),
        len_max=st.integers(min_value=5, max_value=10),
        steps_periodicpattern=st.just([3, 4])
    )
    def test_edge_case_combinations(self, split_types, n_split_min, n_split_max, steps_pattern, n_min, n_max, len_max, steps_periodicpattern):
        """Test edge case combinations of parameters."""
        sf = aa.SequenceFeature()
        if n_split_min > n_split_max:
            n_split_min, n_split_max = n_split_max, n_split_min  # Ensure min <= max
        if n_min > n_max:
            n_min, n_max = n_max, n_min  # Ensure n_min <= n_max
        lo = min(steps_pattern)
        # admit the shortest pattern span n_min*lo so the Pattern bucket is
        # non-empty (else check_split_kws warns about zero Pattern features)
        len_max = max(len_max, lo + 1, n_min * lo)
        if len(steps_pattern) > 1 and len(steps_periodicpattern) == 2:
            result = sf.get_split_kws(split_types=split_types, n_split_min=n_split_min, n_split_max=n_split_max,
                                      steps_pattern=steps_pattern, n_min=n_min, n_max=n_max,
                                      len_max=len_max, steps_periodicpattern=steps_periodicpattern)
            assert isinstance(result, dict)

    @settings(max_examples=5, deadline=None)
    @given(
        split_types=st.sampled_from([None, ["Segment", "Pattern"]]),
        n_split_min=st.integers(min_value=5, max_value=7),
        n_split_max=st.integers(min_value=8, max_value=10),
        steps_pattern=st.lists(st.integers(min_value=1, max_value=2), min_size=2, max_size=4),
        n_min=st.integers(min_value=1, max_value=2),
        n_max=st.integers(min_value=2, max_value=5),
        len_max=st.integers(min_value=11, max_value=15),
        steps_periodicpattern=st.lists(st.integers(min_value=1, max_value=3), min_size=2, max_size=4)
    )
    def test_random_combinations(self, split_types, n_split_min, n_split_max, steps_pattern, n_min, n_max, len_max, steps_periodicpattern):
        """Test random valid combinations of parameters."""
        sf = aa.SequenceFeature()
        if len(steps_pattern) > 1 and len(steps_periodicpattern) == 2:
            result = sf.get_split_kws(split_types=split_types, n_split_min=n_split_min, n_split_max=n_split_max,
                                      steps_pattern=steps_pattern, n_min=n_min, n_max=n_max,
                                      len_max=len_max, steps_periodicpattern=steps_periodicpattern)
            assert isinstance(result, dict)

    # Negative complex cases
    def test_invalid_combinations(self):
        """Test invalid combinations of parameters."""
        sf = aa.SequenceFeature()
        # Example of an invalid combination
        with pytest.raises(ValueError):
            sf.get_split_kws(n_split_min=15, n_split_max=14)
        with pytest.raises(ValueError):
            sf.get_split_kws(steps_pattern=[1, 2], len_max=1)

    def test_invalid_random_combinations(self):
        """Test invalid random combinations of parameters."""
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError):
            sf.get_split_kws(n_split_min=10, n_split_max=5, steps_pattern=[5, 2, 3], len_max=1)
        with pytest.raises(ValueError):
            sf.get_split_kws(split_types=["Invalid", "Segment"], n_min=4, n_max=3)


class TestGetSplitKwsStrategyComplex:
    """Test 'strategy' combined with the remaining split parameters."""

    @settings(max_examples=5, deadline=None)
    @given(steps_pattern=st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=4),
           n_min=st.integers(min_value=1, max_value=2),
           n_max=st.integers(min_value=2, max_value=4),
           steps_periodicpattern=st.lists(st.integers(min_value=1, max_value=6), min_size=2, max_size=2))
    def test_positional_with_pattern_args(self, steps_pattern, n_min, n_max, steps_periodicpattern):
        """'positional' honors Pattern / PeriodicPattern args like the manual call."""
        sf = aa.SequenceFeature()
        kws = dict(steps_pattern=steps_pattern, n_min=n_min, n_max=n_max, len_max=15,
                   steps_periodicpattern=steps_periodicpattern)
        result = sf.get_split_kws(strategy="positional", **kws)
        manual = sf.get_split_kws(split_types=["Segment", "Pattern", "PeriodicPattern"],
                                  n_split_min=2, n_split_max=15, **kws)
        assert result == manual
        assert result["Pattern"]["steps"] == sorted(steps_pattern)

    @settings(max_examples=5, deadline=None)
    @given(steps_pattern=st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=4),
           steps_periodicpattern=st.lists(st.integers(min_value=1, max_value=6), min_size=2, max_size=2))
    def test_compositional_with_pattern_args(self, steps_pattern, steps_periodicpattern):
        """'compositional' ignores Pattern / PeriodicPattern args like split_types='Segment' does."""
        sf = aa.SequenceFeature()
        result = sf.get_split_kws(strategy="compositional", steps_pattern=steps_pattern, n_min=1,
                                  steps_periodicpattern=steps_periodicpattern)
        assert result == {"Segment": {"n_split_min": 1, "n_split_max": 1}}

    def test_presets_partition_default(self):
        """Compositional plus positional cover exactly the default split set."""
        sf = aa.SequenceFeature()
        comp = sf.get_split_kws(strategy="compositional")
        pos = sf.get_split_kws(strategy="positional")
        default = sf.get_split_kws()
        assert set(comp) | set(pos) == set(default)
        assert comp["Segment"]["n_split_min"] == default["Segment"]["n_split_min"]
        assert comp["Segment"]["n_split_max"] + 1 == pos["Segment"]["n_split_min"]
        assert pos["Segment"]["n_split_max"] == default["Segment"]["n_split_max"]
        assert pos["Pattern"] == default["Pattern"]
        assert pos["PeriodicPattern"] == default["PeriodicPattern"]

    def test_presets_yield_features(self):
        """Both presets are consumable by get_features and give disjoint, non-empty feature sets."""
        sf = aa.SequenceFeature()
        list_scales = ["KLEP840101"]
        feat_comp = sf.get_features(split_kws=sf.get_split_kws(strategy="compositional"), list_scales=list_scales)
        feat_pos = sf.get_features(split_kws=sf.get_split_kws(strategy="positional"), list_scales=list_scales)
        feat_all = sf.get_features(split_kws=sf.get_split_kws(), list_scales=list_scales)
        assert len(feat_comp) > 0 and len(feat_pos) > 0
        assert set(feat_comp).isdisjoint(feat_pos)
        assert set(feat_comp) | set(feat_pos) == set(feat_all)
        assert all("Segment(1,1)" in f for f in feat_comp)

    def test_repeated_calls_independent(self):
        """Preset results are fresh dicts (mutating one does not leak into the next call)."""
        sf = aa.SequenceFeature()
        first = sf.get_split_kws(strategy="positional")
        first["Pattern"]["steps"].append(99)
        first["Segment"]["n_split_max"] = 3
        assert sf.get_split_kws(strategy="positional") == sf.get_split_kws(
            split_types=["Segment", "Pattern", "PeriodicPattern"], n_split_min=2, n_split_max=15)

    # Negative complex cases
    def test_invalid_strategy_with_all_split_args(self):
        """Any conflicting split arg raises, even when the others are at their defaults."""
        sf = aa.SequenceFeature()
        for kws in [dict(split_types="Segment", n_split_min=1, n_split_max=1),
                    dict(n_split_min=1, n_split_max=1),
                    dict(split_types=["Segment", "Pattern", "PeriodicPattern"], n_split_min=2)]:
            with pytest.raises(ValueError, match="should be left at its default"):
                sf.get_split_kws(strategy="compositional", **kws)

    def test_invalid_strategy_message_names_strategy(self):
        """The conflict message names both the offending arg and the strategy."""
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError, match=r"'n_split_max' \(10\).*'positional'"):
            sf.get_split_kws(strategy="positional", n_split_max=10)

    def test_invalid_strategy_checked_before_split_args(self):
        """An unknown 'strategy' is reported before the conflict check."""
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError, match="'strategy'"):
            sf.get_split_kws(strategy="unknown", split_types="Segment")

    def test_positional_invalid_pattern_args_still_raise(self):
        """'positional' keeps validating Pattern / PeriodicPattern args."""
        sf = aa.SequenceFeature()
        with pytest.raises(ValueError, match="n_min"):
            sf.get_split_kws(strategy="positional", n_min=5, n_max=4)
        with pytest.raises(ValueError, match="len_max"):
            sf.get_split_kws(strategy="positional", len_max=3, steps_pattern=[4, 5])
        with pytest.raises(ValueError):
            sf.get_split_kws(strategy="positional", steps_periodicpattern=[3, 4, 5])

    def test_invalid_strategy_with_invalid_split_range(self):
        """An out-of-range split arg next to 'strategy' still raises ValueError."""
        sf = aa.SequenceFeature()
        for kws in [dict(n_split_min=0), dict(n_split_max=0), dict(n_split_min=10, n_split_max=5)]:
            with pytest.raises(ValueError):
                sf.get_split_kws(strategy="positional", **kws)


class TestGetSplitKwsGoldenValues:
    """Golden-value and warning regressions for get_split_kws."""

    def test_default_output_unchanged(self):
        """Default get_split_kws() output is frozen (dict-equality regression)."""
        sf = aa.SequenceFeature()
        expected = {
            "Segment": {"n_split_min": 1, "n_split_max": 15},
            "Pattern": {"steps": [3, 4], "n_min": 2, "n_max": 4, "len_max": 15},
            "PeriodicPattern": {"steps": [3, 4]},
        }
        assert sf.get_split_kws() == expected

    def test_empty_pattern_bucket_warns_once(self):
        """A degenerate Pattern config (n_min*min(steps) > len_max) emits exactly one
        UserWarning naming the offending parameters."""
        sf = aa.SequenceFeature()
        # pytest.warns records warnings even though tests/pytest.ini globally ignores
        # "'Pattern' split config" UserWarnings, so the carve-out stays intact.
        with pytest.warns(UserWarning) as record:
            sf.get_split_kws(steps_pattern=[3], n_min=2, len_max=4, split_types="Pattern")
        pattern_warnings = [
            w for w in record
            if issubclass(w.category, UserWarning) and "'Pattern' split config" in str(w.message)
        ]
        assert len(pattern_warnings) == 1
        msg = str(pattern_warnings[0].message)
        assert "len_max" in msg and "n_min" in msg and "steps" in msg

    def test_default_output_unchanged_with_strategy_none(self):
        """strategy=None is byte-identical to omitting it."""
        sf = aa.SequenceFeature()
        assert sf.get_split_kws(strategy=None) == sf.get_split_kws()
        assert repr(sf.get_split_kws(strategy=None)) == repr(sf.get_split_kws())

    def test_compositional_equals_manual_call(self):
        """'compositional' equals get_split_kws(split_types='Segment', n_split_min=1, n_split_max=1)."""
        sf = aa.SequenceFeature()
        result = sf.get_split_kws(strategy="compositional")
        manual = sf.get_split_kws(split_types="Segment", n_split_min=1, n_split_max=1)
        assert result == manual
        assert repr(result) == repr(manual)
        assert result == {"Segment": {"n_split_min": 1, "n_split_max": 1}}

    def test_positional_equals_manual_call(self):
        """'positional' equals the explicit sub-segment + Pattern + PeriodicPattern call."""
        sf = aa.SequenceFeature()
        result = sf.get_split_kws(strategy="positional")
        manual = sf.get_split_kws(split_types=["Segment", "Pattern", "PeriodicPattern"],
                                  n_split_min=2, n_split_max=15)
        assert result == manual
        assert repr(result) == repr(manual)
        assert result == {
            "Segment": {"n_split_min": 2, "n_split_max": 15},
            "Pattern": {"steps": [3, 4], "n_min": 2, "n_max": 4, "len_max": 15},
            "PeriodicPattern": {"steps": [3, 4]},
        }

    def test_preset_feature_counts(self):
        """Hand-computed feature counts per part for one scale: Segment(1,1) is 1 split,
        Segment(2..15) is sum(2..15) = 119 splits."""
        sf = aa.SequenceFeature()
        feat_comp = sf.get_features(list_parts=["tmd"], split_kws=sf.get_split_kws(strategy="compositional"),
                                    list_scales=["KLEP840101"])
        assert feat_comp == ["TMD-Segment(1,1)-KLEP840101"]
        split_kws_seg = {"Segment": sf.get_split_kws(strategy="positional")["Segment"]}
        feat_seg = sf.get_features(list_parts=["tmd"], split_kws=split_kws_seg, list_scales=["KLEP840101"])
        assert len(feat_seg) == sum(range(2, 16))
