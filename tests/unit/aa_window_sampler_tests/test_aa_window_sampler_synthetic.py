"""This is a script to test the AAWindowSampler().sample_synthetic() method."""
import warnings
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
import aaanalysis as aa
from aaanalysis.seq_analysis._backend.aa_window_sampler._utils import collect_test_windows
from aaanalysis.seq_analysis._backend.aa_window_sampler.sample_synthetic import (
    LIST_SYNTH_GENERATORS, LIST_PRESET_GENERATORS, PRESETS,
    _mix_preset_aa_freq, _preset_aa_freq,
)
import numpy as np
import aaanalysis.utils as ut_module

aa.options["verbose"] = False

settings.register_profile("ci", deadline=400)
settings.load_profile("ci")

SCHEMA_SEGMENTS = ["entry_win", "entry", "sequence", "window", "source_position",
                   "label", "role", "strategy"]
LIST_FREE_GENERATORS = ["uniform", "global_freq"] + LIST_PRESET_GENERATORS
LIST_POS_GENERATORS = ["position_specific", "scrambled"]


# I Helper Functions
def _df_seq_with_pos():
    return pd.DataFrame({
        "entry": ["P1", "P2"],
        "sequence": ["ACDEFGHIKLMNPQRSTVWY" * 2] * 2,
        "pos": [[10, 25], [15]],
    })


# II Test Classes
class TestSampleSynthetic:
    """Test sample_synthetic() of the AAWindowSampler class."""

    # Positive tests
    def test_valid_df_seq(self):
        aaws = aa.AAWindowSampler()
        df_seq = aa.load_dataset(name="DOM_GSEC", n=20)
        df = aaws.sample_synthetic(df_seq=df_seq, n=10, window_size=9,
                                generator="global_freq", seed=0)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == SCHEMA_SEGMENTS

    @settings(max_examples=10, deadline=1500)
    @given(n=some.integers(min_value=1, max_value=50))
    def test_valid_n(self, n):
        aaws = aa.AAWindowSampler()
        df = aaws.sample_synthetic(df_seq=_df_seq_with_pos(), n=n, window_size=5,
                                generator="uniform", seed=0)
        assert len(df) == n

    @settings(max_examples=10, deadline=1500)
    @given(window_size=some.integers(min_value=1, max_value=11))
    def test_valid_window_size(self, window_size):
        aaws = aa.AAWindowSampler()
        df = aaws.sample_synthetic(df_seq=_df_seq_with_pos(), n=10,
                                window_size=window_size, generator="uniform", seed=0)
        assert (df["window"].str.len() == window_size).all()

    @settings(max_examples=6, deadline=1500)
    @given(generator=some.sampled_from(LIST_FREE_GENERATORS))
    def test_valid_generator_free(self, generator):
        aaws = aa.AAWindowSampler()
        df = aaws.sample_synthetic(df_seq=_df_seq_with_pos(), n=10, window_size=5,
                                generator=generator, seed=0)
        assert (df["strategy"] == f"synthetic:{generator}").all()
        assert (df["window"].str.len() == 5).all()

    @settings(max_examples=4, deadline=1500)
    @given(generator=some.sampled_from(LIST_POS_GENERATORS))
    def test_valid_generator_pos_dependent(self, generator):
        aaws = aa.AAWindowSampler()
        df = aaws.sample_synthetic(df_seq=_df_seq_with_pos(), n=10, window_size=5,
                                generator=generator, pos_col="pos", seed=0)
        assert (df["strategy"] == f"synthetic:{generator}").all()

    def test_valid_default_role_is_control(self):
        aaws = aa.AAWindowSampler()
        df = aaws.sample_synthetic(df_seq=_df_seq_with_pos(), n=5, window_size=5,
                                generator="uniform", seed=0)
        assert (df["role"] == "Control").all()

    def test_valid_role_override(self):
        aaws = aa.AAWindowSampler()
        df = aaws.sample_synthetic(df_seq=_df_seq_with_pos(), n=5, window_size=5,
                                generator="uniform", role="custom_decoy", seed=0)
        assert (df["role"] == "custom_decoy").all()

    def test_valid_label_ref(self):
        aaws = aa.AAWindowSampler()
        df = aaws.sample_synthetic(df_seq=_df_seq_with_pos(), n=5, window_size=5,
                                generator="uniform", label_ref=99, seed=0)
        assert (df["label"] == 99).all()

    def test_valid_scrambled_preserves_composition(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        df = aaws.sample_synthetic(df_seq=df_seq, n=15, window_size=9,
                                generator="scrambled", pos_col="pos", seed=0)
        test_windows = collect_test_windows(df_seq, "pos", 9)
        compositions = {tuple(sorted(w)) for w in test_windows}
        for w in df["window"]:
            assert tuple(sorted(w)) in compositions

    @settings(max_examples=10, deadline=1500)
    @given(seed=some.integers(min_value=0, max_value=10000))
    def test_valid_seed_determinism(self, seed):
        aaws = aa.AAWindowSampler()
        df_a = aaws.sample_synthetic(df_seq=_df_seq_with_pos(), n=10, window_size=5,
                                  generator="global_freq", seed=seed)
        df_b = aaws.sample_synthetic(df_seq=_df_seq_with_pos(), n=10, window_size=5,
                                  generator="global_freq", seed=seed)
        pd.testing.assert_frame_equal(df_a, df_b)

    @settings(max_examples=12, deadline=2000)
    @given(generator=some.sampled_from(LIST_PRESET_GENERATORS))
    def test_valid_aaontology_preset_generator(self, generator):
        """Each curated AAontology preset draws canonical AAs and matches the schema."""
        import aaanalysis.utils as ut
        aaws = aa.AAWindowSampler()
        df = aaws.sample_synthetic(df_seq=_df_seq_with_pos(), n=20, window_size=10,
                                generator=generator, seed=0)
        assert (df["strategy"] == f"synthetic:{generator}").all()
        canonical = set(ut.LIST_CANONICAL_AA)
        for w in df["window"]:
            assert set(w).issubset(canonical)

    def test_valid_aa_composition_preset_matches_dayhoff(self):
        """The 'aa_composition' preset uses the Dayhoff 1978 scale (DAYM780101)."""
        scale_id, _, _ = PRESETS["aa_composition"]
        assert scale_id == "DAYM780101"

    def test_valid_max_similarity_to_test_filters_synthetic(self):
        df_seq = pd.DataFrame({
            "entry": ["P1"], "sequence": ["AAAAAAAAAAAAA"], "pos": [[5]],
        })
        aaws = aa.AAWindowSampler(max_similarity_to_test=0.0,
                              filter_iteratively=True, max_sampling_attempts=20)
        df = aaws.sample_synthetic(df_seq=df_seq, n=20, window_size=5, generator="uniform",
                                pos_col="pos", seed=0)
        assert (df["window"] != "AAAAA").all()

    def test_valid_max_similarity_within_ref_dedups(self):
        aaws = aa.AAWindowSampler(max_similarity_within_ref=0.99,
                              filter_iteratively=True, max_sampling_attempts=20)
        df = aaws.sample_synthetic(df_seq=_df_seq_with_pos(), n=20, window_size=4,
                                generator="uniform", seed=0)
        assert df["window"].nunique() == len(df)

    def test_valid_concat_with_other_methods(self):
        aaws = aa.AAWindowSampler()
        a = aaws.sample_same_protein(df_seq=_df_seq_with_pos(), pos_col="pos",
                                  n=3, window_size=9, seed=0)
        b = aaws.sample_synthetic(df_seq=_df_seq_with_pos(), n=5, window_size=9,
                               generator="uniform", seed=0)
        merged = pd.concat([a, b], ignore_index=True)
        assert list(merged.columns) == SCHEMA_SEGMENTS
        assert set(merged["role"]) == {"Negative", "Control"}

    def test_valid_synthetic_entry_win_format(self):
        """Synthetic rows use per-call counter ids: ``synth_{i}``."""
        aaws = aa.AAWindowSampler()
        df = aaws.sample_synthetic(df_seq=_df_seq_with_pos(), n=5, window_size=5,
                                generator="uniform", seed=0)
        assert df["entry_win"].tolist() == [f"synth_{i}" for i in range(5)]

    # Negative tests
    def test_invalid_df_seq(self):
        aaws = aa.AAWindowSampler()
        for invalid in [None, [], dict(), 1]:
            with pytest.raises((ValueError, AttributeError, TypeError)):
                aaws.sample_synthetic(df_seq=invalid, n=5, window_size=5,
                                   generator="uniform", seed=0)

    def test_invalid_n(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        for invalid in [0, -1, None, "5", 1.5]:
            with pytest.raises(ValueError):
                aaws.sample_synthetic(df_seq=df_seq, n=invalid, window_size=5,
                                   generator="uniform", seed=0)

    def test_invalid_window_size(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        for invalid in [0, -1, None, "5", 5.5]:
            with pytest.raises(ValueError):
                aaws.sample_synthetic(df_seq=df_seq, n=5, window_size=invalid,
                                   generator="uniform", seed=0)

    def test_invalid_generator(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        for invalid in ["Uniform", None, 1, "", "neg", "markov", "pwm",
                        "dataset_freq"]:
            with pytest.raises(ValueError):
                aaws.sample_synthetic(df_seq=df_seq, n=5, window_size=5,
                                   generator=invalid, seed=0)

    def test_invalid_pos_col_required(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        for generator in LIST_POS_GENERATORS:
            with pytest.raises(ValueError, match="pos_col"):
                aaws.sample_synthetic(df_seq=df_seq, n=5, window_size=5,
                                   generator=generator, seed=0)

    def test_invalid_no_test_windows(self):
        aaws = aa.AAWindowSampler()
        df_seq = pd.DataFrame({"entry": ["P1"], "sequence": ["ACDEFG"], "pos": [[1]]})
        with pytest.raises(ValueError, match="test windows"):
            aaws.sample_synthetic(df_seq=df_seq, n=5, window_size=9,
                               generator="position_specific", pos_col="pos", seed=0)

    def test_invalid_seed(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        for invalid in [-1, "1", 1.5, []]:
            with pytest.raises(ValueError):
                aaws.sample_synthetic(df_seq=df_seq, n=5, window_size=5,
                                   generator="uniform", seed=invalid)

    # Polymorphic generator — list (multiplicative mix of AAontology presets)
    def test_valid_generator_list_mix(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        df = aaws.sample_synthetic(df_seq=df_seq, n=20, window_size=5,
                                    generator=["aa_composition_mp", "alpha_helix"], seed=0)
        assert len(df) == 20
        assert (df["window"].str.len() == 5).all()
        from aaanalysis.utils import LIST_CANONICAL_AA
        assert set("".join(df["window"])) <= set(LIST_CANONICAL_AA)

    def test_valid_generator_list_strategy_tag_sorted(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        df = aaws.sample_synthetic(df_seq=df_seq, n=3, window_size=4,
                                    generator=["alpha_helix", "aa_composition_mp"], seed=0)
        # Strategy tag is alphabetically sorted, '+'-joined.
        assert (df["strategy"] == "synthetic:mix:aa_composition_mp+alpha_helix").all()

    def test_valid_generator_list_three_components(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        df = aaws.sample_synthetic(df_seq=df_seq, n=10, window_size=4,
                                    generator=["aa_composition_mp", "alpha_helix",
                                          "beta_sheet"], seed=0)
        assert len(df) == 10

    def test_valid_mix_equals_renormalized_product(self):
        # Backend-level: _mix_preset_aa_freq returns the renormalized
        # element-wise product of the component priors. Deterministic, no sampling.
        aa_list = list(ut_module.LIST_CANONICAL_AA)
        comp_a = _preset_aa_freq(PRESETS["alpha_helix"][0], aa_list)
        comp_b = _preset_aa_freq(PRESETS["beta_sheet"][0], aa_list)
        expected = comp_a * comp_b
        expected = expected / expected.sum()
        actual = _mix_preset_aa_freq(["alpha_helix", "beta_sheet"], aa_list)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)
        np.testing.assert_allclose(actual.sum(), 1.0, rtol=1e-12)

    def test_valid_mix_three_components_equals_product(self):
        aa_list = list(ut_module.LIST_CANONICAL_AA)
        names = ["aa_composition_mp", "alpha_helix", "beta_sheet"]
        comps = [_preset_aa_freq(PRESETS[m][0], aa_list) for m in names]
        expected = np.prod(comps, axis=0)
        expected = expected / expected.sum()
        actual = _mix_preset_aa_freq(names, aa_list)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_valid_all_presets_strictly_positive(self):
        # Every curated AAontology preset must yield a strictly positive
        # distribution after sum-normalization. PRESETS is restricted to
        # all-positive raw composition / propensity scales by construction,
        # so a single sum-normalization step suffices and no AA is ever
        # assigned zero mass.
        aa_list = list(ut_module.LIST_CANONICAL_AA)
        for name, (scale_id, _, _) in PRESETS.items():
            probs = _preset_aa_freq(scale_id, aa_list)
            assert (probs > 0).all(), (
                f"Preset {name!r} ({scale_id}) has a zero entry; expected "
                f"strictly positive distribution.")
            np.testing.assert_allclose(probs.sum(), 1.0, rtol=1e-12)

    def test_valid_dayhoff_raw_trp_is_nonzero(self):
        # Raw Dayhoff (the 'aa_composition' preset) is non-zero for every AA.
        # Under min-max normalization, Trp (the least-frequent AA) was zero;
        # raw scales preserve true composition ratios so Trp gets ~1.3% mass.
        aa_list = list(ut_module.LIST_CANONICAL_AA)
        comp_aa = _preset_aa_freq(PRESETS["aa_composition"][0], aa_list)
        assert (comp_aa > 0).all()
        w_index = aa_list.index("W")
        # Dayhoff raw W = 1.3 (sum of raw values is 100), so probability ≈ 0.013.
        assert 0.005 < comp_aa[w_index] < 0.05

    def test_valid_preset_aa_freq_rejects_negative_scale(self):
        # Curated PRESETS are all-positive; passing a scale with negatives
        # (e.g. Nakashima NAKH900104, which IS available in scales_raw but
        # is not exposed as a preset) must raise.
        aa_list = list(ut_module.LIST_CANONICAL_AA)
        with pytest.raises(ValueError, match="negative"):
            _preset_aa_freq("NAKH900104", aa_list)

    def test_valid_list_generator_seed_determinism(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        df1 = aaws.sample_synthetic(df_seq=df_seq, n=20, window_size=5,
                                     generator=["alpha_helix", "beta_sheet"], seed=42)
        df2 = aaws.sample_synthetic(df_seq=df_seq, n=20, window_size=5,
                                     generator=["alpha_helix", "beta_sheet"], seed=42)
        assert df1["window"].tolist() == df2["window"].tolist()

    def test_valid_dict_generator_seed_determinism(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        spec = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
        df1 = aaws.sample_synthetic(df_seq=df_seq, n=20, window_size=8,
                                     generator=spec, seed=42)
        df2 = aaws.sample_synthetic(df_seq=df_seq, n=20, window_size=8,
                                     generator=spec, seed=42)
        assert df1["window"].tolist() == df2["window"].tolist()

    def test_valid_tuple_generator_equivalent_to_list(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        df_list = aaws.sample_synthetic(df_seq=df_seq, n=10, window_size=4,
                                         generator=["alpha_helix", "beta_sheet"], seed=7)
        df_tuple = aaws.sample_synthetic(df_seq=df_seq, n=10, window_size=4,
                                          generator=("alpha_helix", "beta_sheet"), seed=7)
        assert df_list["window"].tolist() == df_tuple["window"].tolist()

    # Polymorphic generator — dict (custom alphabet)
    def test_valid_generator_dict_custom_alphabet(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        # DNA alphabet, uniform.
        df = aaws.sample_synthetic(df_seq=df_seq, n=50, window_size=8,
                                    generator={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
                                    seed=0)
        assert len(df) == 50
        # Every character must be one of the four DNA letters.
        assert set("".join(df["window"])) <= {"A", "C", "G", "T"}

    def test_valid_generator_dict_non_protein_symbol(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        df = aaws.sample_synthetic(df_seq=df_seq, n=10, window_size=4,
                                    generator={"@": 0.5, "$": 0.5}, seed=0)
        assert set("".join(df["window"])) <= {"@", "$"}

    def test_valid_generator_dict_strategy_tag(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        df = aaws.sample_synthetic(df_seq=df_seq, n=3, window_size=3,
                                    generator={"C": 0.5, "A": 0.5}, seed=0)
        # Sorted keys joined by '+', regardless of dict insertion order.
        assert (df["strategy"] == "synthetic:custom:A+C").all()

    def test_valid_generator_dict_sum_within_tolerance(self):
        # Floating-point sum may differ from 1.0 by up to ~1e-7; must still pass.
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        # 0.1 * 10 = 1.0 in math but ~0.9999999... in IEEE-754.
        probs = {chr(ord("A") + i): 0.1 for i in range(10)}
        df = aaws.sample_synthetic(df_seq=df_seq, n=5, window_size=3,
                                    generator=probs, seed=0)
        assert len(df) == 5

    # Negative tests for polymorphic generator
    def test_invalid_generator_list_too_short(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        for invalid in [[], ["alpha_helix"]]:
            with pytest.raises(ValueError, match="at least 2"):
                aaws.sample_synthetic(df_seq=df_seq, n=5, window_size=5,
                                       generator=invalid, seed=0)

    def test_invalid_generator_list_unknown_preset(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        with pytest.raises(ValueError):
            aaws.sample_synthetic(df_seq=df_seq, n=5, window_size=5,
                                   generator=["alpha_helix", "uniform"], seed=0)
        with pytest.raises(ValueError):
            aaws.sample_synthetic(df_seq=df_seq, n=5, window_size=5,
                                   generator=["alpha_helix", "not_a_preset"], seed=0)

    def test_invalid_generator_list_non_str_element(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        for invalid in [["alpha_helix", 1], ["alpha_helix", None],
                        ["alpha_helix", ["beta_sheet"]]]:
            with pytest.raises(ValueError):
                aaws.sample_synthetic(df_seq=df_seq, n=5, window_size=5,
                                       generator=invalid, seed=0)

    def test_invalid_generator_list_duplicate(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        with pytest.raises(ValueError, match="duplicate"):
            aaws.sample_synthetic(df_seq=df_seq, n=5, window_size=5,
                                   generator=["alpha_helix", "alpha_helix"], seed=0)
        with pytest.raises(ValueError, match="duplicate"):
            aaws.sample_synthetic(df_seq=df_seq, n=5, window_size=5,
                                   generator=("alpha_helix", "beta_sheet",
                                         "alpha_helix"), seed=0)

    def test_invalid_generator_dict_non_string_key(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        for invalid in [{1: 0.5, 2: 0.5}, {(0,): 0.5, (1,): 0.5}]:
            with pytest.raises(ValueError, match="single-character"):
                aaws.sample_synthetic(df_seq=df_seq, n=5, window_size=5,
                                       generator=invalid, seed=0)

    def test_invalid_generator_dict_sum_not_one(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        for invalid in [{"A": 0.4, "C": 0.4}, {"A": 0.5, "C": 0.6},
                        {"A": 50.0, "C": 50.0}]:
            with pytest.raises(ValueError, match="sum to 1.0"):
                aaws.sample_synthetic(df_seq=df_seq, n=5, window_size=5,
                                       generator=invalid, seed=0)

    def test_invalid_generator_dict_negative_value(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        with pytest.raises(ValueError, match="finite non-negative"):
            aaws.sample_synthetic(df_seq=df_seq, n=5, window_size=5,
                                   generator={"A": -0.5, "C": 1.5}, seed=0)

    def test_invalid_generator_dict_nan_value(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        with pytest.raises(ValueError, match="finite non-negative"):
            aaws.sample_synthetic(df_seq=df_seq, n=5, window_size=5,
                                   generator={"A": float("nan"), "C": 0.5}, seed=0)

    def test_invalid_generator_dict_inf_value(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        for invalid in [float("inf"), float("-inf")]:
            with pytest.raises(ValueError, match="finite non-negative"):
                aaws.sample_synthetic(df_seq=df_seq, n=5, window_size=5,
                                       generator={"A": invalid, "C": 0.5}, seed=0)

    def test_invalid_generator_dict_multi_char_key(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        with pytest.raises(ValueError, match="single-character"):
            aaws.sample_synthetic(df_seq=df_seq, n=5, window_size=5,
                                   generator={"AA": 0.5, "CC": 0.5}, seed=0)

    def test_invalid_generator_dict_too_short(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        with pytest.raises(ValueError, match="at least 2"):
            aaws.sample_synthetic(df_seq=df_seq, n=5, window_size=5,
                                   generator={"A": 1.0}, seed=0)

    def test_invalid_generator_unknown_type(self):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        for invalid in [42, 3.14, object()]:
            with pytest.raises(ValueError):
                aaws.sample_synthetic(df_seq=df_seq, n=5, window_size=5,
                                       generator=invalid, seed=0)


class TestSampleSyntheticComplex:
    """Test sample_synthetic() with combinations of parameters."""

    @settings(max_examples=10, deadline=2500)
    @given(n=some.integers(min_value=1, max_value=30),
           window_size=some.integers(min_value=1, max_value=9),
           generator=some.sampled_from(LIST_FREE_GENERATORS),
           seed=some.integers(min_value=0, max_value=10000))
    def test_valid_combination_free_generators(self, n, window_size, generator, seed):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = aaws.sample_synthetic(df_seq=df_seq, n=n, window_size=window_size,
                                    generator=generator, seed=seed)
        assert len(df) == n
        assert (df["window"].str.len() == window_size).all()
        assert (df["role"] == "Control").all()

    @settings(max_examples=10, deadline=2500)
    @given(n=some.integers(min_value=1, max_value=30),
           window_size=some.integers(min_value=1, max_value=9),
           generator=some.sampled_from(LIST_POS_GENERATORS),
           seed=some.integers(min_value=0, max_value=10000))
    def test_valid_combination_pos_generators(self, n, window_size, generator, seed):
        aaws = aa.AAWindowSampler()
        df_seq = _df_seq_with_pos()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = aaws.sample_synthetic(df_seq=df_seq, n=n, window_size=window_size,
                                    generator=generator, pos_col="pos", seed=seed)
        assert len(df) == n
        assert (df["window"].str.len() == window_size).all()
