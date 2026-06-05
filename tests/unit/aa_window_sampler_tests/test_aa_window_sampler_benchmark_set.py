"""This is a script to test :meth:`aaanalysis.AAWindowSampler.sample_benchmark_set`
(multi-arm orchestration: concat + ``arm`` column, role preservation, per-arm
deterministic sub-seeds, and arm-config validation)."""
import pandas as pd
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut

aa.options["verbose"] = False


# I Helper Functions
def _df_seq():
    return pd.DataFrame({
        "entry":    ["P1", "P2", "P3", "P4"],
        "sequence": ["ACDEFGHIKLMNPQRSTVWY" * 2, "MKLVWYTSRQPNMLKIHGFE" * 2,
                     "GGGGSSSSCCCCVVVVLLLL" * 2, "WWWWYYYYFFFFRRRRKKKK" * 2],
        "pos":      [[10], [12], None, None],
    })


def _three_arms():
    return {
        "neg":  {"method": "same_protein", "n": 10, "window_size": 9},
        "unl":  {"method": "different_protein", "n": 10, "window_size": 9},
        "ctrl": {"method": "synthetic", "n": 10, "window_size": 9,
                 "generator": "global_freq"},
    }


# II Test Classes
class TestBenchmarkSetSchema:
    """Output schema: segments columns + arm column, role/strategy preserved."""

    def test_arm_column_present_and_named(self):
        s = aa.AAWindowSampler(random_state=0)
        df = s.sample_benchmark_set(df_seq=_df_seq(), arms=_three_arms(), seed=42)
        assert ut.COL_ARM in df.columns
        assert set(df[ut.COL_ARM]) == {"neg", "unl", "ctrl"}

    def test_segments_schema_preserved(self):
        s = aa.AAWindowSampler(random_state=0)
        df = s.sample_benchmark_set(df_seq=_df_seq(), arms=_three_arms(), seed=42)
        for col in ut.COLS_SEGMENTS:
            assert col in df.columns

    def test_role_tagging_per_arm(self):
        s = aa.AAWindowSampler(random_state=0)
        df = s.sample_benchmark_set(df_seq=_df_seq(), arms=_three_arms(), seed=42)
        assert set(df.loc[df[ut.COL_ARM] == "neg", ut.COL_ROLE]) == {ut.ROLE_NEG}
        assert set(df.loc[df[ut.COL_ARM] == "unl", ut.COL_ROLE]) == {ut.ROLE_UNL}
        assert set(df.loc[df[ut.COL_ARM] == "ctrl", ut.COL_ROLE]) == {ut.ROLE_CTRL}

    def test_no_auto_dedup_preserves_all_rows(self):
        # Two identical same_protein arms => rows duplicated across arms (kept).
        s = aa.AAWindowSampler(random_state=0)
        arms = {
            "a": {"method": "same_protein", "n": 5, "window_size": 9},
            "b": {"method": "same_protein", "n": 5, "window_size": 9},
        }
        df = s.sample_benchmark_set(df_seq=_df_seq(), arms=arms, seed=1)
        assert (df[ut.COL_ARM] == "a").sum() + (df[ut.COL_ARM] == "b").sum() == len(df)

    def test_motif_arm_adds_motif_score_column(self):
        import numpy as np
        pwm = pd.DataFrame(np.ones((9, 20)), columns=list("ACDEFGHIKLMNPQRSTVWY"))
        s = aa.AAWindowSampler(random_state=0)
        arms = {
            "neg":   {"method": "same_protein", "n": 5, "window_size": 9},
            "decoy": {"method": "motif_matched", "n": 5, "window_size": 9,
                      "motif_pwm": pwm, "motif_score_threshold": 0.0},
        }
        df = s.sample_benchmark_set(df_seq=_df_seq(), arms=arms, seed=1)
        assert "motif_score" in df.columns
        # NaN for the non-motif arm, populated for the motif arm.
        assert df.loc[df[ut.COL_ARM] == "neg", "motif_score"].isna().all()
        assert df.loc[df[ut.COL_ARM] == "decoy", "motif_score"].notna().all()


class TestBenchmarkSetReproducibility:
    """Identical seeds reproduce identical sets; different seeds differ."""

    def test_same_seed_identical(self):
        s = aa.AAWindowSampler(random_state=0)
        df1 = s.sample_benchmark_set(df_seq=_df_seq(), arms=_three_arms(), seed=42)
        df2 = s.sample_benchmark_set(df_seq=_df_seq(), arms=_three_arms(), seed=42)
        assert df1.equals(df2)

    def test_different_seed_differs(self):
        s = aa.AAWindowSampler(random_state=0)
        df1 = s.sample_benchmark_set(df_seq=_df_seq(), arms=_three_arms(), seed=42)
        df2 = s.sample_benchmark_set(df_seq=_df_seq(), arms=_three_arms(), seed=7)
        assert not df1["window"].equals(df2["window"])

    def test_seed_falls_back_to_random_state(self):
        s1 = aa.AAWindowSampler(random_state=123)
        s2 = aa.AAWindowSampler(random_state=123)
        df1 = s1.sample_benchmark_set(df_seq=_df_seq(), arms=_three_arms())
        df2 = s2.sample_benchmark_set(df_seq=_df_seq(), arms=_three_arms())
        assert df1.equals(df2)


class TestBenchmarkSetValidation:
    """Arm-config validation raises ValueError on malformed input."""

    def test_empty_arms_raises(self):
        s = aa.AAWindowSampler(random_state=0)
        with pytest.raises(ValueError):
            s.sample_benchmark_set(df_seq=_df_seq(), arms={})

    def test_unknown_method_raises(self):
        s = aa.AAWindowSampler(random_state=0)
        with pytest.raises(ValueError):
            s.sample_benchmark_set(df_seq=_df_seq(), arms={"x": {"method": "nope"}})

    def test_missing_method_key_raises(self):
        s = aa.AAWindowSampler(random_state=0)
        with pytest.raises(ValueError):
            s.sample_benchmark_set(df_seq=_df_seq(), arms={"x": {"n": 5}})

    def test_reserved_key_raises(self):
        s = aa.AAWindowSampler(random_state=0)
        with pytest.raises(ValueError):
            s.sample_benchmark_set(
                df_seq=_df_seq(),
                arms={"x": {"method": "synthetic", "seed": 1}})

    def test_non_dict_arm_raises(self):
        s = aa.AAWindowSampler(random_state=0)
        with pytest.raises(ValueError):
            s.sample_benchmark_set(df_seq=_df_seq(), arms={"x": ["same_protein"]})
