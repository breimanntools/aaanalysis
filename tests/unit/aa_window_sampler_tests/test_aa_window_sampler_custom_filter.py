"""This is a script to test the ``custom_filter`` hook of
:class:`aaanalysis.AAWindowSampler` (constructor-level keep-predicate that
composes in every ``sample_*`` method's filter pipeline)."""
import pandas as pd
import pytest

import aaanalysis as aa

aa.options["verbose"] = False


# I Helper Functions
def _df_seq():
    return pd.DataFrame({
        "entry":    ["P1", "P2", "P3", "P4"],
        "sequence": ["ACDEFGHIKLMNPQRSTVWY" * 2, "MKLVWYTSRQPNMLKIHGFE" * 2,
                     "GGGGSSSSCCCCVVVVLLLL" * 2, "WWWWYYYYFFFFRRRRKKKK" * 2],
        "pos":      [[10], [12], None, None],
    })


def _reconstruct(seq, source_position, window_size):
    half_left = (window_size - 1) // 2
    start = source_position - 1 - half_left
    return seq[start:start + window_size]


# II Test Classes
class TestCustomFilterValidation:
    """Constructor validation of the ``custom_filter`` argument."""

    def test_valid_callable_accepted(self):
        s = aa.AAWindowSampler(custom_filter=lambda w, e, p: True)
        assert s._custom_filter is not None

    def test_none_accepted(self):
        s = aa.AAWindowSampler(custom_filter=None)
        assert s._custom_filter is None

    def test_non_callable_raises(self):
        with pytest.raises(ValueError):
            aa.AAWindowSampler(custom_filter=123)


class TestCustomFilterApplied:
    """The hook drops windows across every sampling strategy."""

    def test_same_protein_drops_windows_with_A(self):
        s = aa.AAWindowSampler(random_state=0,
                               custom_filter=lambda w, e, p: "A" not in w)
        df = s.sample_same_protein(df_seq=_df_seq(), n=20, window_size=9, seed=1)
        assert not df["window"].str.contains("A").any()

    def test_different_protein_drops_windows_with_A(self):
        s = aa.AAWindowSampler(random_state=0,
                               custom_filter=lambda w, e, p: "A" not in w)
        df = s.sample_different_protein(df_seq=_df_seq(), n=20, window_size=9, seed=1)
        assert not df["window"].str.contains("A").any()

    def test_synthetic_drops_windows_with_A(self):
        s = aa.AAWindowSampler(random_state=0,
                               custom_filter=lambda w, e, p: "A" not in w)
        df = s.sample_synthetic(df_seq=_df_seq(), n=20, window_size=9,
                                generator="global_freq", seed=1)
        assert not df["window"].str.contains("A").any()

    def test_motif_matched_respects_custom_filter(self):
        import numpy as np
        pwm = pd.DataFrame(np.ones((9, 20)),
                           columns=list("ACDEFGHIKLMNPQRSTVWY"))
        s = aa.AAWindowSampler(random_state=0,
                               custom_filter=lambda w, e, p: "A" not in w)
        df = s.sample_motif_matched(df_seq=_df_seq(), n=20, window_size=9,
                                    motif_pwm=pwm, motif_score_threshold=0.0, seed=1)
        assert not df["window"].str.contains("A").any()

    def test_reject_all_returns_empty(self):
        s = aa.AAWindowSampler(random_state=0,
                               custom_filter=lambda w, e, p: False)
        df = s.sample_same_protein(df_seq=_df_seq(), n=10, window_size=9, seed=1)
        assert len(df) == 0


class TestCustomFilterContext:
    """The hook receives the correct (window, entry, 1-based source_position)."""

    def test_protein_context_matches_output(self):
        seen = []
        s = aa.AAWindowSampler(random_state=0,
                               custom_filter=lambda w, e, p: seen.append((w, e, p)) or True)
        df = s.sample_same_protein(df_seq=_df_seq(), n=5, window_size=9, seed=2)
        df_seq = _df_seq().set_index("entry")
        for _, row in df.iterrows():
            seq = df_seq.loc[row["entry"], "sequence"]
            assert _reconstruct(seq, row["source_position"], 9) == row["window"]
        # Every emitted row's (window, entry, pos) was offered to the predicate.
        offered = {(w, e, p) for w, e, p in seen}
        for _, row in df.iterrows():
            assert (row["window"], row["entry"], row["source_position"]) in offered

    def test_synthetic_context_is_empty_entry_and_sentinel_pos(self):
        seen = []
        s = aa.AAWindowSampler(random_state=0,
                               custom_filter=lambda w, e, p: seen.append((e, p)) or True)
        s.sample_synthetic(df_seq=_df_seq(), n=5, window_size=9,
                           generator="global_freq", seed=1)
        assert seen, "custom_filter was never called for synthetic"
        assert all(e == "" and p == -1 for e, p in seen)
