"""This is a script to test the ``preview_options`` error-message helper in
_utils/check_type.py (exposed via ``ut``).

It renders a bounded, readable preview of the valid values for an error about an
unknown name (e.g. a protein entry looked up against a user DataFrame): the first
``n`` options followed by the total count, so the message stays informative without
dumping a potentially huge list.
"""
import pandas as pd
import pytest

import aaanalysis.utils as ut


class TestPreviewOptions:
    def test_short_list_shown_in_full(self):
        out = ut.preview_options(["a", "b", "c"])
        assert out == "['a', 'b', 'c']"

    def test_exactly_n_shown_in_full(self):
        out = ut.preview_options(["a", "b", "c", "d", "e"], n=5)
        assert "..." not in out
        assert "total" not in out

    def test_long_list_truncated_with_total(self):
        options = [f"P{i:05d}" for i in range(20)]
        out = ut.preview_options(options, n=5)
        assert out.startswith("['P00000', 'P00001', 'P00002', 'P00003', 'P00004', ...]")
        assert out.endswith("(20 total)")
        # Truncation never shows more than n names.
        assert "P00005" not in out

    def test_empty(self):
        assert ut.preview_options([]) == "[]"

    def test_accepts_pandas_series(self):
        out = ut.preview_options(pd.Series(["P1", "P2", "P3"]))
        assert out == "['P1', 'P2', 'P3']"

    def test_accepts_dict_keys(self):
        out = ut.preview_options({"P1": "ACDE", "P2": "MKLW"})
        assert out == "['P1', 'P2']"

    def test_custom_n(self):
        out = ut.preview_options(["a", "b", "c", "d"], n=2)
        assert out == "['a', 'b', ...] (4 total)"


class TestPreviewOptionsInErrors:
    """The helper's whole purpose: enriching data-backed-lookup errors with the
    available options. Smoke-check the user-facing message at a representative site.
    """

    def test_seqmut_lists_available_entries(self):
        from aaanalysis.protein_design._seqmut import check_match_mutations_df_seq

        df_seq = pd.DataFrame({ut.COL_ENTRY: ["P1", "P2"], ut.COL_SEQ: ["ACDE", "MKLW"]})
        mutations = pd.DataFrame(
            {ut.COL_ENTRY: ["PX"], ut.COL_POS: [1], ut.COL_TO_AA: ["A"]}
        )
        with pytest.raises(ValueError, match=r"is not in 'df_seq'.*Available entries"):
            check_match_mutations_df_seq(mutations=mutations, df_seq=df_seq)
