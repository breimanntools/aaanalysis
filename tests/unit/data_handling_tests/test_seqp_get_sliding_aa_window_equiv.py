"""Equivalence test for the Batch-6 inlining of ``get_sliding_aa_window`` (#186).

The original called ``get_aa_window`` once per slide position; that helper
re-pads the whole sequence string on every call. The new path inlines a strided
slice (plus the same pad-on-overrun semantics). The list of windows must be
byte-identical for every start / stop / index1 case the loop can produce
(non-negative starts only). The original ``get_aa_window``-per-position loop is
the reference.
"""
import pytest
from hypothesis import given, settings, strategies as st

import aaanalysis as aa
from aaanalysis.data_handling._backend.seq_preproc.get_sliding_aa_window import (
    get_sliding_aa_window,
)
from aaanalysis.data_handling._backend.seq_preproc.get_aa_window import get_aa_window

aa.options["verbose"] = False

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

SEQ = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKAL"
       "PDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMG")


def _ref_loop(seq, slide_start=0, slide_stop=None, window_size=5, gap='-', index1=False):
    """Original: one get_aa_window call (re-padding) per slide position."""
    if slide_stop is None:
        slide_stop = len(seq) - 1
        if index1:
            slide_stop += 1
    n_windows = slide_stop - window_size - slide_start + 1
    return [get_aa_window(seq, pos_start=start, window_size=window_size, gap=gap)
            for start in range(slide_start, slide_start + n_windows + 1)]


class TestSlidingWindowEquivalence:
    """Inlined get_sliding_aa_window == per-position get_aa_window reference."""

    @pytest.mark.parametrize("kw", [
        dict(seq=SEQ, window_size=5),
        dict(seq=SEQ, window_size=9, slide_start=3),
        dict(seq=SEQ, window_size=7, index1=True),
        dict(seq=SEQ, window_size=11, slide_start=2, slide_stop=120),
        dict(seq=SEQ, window_size=1),
        dict(seq=SEQ, window_size=len(SEQ)),
        # slide_stop forced past the end -> exercises the pad-on-overrun branch
        dict(seq=SEQ, window_size=8, slide_stop=len(SEQ) + 30),
        dict(seq="ACDE", window_size=6, slide_stop=20),
        dict(seq="ACDEFGHIK", window_size=4, slide_start=2, slide_stop=15, index1=True),
    ])
    def test_cases_byte_identical(self, kw):
        assert get_sliding_aa_window(**kw) == _ref_loop(**kw)

    @settings(max_examples=60, deadline=None)
    @given(
        seq=st.text(alphabet="ACDEFGHIKLMNPQRSTVWY-", min_size=1, max_size=80),
        window_size=st.integers(min_value=1, max_value=15),
        slide_start=st.integers(min_value=0, max_value=20),
        extra_stop=st.integers(min_value=-5, max_value=40),
        index1=st.booleans(),
    )
    def test_property_byte_identical(self, seq, window_size, slide_start, extra_stop, index1):
        """Across random seq / window / start / stop / index1: identical lists."""
        slide_stop = max(0, len(seq) - 1 + extra_stop)
        got = get_sliding_aa_window(seq=seq, slide_start=slide_start,
                                    slide_stop=slide_stop, window_size=window_size,
                                    index1=index1)
        ref = _ref_loop(seq=seq, slide_start=slide_start, slide_stop=slide_stop,
                        window_size=window_size, index1=index1)
        assert got == ref

    def test_default_slide_stop_no_padding(self):
        """Default stop yields windows that never need gap padding."""
        windows = get_sliding_aa_window(seq=SEQ, window_size=5)
        assert windows == _ref_loop(seq=SEQ, window_size=5)
        assert all("-" not in w for w in windows)
