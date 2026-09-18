"""This is a script to test that a CPP plot owns the figure it draws into.

A plot method that is not handed an ``ax`` must create its own figure. It used to
reuse whatever figure happened to be open, which is invisible interactively --
``plt.show()`` closes the figure, so none is ever open when the next cell runs --
and wrong everywhere else. Headless (``MPLBACKEND=Agg``, the setting CI and any
script use) ``plt.show()` is a no-op, so the second plot was drawn on top of the
first, and a sequence plot then measured the *previous* plot's tick labels and
raised ``ValueError: not enough values to unpack``.

The tests run on the Agg backend that ``conftest.py`` pins, which is exactly the
condition that used to fail.
"""
import matplotlib.pyplot as plt
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.feature_engineering._backend.cpp._utils_cpp_plot_elements import PlotElements


@pytest.fixture(autouse=True)
def _close_figures():
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def df_feat():
    return aa.load_features().head(30)


# I set_figsize() -- the seam the whole contract rests on
class TestSetFigsize:
    """Test that the helper returns an axes rather than None."""

    def test_creates_an_axes_when_none_is_passed(self):
        fig, ax = PlotElements.set_figsize(ax=None, figsize=(4, 3))
        assert ax is not None
        assert ax.figure is fig

    def test_keeps_a_passed_axes_and_its_figure(self):
        fig_in, ax_in = plt.subplots(figsize=(4, 3))
        fig, ax = PlotElements.set_figsize(ax=ax_in, figsize=(6, 4))
        assert ax is ax_in
        assert fig is fig_in

    def test_an_open_figure_is_not_reused(self):
        """The regression: an unrelated open figure must not capture the plot."""
        fig_open, _ = plt.subplots(figsize=(4, 3))
        fig, ax = PlotElements.set_figsize(ax=None, figsize=(4, 3))
        assert fig is not fig_open
        assert ax is not None

    def test_a_second_call_gets_its_own_figure(self):
        fig_a, ax_a = PlotElements.set_figsize(ax=None, figsize=(4, 3))
        fig_b, ax_b = PlotElements.set_figsize(ax=None, figsize=(4, 3))
        assert fig_a is not fig_b
        assert ax_a is not ax_b

    def test_figsize_is_applied_to_the_new_figure(self):
        fig, _ = PlotElements.set_figsize(ax=None, figsize=(5, 3), force_set=True)
        assert tuple(fig.get_size_inches()) == (5, 3)


# II The public methods, called back to back without closing
class TestCPPPlotFigureOwnership:
    """Test the user-visible consequence on CPPPlot."""

    def test_two_profiles_in_a_row_get_two_figures(self, df_feat):
        cpp_plot = aa.CPPPlot(accept_gaps=True)
        fig_a, _ = cpp_plot.profile(df_feat=df_feat)
        fig_b, _ = cpp_plot.profile(df_feat=df_feat)
        assert fig_a is not fig_b

    def test_profile_after_another_plot_does_not_reuse_its_figure(self, df_feat):
        """KPI: the protocol-9 failure -- a ranking plot, then a sequence profile."""
        cpp_plot = aa.CPPPlot(accept_gaps=True)
        fig_rank, _ = cpp_plot.ranking(df_feat=df_feat)
        plt.show()                       # a no-op headless; the figure stays open
        fig_prof, ax_prof = cpp_plot.profile(df_feat=df_feat)
        assert fig_prof is not fig_rank
        assert ax_prof.figure is fig_prof

    def test_sequence_profile_after_another_plot_renders(self, df_feat):
        """The exact crash: the residue letters are sized from the axes' tick labels.

        With a foreign axes there were no tick labels to measure, which raised.
        """
        tmd_seq = "A" * 20
        jmd_seq = "A" * 10
        cpp_plot = aa.CPPPlot(accept_gaps=True)
        cpp_plot.ranking(df_feat=df_feat)
        plt.show()
        fig, ax = cpp_plot.profile(df_feat=df_feat, tmd_seq=tmd_seq,
                                   jmd_n_seq=jmd_seq, jmd_c_seq=jmd_seq)
        assert len(ax.xaxis.get_ticklabels(which="both")) > 0
        assert ax.figure is fig

    def test_a_passed_ax_is_still_honoured(self, df_feat):
        """Composition must keep working: a caller-supplied axes is drawn into."""
        fig_in, ax_in = plt.subplots(figsize=(6, 4))
        cpp_plot = aa.CPPPlot(accept_gaps=True)
        fig, ax = cpp_plot.profile(df_feat=df_feat, ax=ax_in)
        assert ax is ax_in
        assert fig is fig_in

    def test_returned_pair_follows_the_plot_contract(self, df_feat):
        cpp_plot = aa.CPPPlot(accept_gaps=True)
        out = cpp_plot.profile(df_feat=df_feat)
        assert isinstance(out, ut.FigAxResult)
        fig, ax = out
        assert ax.figure is fig
