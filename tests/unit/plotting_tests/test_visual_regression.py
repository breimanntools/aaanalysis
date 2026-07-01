"""
Visual-regression baselines for the signature dense CPP composites, using
pytest-mpl (already a dev dependency). These catch *silent* visual degradation
that the "does it return an Axes" tests cannot — a complement to the bbox
overlap gate in ``test_dense_label_overlap.py``.

Opt-in by design: pytest-mpl only *compares* against the committed baseline
images when ``--mpl`` is passed, so the blocking CI matrix (which does not pass
``--mpl``) just runs the plot functions. Run the comparison locally with::

    pytest tests/unit/plotting_tests/test_visual_regression.py --mpl

Regenerate the baselines (only when an intended visual change lands) with::

    pytest tests/unit/plotting_tests/test_visual_regression.py \
        --mpl-generate-path=tests/unit/plotting_tests/baseline

Baselines are environment-sensitive (matplotlib/freetype versions), which is why
the comparison is not wired into the blocking matrix; treat a local ``--mpl``
diff as a prompt to eyeball the figure, not an automatic failure.
"""
import matplotlib
matplotlib.use("Agg")
import pytest

import aaanalysis as aa

from ._text_overlap import make_dense_df_feat


# Deterministic, moderately dense fixture shared by the baselines.
def _df_feat():
    aa.options["verbose"] = False
    return make_dense_df_feat(20)


# A generous RMS tolerance keeps minor cross-patch antialiasing noise from
# flagging while still catching real layout/content regressions.
_TOL = 20


@pytest.mark.mpl_image_compare(tolerance=_TOL)
def test_feature_map_baseline():
    fig, ax = aa.CPPPlot().feature_map(_df_feat())
    return fig


@pytest.mark.mpl_image_compare(tolerance=_TOL)
def test_heatmap_baseline():
    fig, ax = aa.CPPPlot().heatmap(_df_feat())
    return fig


@pytest.mark.mpl_image_compare(tolerance=_TOL)
def test_profile_baseline():
    fig, ax = aa.CPPPlot().profile(_df_feat())
    return fig
