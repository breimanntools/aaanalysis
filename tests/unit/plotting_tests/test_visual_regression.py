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


# Realistic, deterministic fixture shared by the baselines: the bundled DOM_GSEC
# feature set (varied scales, categories and positions), NOT a synthetic stress
# fixture. Baselines must look like real output so a human reviewing a diff can
# trust them.
def _df_feat():
    aa.options["verbose"] = False
    return aa.load_features(name="DOM_GSEC").head(60)


def _cpp():
    return aa.CPPPlot(df_scales=aa.load_scales())


# A generous RMS tolerance keeps minor cross-patch antialiasing noise from
# flagging while still catching real layout/content regressions.
_TOL = 20


@pytest.mark.mpl_image_compare(tolerance=_TOL)
def test_feature_map_baseline():
    fig, ax = _cpp().feature_map(_df_feat(), figsize=(8, 8))
    return fig


@pytest.mark.mpl_image_compare(tolerance=_TOL)
def test_heatmap_baseline():
    fig, ax = _cpp().heatmap(_df_feat(), figsize=(8, 8))
    return fig


@pytest.mark.mpl_image_compare(tolerance=_TOL)
def test_profile_baseline():
    fig, ax = _cpp().profile(_df_feat(), figsize=(7, 5))
    return fig
