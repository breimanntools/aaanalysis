"""
Pixel-comparison (visual-regression) tests for the signature CPP plots, using
pytest-mpl (already a dev dependency).

Each test renders a plot on a realistic, deterministic fixture and pytest-mpl
compares the pixels against a committed baseline image (within an RMS tolerance),
so any future change to the rendered figure is caught.

Rendered the way plots are actually consumed — ``aa.plot_settings()`` (the house
rcParams every notebook/publication figure uses) and ``bbox_inches="tight"`` — so
the baselines look like real output, not a raw-canvas artifact.

Run the comparison
------------------
    pytest tests/unit/plotting_tests/test_visual_regression.py --mpl

Regenerate the baselines (only when an intended visual change lands)
-------------------------------------------------------------------
    pytest tests/unit/plotting_tests/test_visual_regression.py \
        --mpl-generate-path=tests/unit/plotting_tests/baseline

Pixel comparison is matplotlib/freetype-version sensitive, so it is opt-in
(``--mpl``) rather than wired into the blocking CI matrix; run it in a pinned
environment (or locally) and treat a diff as a prompt to eyeball the figure. The
environment-independent guarantees (no label overlap, ticks on the bottom, font
>= 5pt) live in ``test_dense_label_overlap.py``.
"""
import matplotlib
matplotlib.use("Agg")
import pytest

import aaanalysis as aa


# RMS tolerance: absorbs sub-pixel antialiasing noise, still catches real layout or
# content changes.
_TOL = 15
# Save as plots are consumed: tight bbox (nothing clipped), fixed dpi for stable size.
_SAVE = {"bbox_inches": "tight", "dpi": 130}


def _cpp():
    # Render like real usage: aa.plot_settings() applies the house rcParams, exactly
    # as every notebook/publication figure does. Without it the harness's raw
    # matplotlib defaults over-tick the colorbar and clip bar labels.
    aa.options["verbose"] = False
    aa.plot_settings()
    return aa.CPPPlot(df_scales=aa.load_scales())


def _df_feat():
    # Realistic, deterministic bundled feature set (varied scales/categories/positions).
    return aa.load_features(name="DOM_GSEC").head(40)


@pytest.mark.mpl_image_compare(tolerance=_TOL, savefig_kwargs=_SAVE)
def test_feature_map_baseline():
    fig, ax = _cpp().feature_map(_df_feat(), figsize=(8, 8))
    return fig


@pytest.mark.mpl_image_compare(tolerance=_TOL, savefig_kwargs=_SAVE)
def test_heatmap_baseline():
    fig, ax = _cpp().heatmap(_df_feat(), figsize=(8, 8))
    return fig


@pytest.mark.mpl_image_compare(tolerance=_TOL, savefig_kwargs=_SAVE)
def test_profile_baseline():
    fig, ax = _cpp().profile(_df_feat(), figsize=(7, 5))
    return fig
