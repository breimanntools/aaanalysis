"""Thumbnail for Protocol 8 (Classifier): hand-drawn decision-tree schematic.

CPP feature matrix -> "What labels?" -> {labelled -> TreeModel} /
{positives + unlabelled -> dPULearn} -> always compare a baseline.

Pure matplotlib schematic (no data fixtures needed). Rendered as a 7x7 inch
figure at 150 dpi -> 1050x1050 px square.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import aaanalysis as aa

aa.options["verbose"] = False
aa.plot_settings(font_scale=1.0, weight_bold=False)

# Palette (AAanalysis-flavoured)
BLUE = "#4778b6"
GREY = "#8a8a8a"
GREEN = "#5aa469"
RED = "#cf5c5c"
DARK = "#595959"
ARROW = "#9a9a9a"

OUT = "/Users/stephanbreimann/Programming/1Packages/aaanalysis/docs/source/_static/img/thumbs/protocol8.png"


def box(ax, cx, cy, w, h, text, color, fs=18, tcolor="white"):
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.0,rounding_size=1.6",
        linewidth=0, facecolor=color, mutation_aspect=1.0,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", color=tcolor,
            fontsize=fs, fontweight="bold", linespacing=1.25, zorder=5)
    return (cx, cy, w, h)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=22,
        linewidth=2.6, color=ARROW,
        shrinkA=2, shrinkB=2, zorder=1,
    ))


fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Title
ax.text(50, 93.5, "Choosing an interpretable classifier",
        ha="center", va="center", fontsize=21, fontweight="bold",
        color="#222222")

# Boxes ---------------------------------------------------------------
# A: CPP feature matrix (top)
ax_box = box(ax, 50, 80, 62, 13,
             "CPP feature matrix\n(samples × features)", BLUE, fs=18.5)
# B: decision (mid)
b_box = box(ax, 50, 58, 52, 11,
            "What labels do you have?", GREY, fs=17.5)
# C: labelled -> TreeModel (left)
c_box = box(ax, 25, 34.5, 45, 15,
            "Two labelled sets\n→ TreeModel\n(random forest / GBM)",
            GREEN, fs=16)
# D: PU -> dPULearn (right)
d_box = box(ax, 75, 34.5, 45, 15,
            "Positives + unlabelled\n→ dPULearn\n(PU learning)",
            RED, fs=16)
# E: baseline (bottom, wide)
e_box = box(ax, 50, 10, 82, 13,
            "Always compare against a simple baseline\n"
            "(majority class / single best feature)",
            DARK, fs=16)

# Arrows --------------------------------------------------------------
# A -> B
arrow(ax, 50, 80 - 6.5, 50, 58 + 5.5)
# B -> C (left, diagonal)
arrow(ax, 50 - 18, 58 - 5.5, 25 + 10, 34.5 + 7.5)
# B -> D (right, diagonal)
arrow(ax, 50 + 18, 58 - 5.5, 75 - 10, 34.5 + 7.5)
# C -> E
arrow(ax, 25 + 6, 34.5 - 7.5, 50 - 18, 10 + 6.5)
# D -> E
arrow(ax, 75 - 6, 34.5 - 7.5, 50 + 18, 10 + 6.5)

# Edge labels
ax.text(31.5, 48.5, "labelled", ha="center", va="center", fontsize=14.5,
        fontstyle="italic", color="#444444")
ax.text(68.5, 48.5, "PU / small", ha="center", va="center", fontsize=14.5,
        fontstyle="italic", color="#444444")

plt.tight_layout()
fig.savefig(OUT, dpi=150, facecolor="white")
print("saved", OUT)
