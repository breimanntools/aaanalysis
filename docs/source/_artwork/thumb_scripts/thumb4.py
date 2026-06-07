"""Thumbnail for Protocol 4: prediction levels (residue / domain / protein).

Hand-drawn matplotlib schematic. Three rows, each showing the unit of
comparison as a coloured bar with a level label and a short italic caption.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import aaanalysis as aa

aa.options["verbose"] = False
aa.plot_settings(font_scale=1.2, weight_bold=False)

# Colours (red / blue / green family matching AAanalysis levels)
RED, RED_L = "#c0504d", "#e9cccb"
BLUE, BLUE_L = "#4a6fad", "#c5d2e8"
GREEN = "#5aa469"

OUT = "/Users/stephanbreimann/Programming/1Packages/aaanalysis/docs/source/_static/img/thumbs/protocol4.png"


def bar(ax, x0, y0, w, h, fc, ec="none"):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0,rounding_size=0.012",
        fc=fc, ec=ec, lw=0, mutation_aspect=1.0, zorder=2))


fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Title
ax.text(0.5, 0.975, "Prediction levels:\nthe unit of comparison",
        ha="center", va="top", fontsize=22, fontweight="bold")

# Row geometry (evenly balanced under the title)
bar_x0, bar_w, bar_h = 0.30, 0.62, 0.10
label_x = 0.06
row_y = [0.585, 0.345, 0.105]   # bottom-y of each bar

rows = [
    dict(label="Residue", tag="(AA_*)", color=RED,
         caption="compare a window around each residue",
         segs=[(0.00, 0.40, RED_L), (0.40, 0.20, RED), (0.60, 0.40, RED_L)]),
    dict(label="Domain", tag="(DOM_*)", color=BLUE,
         caption="compare a domain (e.g. TMD + flanks)",
         segs=[(0.00, 0.25, BLUE_L), (0.25, 0.50, BLUE), (0.75, 0.25, BLUE_L)]),
    dict(label="Protein", tag="(SEQ_*)", color=GREEN,
         caption="compare the whole chain",
         segs=[(0.00, 1.00, GREEN)]),
]

for y0, r in zip(row_y, rows):
    yc = y0 + bar_h / 2
    # Level label + tag
    ax.text(label_x, yc + 0.012, r["label"], ha="left", va="center",
            fontsize=19, fontweight="bold", color=r["color"])
    ax.text(label_x, yc - 0.045, r["tag"], ha="left", va="center",
            fontsize=14.5, color=r["color"], alpha=0.85)
    # Bar segments
    for frac0, fracw, fc in r["segs"]:
        bar(ax, bar_x0 + frac0 * bar_w, y0, fracw * bar_w, bar_h, fc)
    # Caption
    ax.text(bar_x0 + bar_w / 2, y0 - 0.058, r["caption"],
            ha="center", va="center", fontsize=15, style="italic",
            color="#333333")

fig.set_size_inches(7, 7)
plt.tight_layout()
fig.savefig(OUT, dpi=150, facecolor="white")
print("saved", OUT)
