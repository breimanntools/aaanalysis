"""Thumbnail for Protocol 3 (Sampling).

P3 schematic: windows on substrate proteins -- test windows (label=1) at P1
anchors vs same-protein hard negatives, drawn as broken_barh lanes along a
few substrate proteins.

Run:
  COVERAGE_CORE=sysmon python3 docs/source/_artwork/thumb_scripts/thumb3.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import aaanalysis as aa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _thumb_utils import save_square  # noqa: E402

# Write into the docs tree of whichever checkout this script lives in (repo root
# or a git worktree), so a worktree render never clobbers the main checkout's
# tracked thumbnail. thumb3.py sits at docs/source/_artwork/thumb_scripts/, so
# parents[2] is docs/source/.
OUT = str(Path(__file__).resolve().parents[2] / "_static" / "img" / "thumbs" / "protocol3.png")

aa.options["verbose"] = False
aa.options["random_state"] = 42

# --- Minimal caspase-style fixture (same as the protocol notebook) -----------
df_seq = pd.DataFrame({
    "entry": ["SUB1", "SUB2", "SUB3"],
    "sequence": [
        "MKTAYIAKQRDEVDSGLAPYKVLNMQATGHIWDETDSARKLMNPQRSTV",
        "GASPMNLKDEVDTAGRWFYHCIKLMNPQRSTVWYDQTDSGKLAANPQRS",
        "MQALDEVDGKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYAKLM",
    ],
    "pos": [[14, 36], [9, 35], [9]],
})

window_size = 8
half_left = (window_size - 1) // 2     # 3

sampler = aa.AAWindowSampler(random_state=42, verbose=False)
df_same = sampler.sample_same_protein(
    df_seq=df_seq, pos_col="pos", n=6, window_size=window_size,
    min_distance_to_pos=window_size, role="Negative", seed=42,
)

# --- Plot --------------------------------------------------------------------
aa.plot_settings(font_scale=1.5, weight_bold=False)
c_neg, c_unl, c_test, c_ctrl = aa.plot_get_clist(n_colors=4)

fig, ax = plt.subplots(figsize=(7, 7))
substrates = ["SUB1", "SUB2", "SUB3"]
bar_h = 0.5
backbone_len = 50  # uniform schematic backbone so windows never overflow


def _clamp(start):
    return min(max(start, 1), backbone_len - window_size + 1)


for i, entry in enumerate(substrates):
    row = df_seq[df_seq["entry"] == entry].iloc[0]
    # uniform protein backbone
    ax.broken_barh([(1, backbone_len)], (i - 0.11, 0.22),
                   color="0.88", zorder=1)
    for s in df_same.loc[df_same["entry"] == entry, "source_position"]:
        ax.broken_barh([(_clamp(s - half_left), window_size)],
                       (i - bar_h / 2, bar_h),
                       color=c_neg, edgecolor="white", linewidth=1.5, zorder=2)
    for p in row["pos"]:
        ax.broken_barh([(_clamp(p - half_left), window_size)],
                       (i - bar_h / 2, bar_h),
                       color=c_test, edgecolor="white", linewidth=1.5, zorder=3)

ax.set_yticks(range(len(substrates)))
ax.set_yticklabels(substrates)
ax.set_ylim(-0.6, len(substrates) - 0.4)
ax.set_xlim(0, backbone_len + 2)
ax.set_xlabel("Residue position")
ax.set_title("Windows on substrate proteins", weight="bold", pad=14)
sns.despine(ax=ax)

aa.plot_legend(
    ax=ax,
    dict_color={"Test window (label=1)": c_test, "Same-protein negative": c_neg},
    n_cols=1, loc="upper center", x=0.5, y=-0.16,
    fontsize=15, handlelength=1.3,
)

plt.tight_layout()
fig.subplots_adjust(bottom=0.26, top=0.91)
save_square(OUT)
