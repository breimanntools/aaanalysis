"""Gallery thumbnail for Protocol 2: multi-logo sampling-strategy comparison.

Three stacked sequence logos (AALogo / AALogoPlot.multi_logo), one per window
sampling strategy drawn with AAWindowSampler on DOM_GSEC:

* Same-protein: windows from the substrate proteins that carry a positive site.
* Different-protein: windows from the non-substrate proteins (no positive site).
* Synthetic: uniform per-residue control windows (background baseline).

Each strategy's fixed-length windows are pooled into a single aligned block and
profiled with AALogo (composition letter stack + per-position bits bar), so the
three logos are directly comparable position by position. This is the same
figure the protocol notebook shows, saved as a compact ~square tile.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import aaanalysis as aa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _thumb_utils import save_square  # noqa: E402

# Write into the docs tree of whichever checkout this script lives in (repo root
# or a git worktree), so a worktree render never clobbers the main checkout's
# tracked thumbnail. thumb2.py sits at docs/source/_artwork/thumb_scripts/, so
# parents[2] is docs/source/.
OUT = str(Path(__file__).resolve().parents[2] / "_static" / "img" / "thumbs" / "protocol2.png")

# Reproducibility knobs, shared with the protocol notebook so the thumbnail
# matches the figure the reader builds there.
WINDOW_SIZE, N_WINDOWS, SEED = 15, 120, 42

aa.options["verbose"] = False

# DOM_GSEC pool (n per class -> 2n proteins). Mark each substrate's TMD centre as a
# positive site and leave the non-substrates unlabeled: sample_same_protein then
# draws from the substrates (proteins that carry a positive), while
# sample_different_protein draws from the non-substrates (proteins with none).
df_seq = aa.load_dataset(name="DOM_GSEC", n=25).copy()
centre = ((df_seq["tmd_start"] + df_seq["tmd_stop"]) // 2).astype(int)
df_seq["pos"] = [[int(c)] if lbl == 1 else [] for c, lbl in zip(centre, df_seq["label"])]

# One df_logo per strategy, all windows the same length so the logos are position-aligned.
aaws = aa.AAWindowSampler(random_state=SEED)
df_same = aaws.sample_same_protein(df_seq=df_seq, n=N_WINDOWS, window_size=WINDOW_SIZE,
                                   pos_col="pos", seed=SEED)
df_diff = aaws.sample_different_protein(df_seq=df_seq, n=N_WINDOWS, window_size=WINDOW_SIZE,
                                        pos_col="pos", seed=SEED)
df_synth = aaws.sample_synthetic(df_seq=df_seq, n=N_WINDOWS, window_size=WINDOW_SIZE,
                                 generator="uniform", seed=SEED)

# The sampled windows are equal-length segments (not TMD/JMD parts), so pool each
# strategy's windows into a single aligned block and profile it with AALogo.
al = aa.AALogo(logo_type="probability")
list_df_logo, list_df_logo_info = [], []
for df in (df_same, df_diff, df_synth):
    df_parts = pd.DataFrame({"tmd": df["window"].to_list()})
    list_df_logo.append(al.get_df_logo(df_parts=df_parts, tmd_len=WINDOW_SIZE))
    list_df_logo_info.append(al.get_df_logo_info(df_parts=df_parts, tmd_len=WINDOW_SIZE))

aa.plot_settings(font_scale=0.85)
# jmd_n_len = jmd_c_len = 0: the whole pooled window is one aligned block (no JMD flanks).
alp = aa.AALogoPlot(logo_type="probability", jmd_n_len=0, jmd_c_len=0, verbose=False)
fig, ax = alp.multi_logo(
    list_df_logo=list_df_logo,
    list_df_logo_info=list_df_logo_info,
    list_name_data=["Same-protein", "Different-protein", "Synthetic"],
    name_data_pos="left",
    figsize_per_logo=(9, 3),
)

# save_square tight-crops (so the left-hand strategy names, name_data_pos="left",
# are never clipped) then centers the block on a uniform white square tile.
save_square(OUT)
