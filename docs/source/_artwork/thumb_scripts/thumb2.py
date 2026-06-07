"""Gallery thumbnail for Protocol 2: sequence logo (AAlogo / AAlogoPlot).

Classic information-content (bits) sequence logo: stacked amino-acid letters
sized by per-position conservation, coloured by property, with JMD-N / TMD /
JMD-C parts annotated. A compact window + a small pooled set keep the dominant
letters tall and legible at thumbnail size (no dense rainbow blob).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import aaanalysis as aa

aa.options["verbose"] = False

# Large, readable fonts; thin (non-bold) weight for a clean look.
aa.plot_settings(font_scale=1.6, weight_bold=False)

# Small, deterministic fixture (no RNG in AAlogo: pooled tally over sequences).
# A modest pooled set keeps dominant residues tall instead of a flat rainbow.
df_seq = aa.load_dataset(name="DOM_GSEC", n=8)

# Compact window: a short flank on each side + a legible TMD core.
JMD_N_LEN, JMD_C_LEN, TMD_LEN = 3, 3, 10
sf = aa.SequenceFeature()
df_parts = sf.get_df_parts(
    df_seq=df_seq,
    list_parts=["jmd_n", "tmd", "jmd_c"],
    jmd_n_len=JMD_N_LEN,
    jmd_c_len=JMD_C_LEN,
)

# Information-content logo (bits): height = conservation -> classic sparse logo.
al = aa.AAlogo(logo_type="information")
df_logo = al.get_df_logo(df_parts=df_parts, tmd_len=TMD_LEN)

alp = aa.AAlogoPlot(
    logo_type="information",
    jmd_n_len=JMD_N_LEN,
    jmd_c_len=JMD_C_LEN,
    verbose=False,
)
fig, ax = alp.single_logo(
    df_logo=df_logo,
    figsize=(7, 7),
    fontsize_tmd_jmd=18,
    weight_tmd_jmd="bold",
    highlight_alpha=0.12,
    logo_width=0.82,          # narrower glyphs -> no edge clipping of tall letters
    logo_vpad=0.06,           # a little vertical breathing room between letters
)

# Drop the numeric position xticks (1/3/13/16): at thumbnail size they collide
# with the JMD-N / TMD / JMD-C part labels. The colored part bar at the baseline
# already conveys the three regions, so the logo reads cleanly without numbers.
ax.set_xticks([])

# Headroom so the tallest stacks never clip the top frame.
y0, y1 = ax.get_ylim()
ax.set_ylim(y0, y1 * 1.10)

fig.set_size_inches(7, 7)
plt.tight_layout()
fig.savefig(
    "/Users/stephanbreimann/Programming/1Packages/aaanalysis/docs/source/_static/img/thumbs/protocol2.png",
    dpi=150,
    facecolor="white",
)
print("saved")
