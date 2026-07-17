"""Gallery thumbnail for Protocol 2: sequence logo (AALogo / AALogoPlot).

Classic two-panel sequence logo: a per-position information-content (bits) bar on
top + the composition letter stack below, coloured by amino-acid property, with
the JMD-N / TMD / JMD-C parts annotated. Standard part sizes (jmd_n_len =
jmd_c_len = 5).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import aaanalysis as aa

OUT = "/Users/stephanbreimann/Programming/1Packages/aaanalysis/docs/source/_static/img/thumbs/protocol2.png"

aa.options["verbose"] = False
# Settings mirror the gamma-secretase use case (font_scale=0.85, 9x3 per logo). A logo
# reads wide and short: squeezing 25 positions into a square stretches the letters and
# leaves the position ticks no room beside the part labels.
aa.plot_settings(font_scale=0.85)

# Pooled (no-label) set; a fuller set gives a more meaningful conservation signal.
df_seq = aa.load_dataset(name="DOM_GSEC", n=50)

# Standard part sizes: 5-residue JMD flanks around the TMD core.
JMD_N_LEN, JMD_C_LEN, TMD_LEN = 5, 5, 15
sf = aa.SequenceFeature()
df_parts = sf.get_df_parts(
    df_seq=df_seq,
    list_parts=["jmd_n", "tmd", "jmd_c"],
    jmd_n_len=JMD_N_LEN,
    jmd_c_len=JMD_C_LEN,
)

# Composition letter stack + the per-position information content (bits).
al = aa.AALogo()
df_logo = al.get_df_logo(df_parts=df_parts, tmd_len=TMD_LEN)
df_logo_info = al.get_df_logo_info(df_parts=df_parts, tmd_len=TMD_LEN)

alp = aa.AALogoPlot(jmd_n_len=JMD_N_LEN, jmd_c_len=JMD_C_LEN, verbose=False)
# Passing df_logo_info renders the bits/conservation bar ON TOP of the logo.
# fontsize_tmd_jmd is left at its default so the JMD-N/TMD/JMD-C labels stay directly
# under the part bar at a size that clears the position ticks.
fig, ax = alp.single_logo(
    df_logo=df_logo,
    df_logo_info=df_logo_info,
    figsize=(9, 3),
)

fig = plt.gcf()
fig.set_size_inches(9, 3)
# NOTE: no extra plt.tight_layout() here — single_logo already lays out the panels
# and sets subplots_adjust(hspace=0); a second tight_layout re-opens the bits/logo gap.
fig.savefig(OUT, dpi=150, facecolor="white")
print("saved", OUT)
