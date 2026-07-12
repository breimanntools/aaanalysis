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
aa.plot_settings(font_scale=1.25, weight_bold=False)

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
fig, ax = alp.single_logo(
    df_logo=df_logo,
    df_logo_info=df_logo_info,
    figsize=(7, 7),
    fontsize_tmd_jmd=15,
    weight_tmd_jmd="bold",
)

fig = plt.gcf()
fig.set_size_inches(7, 7)
plt.tight_layout()
fig.savefig(OUT, dpi=150, facecolor="white")
print("saved", OUT)
