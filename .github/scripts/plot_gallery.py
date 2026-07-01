"""Render an AAanalysis plot gallery across data scales, for visual inspection and
byte-exact A/B comparison.

Two things this gives you:

1. **See the plots at real data ranges** — plots are rendered across data scales
   (``tiny`` ~10% of a normal case up to ``huge`` ~1000%) on realistic bundled data,
   so you can eyeball that layout survives a near-empty map *and* a full 74-subcategory
   one. The step count scales with figure complexity: complex key figures get the full
   5-step sweep, medium figures 3 steps, and simple plots + evals a single step (see
   ``TIER_SCALES``). PNGs land in ``<outdir>/<label>/`` named ``<plot>__<scale>.png``.
2. **Detect visual change byte-exactly** — each plot is hashed and written to a
   manifest; ``compare`` diffs manifests across code versions and reports which
   plot/scale moved.

Critical: everything is saved with ``bbox_inches="tight"`` — the way notebooks (inline
backend) and publication ``savefig`` actually render. Hashing the raw canvas instead
silently misses layout changes (e.g. tick labels that only appear once the tight box
expands), so this tool renders the way plots are *consumed*.

Byte-exactness holds only within one matplotlib/freetype build: compare versions in the
same environment, or pin matplotlib for the 1.0.3 leg.

Usage
-----
    python .github/scripts/plot_gallery.py render --label branch --outdir /tmp/gallery
    python .github/scripts/plot_gallery.py compare /tmp/gallery/*/manifest.json
"""
import argparse
import glob
import hashlib
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import aaanalysis as aa


# Data-scale points: (n_subcat, n_features), ~10% of a normal (~40-feature) case up to ~1000%.
SCALE_DIMS = {
    "tiny": (2, 3),
    "small": (5, 10),
    "normal": (12, 40),
    "large": (74, 74),
    "huge": (74, 400),
}

# How many scale steps each plot tier is swept over:
#   key    = complex signature figures            -> full 5-step sweep
#   medium = scale-sensitive but simpler figures  -> 3 steps (extremes + normal)
#   simple = shape-insensitive plots + all evals   -> 1 step (normal)
TIER_SCALES = {
    "key": ["tiny", "small", "normal", "large", "huge"],
    "medium": ["tiny", "normal", "huge"],
    "simple": ["normal"],
}

# TMD sequence lengths (aa) for the sequence-rendering sweep: short sequences must
# grow the residue letters (seq_char_fill), long ones must shrink-to-fit without
# overlap. The most layout-sensitive path, so it is swept in both plain and CPP-SHAP
# modes for feature_map and heatmap.
SEQ_LENS = [5, 10, 20, 40, 80, 100]
_AA_CYCLE = "ACDEFGHIKLMNPQRSTVWY"


def _tight_png_bytes(fig):
    """PNG bytes as a notebook/publication would save it (tight bbox)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    return buf.getvalue()


def _scaled_df_feat(n_subcat, n_features):
    """A realistic df_feat resized to (n_subcat subcategories, n_features rows).

    Built from the bundled DOM_GSEC feature set (real scales/positions/values), so
    it looks like true output; sampled with replacement + subcategory subsetting to
    hit the requested size, up or down.
    """
    aa.options["verbose"] = False
    base = aa.load_features(name="DOM_GSEC")
    subs = list(dict.fromkeys(base["subcategory"]))[:n_subcat]
    pool = base[base["subcategory"].isin(subs)]
    idx = np.resize(np.arange(len(pool)), n_features)  # deterministic up/down sample
    df = pool.iloc[idx].reset_index(drop=True)
    return df


def _cpp_composite(method, dims, **extra):
    """Render one CPP composite (feature_map/heatmap/profile/ranking) at a data scale."""
    n_subcat, n_features = dims
    df = _scaled_df_feat(n_subcat, n_features)
    cpp = aa.CPPPlot(df_scales=aa.load_scales())
    figsize = (8, 8) if method in ("feature_map", "heatmap") else (7, 5)
    return getattr(cpp, method)(df_feat=df, figsize=figsize, **extra)[0]


def _aaclust_fig(method):
    """Render an AAclust plot on the bundled scales (shape-insensitive -> 1 step)."""
    X = aa.load_scales().T.to_numpy()
    labels = aa.AAclust().fit(X, n_clusters=5).labels_
    return getattr(aa.AAclustPlot(), method)(X, labels=labels)[0]


def _seq_df_feat(tmd_len, n_subcat=8):
    """A df_feat valid for a given TMD length, with SHAP (feat_impact/mean_dif)
    columns so both plain and shap_plot renders work. Whole-TMD segment features
    keep it valid at any ``tmd_len`` (5..100)."""
    dcat = aa.CPPPlot()._df_cat
    subs = list(dict.fromkeys(dcat["subcategory"]))[:n_subcat]
    rows = []
    for s in subs:
        r = dcat[dcat["subcategory"] == s].iloc[0]
        rows.append(dict(feature=f"TMD-Segment(1,1)-{r['scale_id']}", category=r["category"],
            subcategory=s, scale_name=r["scale_name"], scale_description=r["scale_description"],
            abs_auc=0.2, abs_mean_dif=0.3, mean_dif=0.3, std_test=0.1, std_ref=0.1,
            p_val_mann_whitney=0.01, p_val_fdr_bh=0.02,
            positions=",".join(str(p) for p in range(11, 11 + tmd_len)),
            feat_importance=1.0, feat_importance_std=0.1,
            feat_impact_test=2.0, mean_dif_test=1.0))
    return pd.DataFrame(rows)


def _seqs(tmd_len):
    """JMD-N (10) + TMD (tmd_len) + JMD-C (10) sequences, varied residues."""
    def take(n, off=0):
        return "".join(_AA_CYCLE[(off + i) % len(_AA_CYCLE)] for i in range(n))
    return dict(jmd_n_seq=take(10), tmd_seq=take(tmd_len, 3), jmd_c_seq=take(10, 7))


def _seq_fig(method, tmd_len, shap):
    """feature_map/heatmap with a TMD-JMD sequence of length tmd_len; SHAP or plain."""
    df = _seq_df_feat(tmd_len)
    cpp = aa.CPPPlot(df_scales=aa.load_scales())
    kws = dict(tmd_len=tmd_len, figsize=(8, 8), **_seqs(tmd_len))
    if shap:
        kws.update(shap_plot=True, col_val="feat_impact_test")
        if method == "feature_map":  # only feature_map has the importance bars needing col_imp
            kws["col_imp"] = "feat_impact_test"
    return getattr(cpp, method)(df_feat=df, **kws)[0]


# (name, tier, render_fn(scale_label)) -> Figure. render_fn ignores scale for simple plots.
PLOTS = [
    ("feature_map", "key", lambda s: _cpp_composite("feature_map", SCALE_DIMS[s])),
    ("heatmap", "key", lambda s: _cpp_composite("heatmap", SCALE_DIMS[s])),
    ("profile", "medium", lambda s: _cpp_composite("profile", SCALE_DIMS[s])),
    ("ranking", "medium", lambda s: _cpp_composite(
        "ranking", SCALE_DIMS[s], n_top=min(15, SCALE_DIMS[s][1]))),
    ("aaclust_centers", "simple", lambda s: _aaclust_fig("centers")),
    ("aaclust_medoids", "simple", lambda s: _aaclust_fig("medoids")),
]


def render(label, outdir):
    dest = os.path.join(outdir, label)
    os.makedirs(dest, exist_ok=True)
    manifest = {"label": label, "aaanalysis_version": aa.__version__,
                "aaanalysis_file": aa.__file__, "matplotlib": matplotlib.__version__,
                "plots": {}}
    aa.options["verbose"] = False

    def _emit(key, thunk, tag):
        try:
            fig = thunk()
            png = _tight_png_bytes(fig)
            manifest["plots"][key] = hashlib.sha256(png).hexdigest()
            with open(os.path.join(dest, f"{key}.png"), "wb") as fh:
                fh.write(png)
            print(f"  rendered {key}  ({tag})")
        except Exception as exc:
            manifest["plots"][key] = f"ERROR: {type(exc).__name__}: {exc}"
            print(f"  FAILED  {key}: {exc}")
        finally:
            plt.close("all")

    # Data-scale sweep (n_subcat x n_features), tiered by figure complexity.
    for name, tier, fn in PLOTS:
        for sname in TIER_SCALES[tier]:
            _emit(f"{name}__{sname}", lambda fn=fn, s=sname: fn(s), tier)

    # Sequence-length sweep (5..100 aa), plain and CPP-SHAP, for feature_map + heatmap.
    for method in ("feature_map", "heatmap"):
        for shap in (False, True):
            mode = "shap" if shap else "plain"
            for L in SEQ_LENS:
                key = f"{method}_seq_{mode}__L{L:03d}"
                _emit(key, lambda m=method, L=L, sh=shap: _seq_fig(m, L, sh), f"seq/{mode}")
    with open(os.path.join(dest, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    n_ok = sum(1 for v in manifest["plots"].values() if not str(v).startswith("ERROR"))
    print(f"wrote {dest}/manifest.json  ({n_ok}/{len(manifest['plots'])} ok, "
          f"aaanalysis {manifest['aaanalysis_version']})")


def compare(manifest_paths):
    mans = [json.load(open(p)) for p in manifest_paths]
    labels = [m["label"] for m in mans]
    print("versions:")
    for m in mans:
        print(f"  {m['label']:>10}: aaanalysis {m['aaanalysis_version']}  (mpl {m['matplotlib']})")
    names = sorted({n for m in mans for n in m["plots"]})
    print(f"\n{'plot__scale':<26} " + " ".join(f"{l:<12}" for l in labels) + " verdict")
    n_changed = 0
    for name in names:
        hashes = [m["plots"].get(name, "MISSING") for m in mans]
        cells = ["ERR" if str(h).startswith("ERROR") else str(h)[:10] for h in hashes]
        real = {h for h in hashes if not str(h).startswith("ERROR")}
        errs = [h for h in hashes if str(h).startswith("ERROR")]
        verdict = "ERROR" if errs else ("identical" if len(real) <= 1 else "CHANGED")
        if verdict == "CHANGED":
            n_changed += 1
        print(f"{name:<26} " + " ".join(f"{c:<12}" for c in cells) + f" {verdict}")
    print(f"\n{n_changed} of {len(names)} plot/scale renders CHANGED.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--label", required=True)
    r.add_argument("--outdir", required=True)
    c = sub.add_parser("compare")
    c.add_argument("manifests", nargs="+")
    args = ap.parse_args()
    if args.cmd == "render":
        render(args.label, args.outdir)
    else:
        paths = []
        for m in args.manifests:
            paths.extend(sorted(glob.glob(m)) if "*" in m else [m])
        compare(paths)


if __name__ == "__main__":
    main()
