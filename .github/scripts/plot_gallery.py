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
    for name, tier, fn in PLOTS:
        for sname in TIER_SCALES[tier]:
            key = f"{name}__{sname}"
            try:
                fig = fn(sname)
                png = _tight_png_bytes(fig)
                manifest["plots"][key] = hashlib.sha256(png).hexdigest()
                with open(os.path.join(dest, f"{key}.png"), "wb") as fh:
                    fh.write(png)
                print(f"  rendered {key}  ({tier})")
            except Exception as exc:
                manifest["plots"][key] = f"ERROR: {type(exc).__name__}: {exc}"
                print(f"  FAILED  {key}: {exc}")
            finally:
                plt.close("all")
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
