# Repository layout

Not path-scoped. Referenced from the root CLAUDE.md.

> **How the subpackages connect at runtime** (the `load → parts → CPP → model →
> explain → plot` dataflow): see `docs/module_map.md`. This tree shows *where things
> live*; the map shows *how they flow*.

```
aaanalysis/
  __init__.py            # public re-exports + pro stubs
  utils.py               # IMPORT BARREL — constants + validators + plotting glue
  config.py              # global `options`
  template_classes.py    # Wrapper, Tool ABCs
  _utils/                # truly internal helpers (do not import from outside utils.py)
  _data/                 # bundled XLSX/TSV scales/datasets
  data_handling/         # load_dataset/scales/features, read_fasta, SequencePreprocessor
  feature_engineering/   # AAclust(+Plot), SequenceFeature, NumericalFeature, CPP(+Plot)
  pu_learning/           # dPULearn(+Plot)
  explainable_ai/        # TreeModel
  explainable_ai_pro/    # ShapModel (pro)
  seq_analysis/          # AALogo(+Plot), AAWindowSampler
  seq_analysis_pro/      # comp_seq_sim, filter_seq, scan_motif (pro)
  protein_engineering/        # AAMut(+Plot), SeqMut(+Plot)
  plotting/              # plot_settings, plot_get_clist/cmap/cdict, plot_legend, plot_gcfs
  metrics/               # comp_auc_adjusted, comp_bic_score, comp_kld
  show_html/             # display_df (dev)
tests/{unit,integration,e2e}/
docs/source/             # Sphinx + nbsphinx
examples/                # ipynb examples (one per public method, included into docstrings)
tutorials/               # narrative ipynb tutorials
dev_scripts/             # ad-hoc scripts for prototyping (not packaged, not tested)
.claude/rules/           # path-scoped Claude Code instructions
.github/workflows/       # CI
pyproject.toml           # PEP 621 + Poetry duality
```

Two class templates in `aaanalysis/template_classes.py`:
- `Wrapper` — sklearn-style; implements `.fit` and `.eval`.
- `Tool` — pipeline; implements `.run` and `.eval`.

Plot classes mirror logic classes 1:1 with a `Plot` suffix and visualize
via `.eval`.

Optional extras: `pro` (heavy or fragile deps: biopython, shap, UpSetPlot,
meme suite for FIMO, etc.), `docs`, `dev`. Pro features are imported in
top-level `__init__.py` inside `try/except ImportError` and replaced with
`missing_feature_stub` so import always succeeds — see
`pro-core-boundary.md` for the mechanics.
