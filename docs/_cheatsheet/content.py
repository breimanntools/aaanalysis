"""This is the single source of truth for the AAanalysis cheat sheet content.

The same ``CONTENT`` dict drives both rendered outputs (``cheat_sheet.html`` and
``cheat_sheet.pdf``) via ``build_cheat_sheet.py`` -> ``template.html.jinja``, so the
two artifacts can never drift. Edit content here, then regenerate (see README.md).

Every code snippet uses only public ``aaanalysis.__all__`` symbols with real
signatures, and every term matches the canonical glossary in ``CONTEXT.md``.
"""

META = {
    "version": "v1.1.0",
    "date": "2026-06",
    "pip": "pip install aaanalysis",
    "docs_url": "aaanalysis.readthedocs.io",
    "tagline": "Sequence → physicochemical scales → interpretable features → "
               "explainable ML → biological mechanism",
    "subtitle": "Interpretable, sequence-based protein prediction — small-data "
                "robust, mechanism-aware, publication-ready.",
    "mental_model": "Every feature is a Part × Split × Scale — every prediction is "
                    "traceable to a residue × property × group comparison.",
    "intro": "<b>AAanalysis</b> is a Python framework for interpretable, "
             "sequence-based protein prediction. It turns sequences into "
             "physicochemical features (CPP), trains explainable models, and "
             "traces every prediction back to a residue × property × group "
             "comparison — robust for small datasets. <b>v1.1</b> extends the "
             "core feature engine beyond physicochemical scales to PLM "
             "embeddings and protein structure.",
    "copyright": "© Stephan Breimann · BSD-3",
}

# -- Page 1: orient & think ---------------------------------------------------

GOLDEN_WORKFLOW = [
    {"n": "1", "title": "LOAD", "call": "load_dataset · load_scales",
     "out": "df_seq · df_scales"},
    {"n": "2", "title": "PARTS", "call": "SequenceFeature.get_df_parts",
     "out": "df_parts"},
    {"n": "3", "title": "FEATURES", "call": "Part × Split × Scale · CPP.run",
     "out": "df_feat"},
    {"n": "4", "title": "MODEL", "call": "TreeModel.fit · dPULearn.fit",
     "out": "feat_importance · labels_"},
    {"n": "5", "title": "EXPLAIN", "call": "CPPPlot.feature_map · ShapModel",
     "out": "figure · feat_impact"},
]

# Prediction-task taxonomy (ADR-0022): the level is a proxy; what actually
# determines the CPP setup is the unit profiled + how the reference is built.
PREDICTION_LEVELS = [
    {"level": "Residue", "prefix": "AA_*",
     "unit": "sliding window (aa_window_size)",
     "submodes": "single-residue · odd window (PTM/site) · between-residues · "
                 "even window (cleavage bond)",
     "reference": "non-site windows / shuffled background",
     "strategy": "positional"},
    {"level": "Domain", "prefix": "DOM_*",
     "unit": "part-set jmd_n · tmd · jmd_c (from tmd_start / tmd_stop)",
     "submodes": "flagship — the TMD model",
     "reference": "labelled A vs B groups",
     "strategy": "both"},
    {"level": "Protein", "prefix": "SEQ_*",
     "unit": "whole chain (composition)",
     "submodes": "‘protein’ is the user-facing alias of the SEQ_ prefix",
     "reference": "labelled groups / composition-matched background",
     "strategy": "compositional"},
]

FEATURE_ONTOLOGY = {
    "part": {"sub": "where on the sequence",
             "items": ["tmd · jmd_n · jmd_c", "tmd_jmd · jmd_n_tmd_n", "tmd_c_jmd_c"]},
    "split": {"sub": "how to read the part",
              "items": ["Segment — contiguous", "Pattern — sparse pairs",
                        "PeriodicPattern — i, i+3/4"]},
    "scale": {"sub": "which physicochemical property",
              "items": ["AAontology (~600 scales)", "hydrophobicity · charge",
                        "helix propensity"]},
    "examples": [
        "TMD × Segment × hydrophobicity → membrane insertion",
        "JMD × Pattern × net charge → electrostatic recognition",
        "TMD × PeriodicPattern × helix → α-helical interface",
    ],
}

# Splits schema (page 1): how each Split type selects residues of a Part — a
# simplified take on Breimann25a Suppl. Fig. C (APP / NOTCH1 TMD splits). Each
# mask aligns 1:1 with the residues above it (■ = selected, · = skipped).
SPLITS_SCHEMA = {
    "intro": "A Split picks which residues of a Part feed each Scale:",
    "seq_label": "TMD",
    "seq":       "A I I G L M V G G V V I",
    "rows": [
        ("Segment(1,4)",     "■ ■ ■ · · · · · · · · ·"),
        ("Pattern(N,1,4,8)", "■ · · ■ · · · ■ · · · ·"),
        ("PeriodicPattern",  "■ · · ■ · · ■ · · ■ · ·"),
    ],
    "caption": "contiguous (Segment) · fixed positions (Pattern) · "
               "periodic, α-helix face (PeriodicPattern).",
    "ref_text": "Simplified from Breimann25a (Suppl. Fig. 1C ↗)",
    "ref_url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-60638-z/"
               "MediaObjects/41467_2025_60638_MOESM1_ESM.pdf",
}

# Compositional vs positional is not a setting — it emerges from split_kws (#86).
CPP_STRATEGIES = {
    "intro": "Compositional vs positional is not a flag — it emerges from "
             "split_kws. The two regimes map onto the prediction levels.",
    "compositional": {
        "name": "Compositional", "maps": "≈ sequence/protein-level",
        "desc": "one whole-part average (composition-like, position-agnostic)",
        "code": 'split_kws = sf.get_split_kws(\n'
                '    split_types="Segment",\n'
                '    n_split_max=1)\n'
                'cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws)',
    },
    "positional": {
        "name": "Positional", "maps": "≈ residue-/region-level",
        "desc": "sub-segments and/or patterns resolved to positions",
        "code": 'split_kws = sf.get_split_kws(\n'
                '    split_types=["Segment", "Pattern", "PeriodicPattern"],\n'
                '    n_split_max=5,\n'
                '    steps_pattern=[3, 4],\n'
                '    steps_periodicpattern=[3, 4])\n'
                'cpp = aa.CPP(df_parts=df_parts, split_kws=split_kws)',
    },
    "note": "Domain level uses both. → CPP strategies: see the CPP tutorial (docs).",
}

# Beginner decision flow: user intent -> the module to reach for. Ordered to
# follow the RTD API subpackages — Sequence Analysis, Feature Engineering,
# PU Learning, Explainable AI — while keeping the intent -> key-class mapping.
WHICH_MODULE = [
    ("Explore sequence patterns / composition", "AAlogo"),
    ("Sample reference windows (if negatives are missing)", "AAWindowSampler"),
    ("Reduce redundant amino acid scales", "AAclust"),
    ("Discover discriminative physicochemical features", "CPP"),
    ("Train with positives + unlabeled data", "dPULearn"),
    ("Train an interpretable classifier", "TreeModel"),
    ("Explain a prediction (per feature / sample)", "ShapModel  [pro]"),
]

SEQUENCE_ANATOMY = {
    "rows": [
        ("TMD", "Target Middle Domain — the central segment of interest "
                "(e.g. transmembrane stretch, binding region)."),
        ("JMD", "Juxta Middle Domain — the flanks adjoining the TMD "
                "(jmd_n on the N-side, jmd_c on the C-side)."),
    ],
    "track": ["JMD-N", "TMD", "JMD-C"],
    "coords": "0 · tmd_start · tmd_stop · len(seq)",
    "note": "JMD widths set globally: aa.options['jmd_n_len'] · ['jmd_c_len'].",
    "pu_note": "dPULearn labels (dpu.labels_): 1 positives · 0 reliable-negatives · "
               "2 unlabeled.",
}

# -- Page 2: capabilities & API -----------------------------------------------

INSTALL = (
    "# Python >= 3.11\n"
    "pip install aaanalysis         # core\n"
    "pip install 'aaanalysis[pro]'  # SHAP, FIMO, Bio\n"
    "\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import aaanalysis as aa\n"
    "df_seq = aa.load_dataset(name='DOM_GSEC')   # γ-secretase\n"
    "labels = df_seq['label'].to_list()\n"
    "df_scales = aa.load_scales()"
)

# Five capability families mirroring the package subpackages. Each row is
# (capability, canonical symbol(s), minimal snippet | None). [pro] = extra.
# Module map: each row shows the call with 1-2 key params and how it connects to
# the Golden-Workflow objects (df_seq · df_parts · df_feat · X · labels).
CAPABILITY_FAMILIES = [
    {"name": "Data & Preparation", "tag": "load · clean",
     "rows": [
         ("Load benchmark sequences", "load_dataset(name) → df_seq", None),
         ("Load AAontology scales", "load_scales() → df_scales", None),
         ("Load precomputed features", "load_features(name) → df_feat", None),
         ("Read / write FASTA", "read_fasta(file) → df_seq", None),
         ("Cluster redundant homologs", "filter_seq(df_seq) → df_clust  [pro]", None),
     ]},
    {"name": "Sequence Analysis", "tag": "logos · motifs",
     "rows": [
         ("Position-specific logo", "AAlogo().get_df_logo(df_parts) → df_logo", None),
         ("Sample reference windows", "AAWindowSampler().sample_*(df_seq)", None),
         ("Pairwise sequence similarity", "comp_seq_sim(df_seq)  [pro]", None),
         ("Scan motifs (FIMO / MEME)", "scan_motif(df_seq, pwm) → df_hits  [pro]", None),
     ]},
    {"name": "Feature Engineering", "tag": "parts · CPP · scales", "flagship": True,
     "rows": [
         ("SequenceFeature  →  sf", "sf = aa.SequenceFeature()", None),
         ("· split sequence into parts", "sf.get_df_parts(df_seq) → df_parts", None),
         ("· assemble feature matrix X", "sf.feature_matrix(df_feat, df_parts) → X", None),
         ("Discover discriminative features", "CPP(df_parts).run(labels) → df_feat  ★", None),
         ("Sweep CPP configs (grid)", "CPPGrid().run(...) · .eval() → ranked configs", None),
         ("Simplify → interpretable scales", "CPP.simplify(df_feat, labels) → df_feat", None),
         ("Reduce redundant scales", "AAclust().fit(X)  [Wrapper]", None),
         ("Drop correlated features", "NumericalFeature().filter_correlation(X)", None),
     ]},
    {"name": "Feature Preprocessing", "tag": "one-hot · PLM · structure · PTM",
     "rows": [
         ("Encode sequences (one-hot / int)", "SequencePreprocessor().encode_*(seqs) → X", None),
         ("PLM embeddings", "EmbeddingPreprocessor().encode(...) → dict_num", None, "v1.1"),
         ("Structure / DSSP / PAE", "StructurePreprocessor().encode_dssp(...) → dict_num  [pro]", None, "v1.1"),
         ("PTM / site annotations", "AnnotationPreprocessor().encode(...) → dict_num  [pro]", None, "v1.1"),
         ("Combine sources", "combine_dict_nums([...]) → dict_num", None, "v1.1"),
         ("Numerical CPP", "CPP(df_parts).run_num(dict_num_parts, labels) → df_feat", None, "v1.1"),
     ]},
    {"name": "Modeling & Explainability", "tag": "",
     "rows": [
         ("Train with positives + unlabeled data", "dPULearn().fit(X, labels)  [Wrapper]", None),
         ("Train + RFE + MC importance", "TreeModel().fit(X, labels)  [Wrapper]", None),
         ("Per-feature / sample SHAP impact", "ShapModel().fit(X, labels)  [pro]", None),
     ]},
    {"name": "Metrics & Plotting", "tag": "utilities",
     "rows": [
         ("Adjusted AUC (class imbalance)", "comp_auc_adjusted(X, labels)", None),
         ("BIC score · KL divergence", "comp_bic_score(X, labels) · comp_kld", None),
         ("Per-protein / detection (v1.1)", "comp_per_protein_ap · comp_detection_metrics", None),
         ("Plot style, fonts & standalone legend", "plot_settings(font_scale) · plot_legend(ax)", None),
     ]},
    {"name": "Protein Design", "tag": "",
     "under_construction": True,
     "rows": [
         ("In-silico point mutations", "AAMut · AAMutPlot", None, "v1.1"),
         ("Sequence-design libraries", "SeqMut · SeqMutPlot", None, "v1.1"),
     ]},
]

# Detailed recipes for the core analytical classes (page 2, right zone).
# Worked examples — tutorial-convention code paired with the figure it produces.
FLAGSHIP_RECIPES = [
    {"cls": "AAlogo — see the data", "tag": "dataset at a glance", "img": "logo", "logo": True,
     "caption": "AAlogoPlot.single_logo · per-position enrichment",
     "code": "import numpy as np, matplotlib.pyplot as plt, aaanalysis as aa\n"
             "df_seq = aa.load_dataset(name='DOM_GSEC')   # γ-secretase\n"
             "labels = list(df_seq['label']); df_scales = aa.load_scales()\n"
             "sf = aa.SequenceFeature()\n"
             "df_parts = sf.get_df_parts(df_seq=df_seq,\n"
             "    list_parts=['tmd', 'jmd_n', 'jmd_c'])\n"
             "aa.plot_settings(font_scale=0.7)\n"
             "# aal_kws builds df_logo + bits bar for you\n"
             "aa.AAlogoPlot().single_logo(\n"
             "    aal_kws=dict(df_parts=df_parts, labels=labels,\n"
             "        label_test=1, tmd_len=20),\n"
             "    name_data='Test set: substrates')\n"
             "plt.tight_layout(); plt.show()"},
    {"cls": "CPP — ranking", "tag": "top features · effect + importance", "img": "ranking",
     "caption": "CPPPlot.ranking · top-15 features",
     "code": "# same df_feat — rank the top discriminative features\n"
             "aa.plot_settings(font_scale=0.6)\n"
             "cpp_plot.ranking(df_feat=df_feat, n_top=15, rank=True,\n"
             "    name_test='substrates', name_ref='non-subs.')\n"
             "plt.tight_layout(); plt.show()"},
    {"cls": "CPP — feature", "tag": "top feature · REF vs TEST", "img": "feature",
     "caption": "CPPPlot.feature · top feature, REF vs TEST",
     "code": "# default parts + a redundancy-reduced set of 100 scales\n"
             "df_parts = sf.get_df_parts(df_seq=df_seq)\n"
             "df_scales = aa.AAclust().select_scales(\n"
             "    df_scales=df_scales, n_clusters=100)\n"
             "cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)\n"
             "df_feat = cpp.run(labels=labels, n_filter=100)\n"
             "X = sf.feature_matrix(df_feat['feature'], df_parts)\n"
             "tm = aa.TreeModel(); tm.fit(X, labels=labels)\n"
             "df_feat = tm.add_feat_importance(df_feat=df_feat, sort=True)\n"
             "cpp_plot = aa.CPPPlot(); aa.plot_settings()\n"
             "# distribution of the top feature (feat_rank=1 of the sorted df_feat)\n"
             "cpp_plot.feature(feature=df_feat, feat_rank=1, df_seq=df_seq,\n"
             "    labels=labels, name_test='substrates', name_ref='non-subs.')\n"
             "plt.tight_layout(); plt.show()"},
    {"cls": "CPP — feature map", "tag": "group level · full vs simplified",
     "imgs": ["feature_map", "feature_map_simplified"],
     "img_labels": ["CPPPlot.feature_map · all scales", "CPPPlot.feature_map · simplified"], "h": 44,
     "code": "# global Part × Split × Scale map — all AAontology scales\n"
             "cpp_plot = aa.CPPPlot(); aa.plot_settings(font_scale=0.65)\n"
             "cpp_plot.feature_map(df_feat=df_feat)\n"
             "# CPP.simplify → fewer, interpretable correlated scales\n"
             "df_feat = cpp.simplify(df_feat=df_feat, labels=labels)\n"
             "cpp_plot.feature_map(df_feat=df_feat)\n"
             "plt.tight_layout(); plt.show()"},
    {"cls": "ShapModel — explain a prediction", "tag": "sample level · advanced · [pro]",
     "imgs": ["shap_profile", "feature_map_shap"],
     "img_labels": ["CPPPlot.profile · SHAP", "CPPPlot.feature_map · SHAP"], "h": 36,
     "code": "# advanced: per-sample explanation (fuzzy labeling demo)\n"
             "# fuzzy labeling: APP's label is its soft prediction score (0.6, not 1)\n"
             "i = list(df_seq['entry']).index('P05067')   # APP\n"
             "y = [float(v) for v in labels]; y[i] = 0.6\n"
             "sm = aa.ShapModel(); sm.fit(X, labels=y, fuzzy_labeling=True)\n"
             "df_feat = sm.add_feat_impact(df_feat=df_feat,\n"
             "              sample_positions=i, names='APP')\n"
             "args_seq = {k + '_seq': v for k, v in sf.get_df_parts(\n"
             "    df_seq=df_seq).loc['P05067'].to_dict().items()}\n"
             "ka = dict(col_imp='feat_impact_APP', shap_plot=True, **args_seq)\n"
             "cpp_plot.profile(df_feat=df_feat, **ka)\n"
             "cpp_plot.feature_map(df_feat=df_feat, name_test='APP', **ka)\n"
             "plt.tight_layout(); plt.show()"},
    {"cls": "AAclust — clusters", "tag": "scale reduction · Wrapper", "img": "centers",
     "caption": "AAclustPlot.centers · cluster scale profiles",
     "code": "aac = aa.AAclust()\n"
             "aac.select_scales(df_scales, n_clusters=10)\n"
             "aac.medoid_names_   # 10 reduced scales (labels_ also set)\n"
             "\n"
             "aac_plot = aa.AAclustPlot()\n"
             "aac_plot.centers(np.array(df_scales).T, labels=aac.labels_)\n"
             "plt.tight_layout(); plt.show()"},
    {"cls": "dPULearn — PCA", "tag": "reliable negatives · Wrapper", "img": "pca",
     "caption": "dPULearnPlot.pca · reliable negatives",
     "code": "# DOM_GSEC ships 1/0 — encode as PU labels: 1 = positive, 2 = unlabeled\n"
             "labels_pu = [1 if x == 1 else 2 for x in labels]\n"
             "n_pos = sum(np.array(labels_pu) == 1)\n"
             "dpul = aa.dPULearn()\n"
             "dpul.fit(X=X, labels=labels_pu, n_unl_to_neg=n_pos // 2)\n"
             "df_pu = dpul.df_pu_   # 1 pos · 0 rel-neg · 2 unl\n"
             "\n"
             "dpul_plot = aa.dPULearnPlot()\n"
             "dpul_plot.pca(df_pu=df_pu, labels=dpul.labels_)\n"
             "plt.tight_layout(); plt.show()"},
    {"cls": "AAWindowSampler", "tag": "build reference windows",
     "code": "# Reference windows around sites when you lack negatives:\n"
             "aaws = aa.AAWindowSampler()\n"
             "# SAME proteins · window 9 (odd) -> PTM / single-residue site\n"
             "df_same = aaws.sample_same_protein(df_seq, n=100, window_size=9)\n"
             "# DIFFERENT proteins · window 10 (even) -> cleavage bond\n"
             "df_diff = aaws.sample_different_protein(df_seq, n=100, window_size=10)\n"
             "# SYNTHETIC — AA-frequency priors (null background)\n"
             "df_syn = aaws.sample_synthetic(df_seq, n=100, generator='global_freq')"},
]

# Page-2 layout: AAlogo stands alone (big logo); the rest are shown as PAIRS —
# two code boxes stacked, then both figures side-by-side below (left | right),
# so each figure pair is as wide as the code block.
# Each pair renders two code boxes then both figures SIDE-BY-SIDE in one row, set
# to a shared row height "h" (mm) so the two plots are exactly the same height
# (widths follow each plot's aspect ratio). h is tuned so the two widths fit one
# column. Feature maps sit in the middle column.
RECIPE_GROUPS = [
    {"recipes": [FLAGSHIP_RECIPES[0]]},                                  # AAlogo (col1, big logo)
    {"recipes": [FLAGSHIP_RECIPES[2], FLAGSHIP_RECIPES[1]], "h": 30},    # feature | ranking (col1)
    {"recipes": [FLAGSHIP_RECIPES[3]]},                                  # feature map: full | simplified (col2 top)
    {"recipes": [FLAGSHIP_RECIPES[4]]},                                  # SHAP: profile | feature map (col2 bottom)
    {"recipes": [FLAGSHIP_RECIPES[5], FLAGSHIP_RECIPES[6]], "h": 37},    # AAclust | dPULearn (col3)
    {"recipes": [FLAGSHIP_RECIPES[7]]},                                  # AAWindowSampler (code-only, pairs with dPULearn — col3)
]

# -- Page 3: reference --------------------------------------------------------

# Compact copy-paste minimal pipeline.
FIVE_MINUTE = (
    "import matplotlib.pyplot as plt\n"
    "import aaanalysis as aa\n"
    "df_seq = aa.load_dataset(name='DOM_GSEC')\n"
    "labels = df_seq['label'].to_list()\n"
    "sf = aa.SequenceFeature()\n"
    "df_parts = sf.get_df_parts(df_seq=df_seq)\n"
    "cpp = aa.CPP(df_parts=df_parts)\n"
    "df_feat = cpp.run(labels=labels)\n"
    "X = sf.feature_matrix(features=df_feat['feature'],\n"
    "                      df_parts=df_parts)\n"
    "tm = aa.TreeModel(); tm.fit(X, labels=labels)\n"
    "df_feat = tm.add_feat_importance(df_feat=df_feat)\n"
    "cpp_plot = aa.CPPPlot()\n"
    "aa.plot_settings()\n"
    "cpp_plot.feature_map(df_feat=df_feat)\n"
    "plt.tight_layout(); plt.show()"
)

# Decision guide (A): the two decisions that define a task — what to predict
# (level + parts) and what training signal you have (which model) — then how to
# discover and explain. Mirrors the scikit-learn "choose an estimator" map idea.
DECISION_GUIDE = [
    ("What are you predicting?", [
        ("per residue / site", "AA_* · odd/even window · parts = window"),
        ("per domain / region", "DOM_* · TMD model · parts = jmd_n·tmd·jmd_c"),
        ("whole protein", "SEQ_* · composition · whole chain"),
    ]),
    ("What labels do you have?", [
        ("labeled 0 / 1", "CPP → ML model"),
        ("positives + unlabeled (1 / 2)", "CPP → dPULearn → ML model"),
        ("no negatives at all", "AAWindowSampler → CPP → ML model"),
    ]),
    ("What is your learning task?", [
        ("classify", "any classifier (sklearn)"),
        ("regress", "get_labels_quantile / tiered"),
        ("multi-class", "get_labels_ovr / ovo"),
        ("cluster · PCA", "AAclust · dPULearn"),
    ]),
    ("Which explainability do you need?", [
        ("feature importance (group)", "TreeModel → CPPPlot"),
        ("feature impact (per protein)",
         'ShapModel [pro] → CPPPlot '
         '<span class="shap-up">↑</span><span class="shap-dn">↓</span>'),
    ]),
]

# Gotchas (B): the non-obvious rules that bite. <b> spans -> rendered |safe.
GOTCHAS = [
    "Labels: <b>1/0</b> = supervised (pos/neg). <b>dPULearn needs 1/2</b> "
    "(pos/unlabeled) and outputs <b>0 = reliable-negative</b>.",
    "<b>load_dataset(name, n=N)</b> returns <b>2N</b> rows (N per class) — "
    "count classes via df_seq['label'].",
    "Compositional vs positional is not a flag — it <b>emerges from split_kws</b>.",
    "Reproducibility: layered seeds — seed= ▸ random_state= ▸ "
    "options['random_state'] ▸ default.",
    "<b>DOM_*</b> parts need tmd_start/tmd_stop in df_seq; <b>[pro]</b> features "
    "need <span style='font-family:\"AA Mono\",monospace'>pip install 'aaanalysis[pro]'</span>.",
    "<b>TMD</b> = <b>Target</b> Middle Domain — generalized from <b>Transmembrane</b> "
    "domain (Breimann25a) to any central segment; JMD-N / JMD-C are its flanks.",
]

# Data objects (C): the canonical tables/arrays and their columns/shape.
DATA_OBJECTS = [
    ("df_seq", "entry · sequence · label · tmd_start · tmd_stop"),
    ("df_parts", "one column per part: tmd · jmd_n · jmd_c · …"),
    ("df_feat", "feature · category · subcategory · scale_name · abs_auc · mean_dif · p_val · positions"),
    ("X", "feature matrix (samples × features) from sf.feature_matrix"),
    ("dict_num", "{entry: ndarray (L×D)} — numerical per-residue values (v1.1)"),
]

OPTIONS = (
    "aa.options['random_state'] = 42\n"
    "aa.options['verbose'] = True\n"
    "aa.options['n_jobs'] = -1            # all cores (None = auto)\n"
    "aa.options['allow_multiprocessing'] = True\n"
    "\n"
    "# TMD model — JMD flank widths\n"
    "aa.options['jmd_n_len'] = 10\n"
    "aa.options['jmd_c_len'] = 10\n"
    "\n"
    "# plot labels & system-level scales\n"
    "aa.options['name_tmd'] = 'P5-P5′'   # e.g. cleavage site\n"
    "aa.options['df_scales'] = my_scales"
)

# (class, abbr, plot class | "—", kind tag) — abbr = canonical instance name
# (mirrors the registry in docstring_guide.rst / test_class_abbreviation_registry.py)
CLASS_PLOT = [
    # Data Handling
    ("SequencePreprocessor", "sp", "—", ""),
    ("EmbeddingPreprocessor", "ep", "—", ""),
    ("StructurePreprocessor  [pro]", "stp", "—", ""),
    ("AnnotationPreprocessor  [pro]", "ap", "—", ""),
    # Sequence Analysis
    ("AAlogo", "aal", "AAlogoPlot", ""),
    ("AAWindowSampler", "aaws", "—", ""),
    # Feature Engineering
    ("SequenceFeature", "sf", "—", ""),
    ("CPP", "cpp", "CPPPlot", ""),
    ("AAclust", "aac", "AAclustPlot", "Wrapper"),
    ("NumericalFeature", "nf", "—", ""),
    # PU Learning
    ("dPULearn", "dpul", "dPULearnPlot", "Wrapper"),
    # Explainable AI
    ("TreeModel", "tm", "—", "Wrapper"),
    ("ShapModel  [pro]", "sm", "—", "Wrapper"),
    # Protein Design
    ("AAMut", "aamut", "AAMutPlot", ""),
    ("SeqMut", "seqmut", "SeqMutPlot", ""),
]

DESIGN_PRINCIPLES = [
    "Explicit over implicit — DataFrames everywhere",
    "Wrappers (.fit / .predict / .eval) set trailing *_ attributes after fit",
    "Biological interpretability is first-class",
    "Small-data robust and reproducible (layered seeds)",
]

CITATIONS = [
    {"name": "AAclust", "key": "[Breimann24a]",
     "ref": "Breimann & Frishman (2024a), AAclust: k-optimized clustering for "
            "selecting redundancy-reduced sets of amino acid scales",
     "journal": "Bioinformatics Advances",
     "url": "https://academic.oup.com/bioinformaticsadvances/article/4/1/vbae165/7852846"},
    {"name": "AAontology", "key": "[Breimann24b]",
     "ref": "Breimann et al. (2024b), AAontology: An ontology of amino acid "
            "scales for interpretable machine learning",
     "journal": "Journal of Molecular Biology",
     "url": "https://www.sciencedirect.com/science/article/pii/S0022283624003267"},
    {"name": "CPP & dPULearn", "key": "[Breimann25a]",
     "ref": "Breimann & Kamp et al. (2025), Charting γ-secretase substrates by "
            "explainable AI",
     "journal": "Nature Communications",
     "url": "https://www.nature.com/articles/s41467-025-60638-z"},
]

# Deeper mental models promoted from CONTEXT.md (the canonical glossary); the
# df_seq/df_parts/df_feat data-shape entries now live in the "Data Objects" panel.
GLOSSARY = [
    # -- The CPP feature model --------------------------------------------
    ("Feature (CPP)", "(Part × Split × Scale) — the atomic, residue-grounded, "
     "interpretable unit of CPP."),
    ("Part", "Named segment used as feature input: tmd, jmd_n, jmd_c, tmd_jmd, "
     "jmd_n_tmd_n, tmd_c_jmd_c."),
    ("Split", "How a scale is read across a part: Segment (contiguous), Pattern "
     "(sparse), PeriodicPattern (i, i+3/4)."),
    ("Scale", "AA → ℝ mapping. AAontology ships ~600 curated scales in two-level "
     "categories."),
    ("AAontology", "Two-level scale taxonomy; CPP uses its categories to organize "
     "and rank features."),
    ("CPP", "Comparative Physicochemical Profiling — discovers ranked "
     "Part × Split × Scale features."),
    ("Test vs reference group", "The A-vs-B contrast CPP profiles: a feature's "
     "mean_dif is test − reference (name_test / name_ref in CPPPlot)."),
    ("Compositional vs positional", "How split_kws resolves locality: a whole-part "
     "average (compositional) vs sub-region/position-resolved (positional)."),
    # -- CPP modes & numerical CPP ----------------------------------------
    ("Numerical CPP (pseudo-scale)", "CPP generalizes from AA→scale lookup to any "
     "per-residue tensor — PLM · structure · PTM — each a pseudo-scale via CPP.run_num.", "v1.1"),
    # -- Models, explainability & feature reduction -----------------------
    ("Feature importance vs impact", "Two explainability axes: importance = unsigned, "
     "group-level (TreeModel); impact = signed, per-sample (ShapModel, shap_plot)."),
    ("Reducing features", "Four distinct ops: redundancy reduction (AAclust scales) · "
     "feature pruning · selection (RFE) · simplification (CPP.simplify → interpretable scales)."),
    ("PU labels", "dPULearn input: 1 = positive, 2 = unlabeled. Output: "
     "1 / 0 (reliable-negative) / 2."),
    # -- Class conventions ------------------------------------------------
    ("Wrapper class", "sklearn-style class — .fit / .predict / .eval, sets trailing *_ "
     "attributes after fit."),
    ("Plot class", "*Plot mirror of an analytical class — same arguments, "
     "visualization only."),
]

FOOTER_NOTE = ("Layered seeds: seed= (call) ▸ random_state= (init) ▸ "
               "aa.options['random_state'] ▸ default.")

PREDICTION_OUTPUTS = "classification · regression · ranking · explanation"

CONTENT = {
    "meta": META,
    "golden_workflow": GOLDEN_WORKFLOW,
    "prediction_levels": PREDICTION_LEVELS,
    "prediction_outputs": PREDICTION_OUTPUTS,
    "feature_ontology": FEATURE_ONTOLOGY,
    "splits_schema": SPLITS_SCHEMA,
    "cpp_strategies": CPP_STRATEGIES,
    "sequence_anatomy": SEQUENCE_ANATOMY,
    "install": INSTALL,
    "which_module": WHICH_MODULE,
    "five_minute": FIVE_MINUTE,
    "decision_guide": DECISION_GUIDE,
    "gotchas": GOTCHAS,
    "data_objects": DATA_OBJECTS,
    "capability_families": CAPABILITY_FAMILIES,
    "flagship_recipes": FLAGSHIP_RECIPES,
    "recipe_groups": RECIPE_GROUPS,
    "options": OPTIONS,
    "class_plot": CLASS_PLOT,
    "design_principles": DESIGN_PRINCIPLES,
    "citations": CITATIONS,
    "glossary": GLOSSARY,
    "footer_note": FOOTER_NOTE,
}
