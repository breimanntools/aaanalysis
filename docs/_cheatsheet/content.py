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
             "comparison — robust for small datasets.",
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
                        "helix propensity · shape"]},
    "examples": [
        "TMD × Segment × hydrophobicity → membrane insertion",
        "JMD × Pattern × net charge → electrostatic recognition",
        "TMD × PeriodicPattern × helix → α-helical interface",
    ],
}

# Compositional vs positional is not a setting — it emerges from split_kws (#86).
CPP_STRATEGIES = {
    "intro": "Compositional vs positional is not a flag — it emerges from "
             "split_kws. The two regimes map onto the prediction levels.",
    "compositional": {
        "name": "Compositional", "maps": "≈ sequence/protein-level",
        "desc": "one whole-part average (composition-like, position-agnostic)",
        "code": 'split_kws = sf.get_split_kws(\n'
                '    split_types="Segment", n_split_max=1)\n'
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

# Beginner decision flow: user intent -> the module to reach for.
WHICH_MODULE = [
    ("Explore sequence patterns / composition", "AAlogo · AAlogoPlot"),
    ("Discover discriminative physicochemical features", "CPP · CPPPlot"),
    ("Reduce redundant amino acid scales", "AAclust"),
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
    {"name": "Data & Preparation", "tag": "load · encode · clean",
     "rows": [
         ("Load benchmark sequences", "load_dataset(name) → df_seq", None),
         ("Load AAontology scales", "load_scales() → df_scales", None),
         ("Load precomputed features", "load_features(name) → df_feat", None),
         ("Read / write FASTA", "read_fasta(file) → df_seq", None),
         ("Encode sequences (one-hot / int)", "SequencePreprocessor().encode_*(seqs)", None),
         ("Cluster redundant homologs", "filter_seq(df_seq)  [pro]", None),
     ]},
    {"name": "Feature Engineering", "tag": "parts · CPP · scales", "flagship": True,
     "rows": [
         ("SequenceFeature  →  sf", "sf = aa.SequenceFeature()", None),
         ("· split sequence into parts", "sf.get_df_parts(df_seq) → df_parts", None),
         ("· assemble feature matrix X", "sf.feature_matrix(df_feat, df_parts) → X", None),
         ("Discover discriminative features", "CPP(df_parts).run(labels) → df_feat  ★", None),
         ("Sweep CPP configs (grid)", "CPPGrid().run(...) · .eval() → ranked configs", None),
         ("Simplify → interpretable scales", "CPP.simplify(df_feat, labels)", None),
         ("Reduce redundant scales", "AAclust().fit(X)  [Wrapper]", None),
         ("Drop correlated features", "NumericalFeature().filter_correlation(X)", None),
     ]},
    {"name": "Numerical Feature Sources", "tag": "PLM · structure · PTM", "new": True,
     "rows": [
         ("PLM embeddings", "EmbeddingPreprocessor().encode(...) → dict_num", None),
         ("Structure / DSSP / PAE", "StructurePreprocessor().encode_dssp(...)  [pro]", None),
         ("PTM / site annotations", "AnnotationPreprocessor().encode(...)  [pro]", None),
         ("Combine sources", "combine_dict_nums([...]) → dict_num", None),
         ("Slice to parts", "NumericalFeature().get_parts(...) → dict_num_parts", None),
         ("Numerical CPP", "CPP(df_parts).run_num(dict_num_parts, labels) → df_feat", None),
     ]},
    {"name": "Modeling & Explainability", "tag": "",
     "rows": [
         ("Train + RFE + MC importance", "TreeModel().fit(X, labels)  [Wrapper]", None),
         ("Train with positives + unlabeled", "dPULearn().fit(X, labels)  [Wrapper]", None),
         ("Per-feature / sample SHAP impact", "ShapModel().fit(X, labels)  [pro]", None),
     ]},
    {"name": "Sequence Analysis", "tag": "logos · motifs",
     "rows": [
         ("Position-specific logo", "AAlogo().get_df_logo(df_parts)", None),
         ("Sample sequence windows", "AAWindowSampler().sample(df_seq)", None),
         ("Pairwise sequence similarity", "comp_seq_sim(df_seq)  [pro]", None),
         ("Scan motifs (FIMO / MEME)", "scan_motif(df_seq, pwm)  [pro]", None),
     ]},
    {"name": "Metrics & Plotting", "tag": "utilities",
     "rows": [
         ("Adjusted AUC (class imbalance)", "comp_auc_adjusted(X, labels)", None),
         ("BIC score · KL divergence", "comp_bic_score(X, labels) · comp_kld", None),
         ("Per-protein / detection (v1.1)", "comp_per_protein_ap · comp_detection_metrics", None),
         ("Global plot style & fonts", "plot_settings(font_scale)", None),
         ("Colours & standalone legend", "plot_get_clist(n) · plot_legend(ax)", None),
     ]},
    {"name": "Protein Design", "tag": "",
     "under_construction": True, "new": True,
     "rows": [
         ("In-silico point mutations", "AAMut · AAMutPlot", None),
         ("Sequence-design libraries", "SeqMut · SeqMutPlot", None),
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
             "df_scales_sel = aa.AAclust().select_scales(\n"
             "    df_scales=df_scales, n_clusters=100)\n"
             "cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales_sel)\n"
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
             "cpp_plot.feature_map(df_feat=df_feat)            # left\n"
             "# CPP.simplify → fewer, interpretable correlated scales\n"
             "df_feat = cpp.simplify(df_feat=df_feat, labels=labels)\n"
             "cpp_plot.feature_map(df_feat=df_feat)            # right\n"
             "plt.tight_layout(); plt.show()"},
    {"cls": "ShapModel — explain a prediction", "tag": "sample level · [pro]",
     "imgs": ["shap_profile", "feature_map_shap"],
     "img_labels": ["CPPPlot.profile · SHAP", "CPPPlot.feature_map · SHAP"], "h": 36,
     "code": "# fuzzy labeling: APP's label is its soft prediction score (0.6, not 1)\n"
             "i = list(df_seq['entry']).index('P05067')   # APP\n"
             "y = [float(v) for v in labels]; y[i] = 0.6\n"
             "sm = aa.ShapModel(); sm.fit(X, labels=y, fuzzy_labeling=True)\n"
             "df_feat = sm.add_feat_impact(df_feat=df_feat,\n"
             "              sample_positions=i, names='APP')\n"
             "args_seq = {k + '_seq': v for k, v in sf.get_df_parts(\n"
             "    df_seq=df_seq).loc['P05067'].to_dict().items()}\n"
             "ka = dict(col_imp='feat_impact_APP', shap_plot=True, **args_seq)\n"
             "cpp_plot.profile(df_feat=df_feat, **ka)            # left\n"
             "cpp_plot.feature_map(df_feat=df_feat, name_test='APP', **ka)  # right\n"
             "plt.tight_layout(); plt.show()"},
    {"cls": "AAclust — clusters", "tag": "scale reduction · Wrapper", "img": "centers",
     "caption": "AAclustPlot.centers · cluster scale profiles",
     "code": "X = np.array(df_scales).T\n"
             "aac = aa.AAclust()\n"
             "aac.fit(X, names=list(df_scales), n_clusters=10)\n"
             "aac.medoid_names_   # redundancy-reduced scales\n"
             "\n"
             "aac_plot = aa.AAclustPlot()\n"
             "aac_plot.centers(X, labels=aac.labels_)\n"
             "plt.tight_layout(); plt.show()"},
    {"cls": "dPULearn — PCA", "tag": "reliable negatives · Wrapper", "img": "pca",
     "caption": "dPULearnPlot.pca · reliable negatives",
     "code": "# labels: 1 = positive, 2 = unlabeled\n"
             "n_pos = sum(np.array(labels) == 1)\n"
             "dpul = aa.dPULearn()\n"
             "dpul.fit(X=X, labels=labels, n_unl_to_neg=n_pos)\n"
             "df_pu = dpul.df_pu_   # 1 pos · 0 rel-neg · 2 unl\n"
             "\n"
             "dpul_plot = aa.dPULearnPlot()\n"
             "dpul_plot.pca(df_pu=df_pu, labels=labels)\n"
             "plt.tight_layout(); plt.show()"},
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

OPTIONS = (
    "aa.options['random_state'] = 42\n"
    "aa.options['verbose'] = True\n"
    "aa.options['allow_multiprocessing'] = True\n"
    "\n"
    "# TMD model — JMD flank widths\n"
    "aa.options['jmd_n_len'] = 10\n"
    "aa.options['jmd_c_len'] = 10\n"
    "\n"
    "# plot labels & system-level scales\n"
    "aa.options['name_tmd'] = 'P5-P5′'   # e.g. cleavage-site prediction\n"
    "aa.options['df_scales'] = my_scales"
)

# (class, plot class | "—", kind tag)
CLASS_PLOT = [
    ("SequencePreprocessor", "—", ""),
    ("EmbeddingPreprocessor", "—", "v1.1"),
    ("StructurePreprocessor  [pro]", "—", "v1.1"),
    ("AnnotationPreprocessor  [pro]", "—", "v1.1"),
    ("CPP", "CPPPlot", ""),
    ("AAclust", "AAclustPlot", "Wrapper"),
    ("AAlogo", "AAlogoPlot", ""),
    ("dPULearn", "dPULearnPlot", "Wrapper"),
    ("TreeModel", "—", "Wrapper"),
    ("ShapModel  [pro]", "—", "Wrapper"),
    ("AAMut", "AAMutPlot", "to be extended"),
    ("SeqMut", "SeqMutPlot", "to be extended"),
    ("AAWindowSampler", "—", ""),
    ("SequenceFeature", "—", ""),
    ("NumericalFeature", "—", ""),
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
     "journal": "Bioinformatics Advances"},
    {"name": "AAontology", "key": "[Breimann24b]",
     "ref": "Breimann et al. (2024b), AAontology: An ontology of amino acid "
            "scales for interpretable machine learning",
     "journal": "Journal of Molecular Biology"},
    {"name": "CPP & dPULearn", "key": "[Breimann25a]",
     "ref": "Breimann & Kamp et al. (2025), Charting γ-secretase substrates by "
            "explainable AI",
     "journal": "Nature Communications"},
]

GLOSSARY = [
    ("Prediction level", "Residue (AA_*) · Domain (DOM_*) · Protein (SEQ_*) — the "
     "unit a task predicts at; sets the dataset, the part, and the reference."),
    ("Part", "Named segment used as feature input: tmd, jmd_n, jmd_c, tmd_jmd, "
     "jmd_n_tmd_n, tmd_c_jmd_c."),
    ("Split", "How a scale is read across a part: Segment (contiguous), Pattern "
     "(sparse), PeriodicPattern (i, i+3/4)."),
    ("Scale", "AA → ℝ mapping. AAontology ships ~600 curated scales in two-level "
     "categories."),
    ("Feature", "(Part × Split × Scale) — the atomic, residue-grounded, "
     "interpretable unit of CPP."),
    ("Compositional vs positional", "How split_kws resolves locality: a whole-part "
     "average (compositional) vs sub-region/position-resolved (positional)."),
    ("df_seq", "Sequence table: entry, sequence, label, TMD bounds (tmd_start, "
     "tmd_stop)."),
    ("df_parts", "Wide table — one column per part (tmd, jmd_n, jmd_c, …)."),
    ("df_feat", "Ranked features: feature, abs_auc, mean_dif, p_val, positions, "
     "scale, category."),
    ("Wrapper", "sklearn-style class — .fit / .predict / .eval, sets trailing *_ "
     "attributes after fit."),
    ("Plot class", "*Plot mirror of an analytical class — same arguments, "
     "visualization only."),
    ("PU labels", "dPULearn input: 1 = positive, 2 = unlabeled. Output: "
     "1 / 0 (reliable-negative) / 2."),
    ("CPP", "Comparative Physicochemical Profiling — discovers ranked "
     "Part × Split × Scale features."),
    ("AAontology", "Two-level scale taxonomy; CPP uses its categories to organize "
     "and rank features."),
]

FOOTER_NOTE = ("Layered seeds: seed= (call) ▸ random_state= (init) ▸ "
               "aa.options['random_state'] ▸ default.")

CONTENT = {
    "meta": META,
    "golden_workflow": GOLDEN_WORKFLOW,
    "prediction_levels": PREDICTION_LEVELS,
    "feature_ontology": FEATURE_ONTOLOGY,
    "cpp_strategies": CPP_STRATEGIES,
    "sequence_anatomy": SEQUENCE_ANATOMY,
    "install": INSTALL,
    "which_module": WHICH_MODULE,
    "five_minute": FIVE_MINUTE,
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
