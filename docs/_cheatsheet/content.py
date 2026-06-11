"""This is the single source of truth for the AAanalysis cheat sheet content.

The same ``CONTENT`` dict drives both rendered outputs (``cheat_sheet.html`` and
``cheat_sheet.pdf``) via ``build_cheat_sheet.py`` -> ``template.html.jinja``, so the
two artifacts can never drift. Edit content here, then regenerate (see README.md).

Every code snippet uses only public ``aaanalysis.__all__`` symbols with real
signatures, and every term matches the canonical glossary in ``CONTEXT.md``.
"""

META = {
    "version": "v10",
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
    {"n": "3", "title": "FEATURES", "call": "CPP.run",
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
        "code": 'skw = sf.get_split_kws(\n'
                '    split_types="Segment",\n'
                '    n_split_min=1, n_split_max=1)\n'
                'cpp = aa.CPP(df_parts=df_parts,\n'
                '             split_kws=skw)',
    },
    "positional": {
        "name": "Positional", "maps": "≈ residue-/region-level",
        "desc": "sub-segments and/or patterns resolved to positions",
        "code": 'skw = sf.get_split_kws(\n'
                '    split_types=["Segment", "Pattern",\n'
                '                 "PeriodicPattern"],\n'
                '    n_split_max=5, steps_pattern=[3, 4],\n'
                '    steps_periodicpattern=[3, 4])\n'
                'cpp = aa.CPP(df_parts=df_parts,\n'
                '             split_kws=skw)',
    },
    "note": "Domain level uses both. → CPP strategies: see the CPP tutorial (docs).",
}

# Beginner decision flow: user intent -> the module to reach for.
WHICH_MODULE = [
    ("Explore sequence patterns / composition", "AAlogo"),
    ("Discover discriminative physicochemical features", "CPP"),
    ("Reduce redundant amino-acid scales", "AAclust"),
    ("Train with positives + unlabelled data", "dPULearn"),
    ("Train an interpretable classifier", "TreeModel"),
    ("Explain a prediction (per feature / sample)", "ShapModel  [pro]"),
    ("Visualize CPP features", "CPPPlot"),
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
               "2 unlabelled.",
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
    "df_scales = aa.load_scales()\n"
    "# TMD model: 20-aa TMD, short JMD flanks\n"
    "aa.options['jmd_n_len'] = 6\n"
    "aa.options['jmd_c_len'] = 6"
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
         ("Extract sequence parts", "get_df_parts(df_seq) → df_parts", None),
         ("Discover discriminative features", "CPP(df_parts).run(labels) → df_feat  ★", None),
         ("Simplify → interpretable scales", "CPP.simplify(df_feat, labels)", None),
         ("Build feature matrix X", "feature_matrix(df_feat, df_parts) → X", None),
         ("Reduce redundant scales", "AAclust().fit(X)  [Wrapper]", None),
         ("Drop correlated features", "filter_correlation(X)", None),
     ]},
    {"name": "Structural & Embedding", "tag": "alternative feature sources",
     "rows": [
         ("Fetch AlphaFold + encode 3D/DSSP/PAE", "StructurePreprocessor().encode_dssp(df_seq)", None),
         ("Encode protein-LM embeddings", "EmbeddingPreprocessor().encode(df_seq)", None),
         ("Embeddings → pseudo-scales", "EmbeddingPreprocessor().build_scales(df_seq)", None),
         ("Combine numeric feature dicts → CPP", "combine_dict_nums(dict_nums)", None),
     ]},
    {"name": "Modeling & Explainability", "tag": "",
     "rows": [
         ("Train + RFE + MC importance", "TreeModel().fit(X, labels)  [Wrapper]", None),
         ("Train with positives + unlabelled", "dPULearn().fit(X, labels)  [Wrapper]", None),
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
         ("Global plot style & fonts", "plot_settings(font_scale)", None),
         ("Colours & standalone legend", "plot_get_clist(n) · plot_legend(ax)", None),
     ]},
    {"name": "Protein Design", "tag": "",
     "under_construction": True,
     "rows": [
         ("In-silico point mutations", "AAMut · AAMutPlot", None),
         ("Sequence-design libraries", "SeqMut · SeqMutPlot", None),
     ]},
]

# Detailed recipes for the core analytical classes (page 2, right zone).
# Worked examples — tutorial-convention code paired with the figure it produces.
FLAGSHIP_RECIPES = [
    {"cls": "AAlogo — see the data", "tag": "dataset at a glance", "img": "logo",
     "code": "sf = aa.SequenceFeature()\n"
             "df_parts = sf.get_df_parts(df_seq=df_seq,\n"
             "    list_parts=['tmd', 'jmd_n', 'jmd_c'])\n"
             "\n"
             "aal = aa.AAlogo()\n"
             "df_logo = aal.get_df_logo(df_parts=df_parts,\n"
             "    labels=labels, label_test=1, tmd_len=20)\n"
             "df_info = aal.get_df_logo_info(df_parts=df_parts,\n"
             "    labels=labels, label_test=1, tmd_len=20)\n"
             "aa.plot_settings(font_scale=0.7)\n"
             "aa.AAlogoPlot().single_logo(df_logo=df_logo,\n"
             "    df_logo_info=df_info,        # bits bar on top\n"
             "    name_data='Test set: substrates')\n"
             "plt.tight_layout(); plt.show()"},
    {"cls": "CPP — feature map", "tag": "flagship · interpretability", "img": "feature_map",
     "big": True,
     "code": "# extended parts -> default split grid applies\n"
             "df_parts = sf.get_df_parts(df_seq=df_seq,\n"
             "    list_parts=['tmd', 'jmd_n_tmd_n', 'tmd_c_jmd_c'])\n"
             "cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales)\n"
             "df_feat = cpp.run(labels=labels, n_filter=40)\n"
             "# swap scales for more interpretable correlated ones\n"
             "df_feat = cpp.simplify(df_feat=df_feat, labels=labels)\n"
             "X = sf.feature_matrix(features=df_feat['feature'],\n"
             "                      df_parts=df_parts)\n"
             "tm = aa.TreeModel(); tm.fit(X, labels=labels)\n"
             "df_feat = tm.add_feat_importance(df_feat=df_feat)\n"
             "\n"
             "cpp_plot = aa.CPPPlot()\n"
             "aa.plot_settings(font_scale=0.65)\n"
             "cpp_plot.feature_map(df_feat=df_feat)\n"
             "plt.tight_layout(); plt.show()"},
    {"cls": "ShapModel — explain a prediction", "tag": "per-protein · [pro]", "img": "feature_map_shap",
     "big": True,
     "code": "se = aa.ShapModel()\n"
             "# fuzzy_labeling captures the true SHAP impact\n"
             "se.fit(X, labels=labels, fuzzy_labeling=True)\n"
             "# explain a borderline call — LRP6 (~60% substrate)\n"
             "i = list(df_seq['entry']).index('O75581')\n"
             "df_feat = se.add_feat_impact(df_feat=df_feat,\n"
             "              sample_positions=i, names='LRP6')\n"
             "\n"
             "cpp_plot.feature_map(df_feat=df_feat, shap_plot=True,\n"
             "    col_imp='feat_impact_LRP6', name_test='LRP6',\n"
             "    **args_seq)\n"
             "plt.tight_layout(); plt.show()"},
    {"cls": "AAclust — clusters", "tag": "scale reduction · Wrapper", "img": "centers",
     "code": "X = np.array(df_scales).T\n"
             "aac = aa.AAclust()\n"
             "aac.fit(X, names=list(df_scales), n_clusters=10)\n"
             "aac.medoid_names_   # redundancy-reduced scales\n"
             "\n"
             "aac_plot = aa.AAclustPlot()\n"
             "aac_plot.centers(X, labels=aac.labels_)\n"
             "plt.tight_layout(); plt.show()"},
    {"cls": "dPULearn — PCA", "tag": "reliable negatives · Wrapper", "img": "pca",
     "code": "# labels: 1 = positive, 2 = unlabelled\n"
             "dpul = aa.dPULearn()\n"
             "dpul.fit(X=X, labels=labels, n_unl_to_neg=n_pos)\n"
             "df_pu = dpul.df_pu_   # 1 pos · 0 rel-neg · 2 unl\n"
             "\n"
             "dpul_plot = aa.dPULearnPlot()\n"
             "dpul_plot.pca(df_pu=df_pu, labels=labels)\n"
             "plt.tight_layout(); plt.show()"},
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
    "aa.options['name_tmd'] = 'TMD'\n"
    "aa.options['df_scales'] = my_scales"
)

# (class, plot class | "—", kind tag)
CLASS_PLOT = [
    ("CPP", "CPPPlot", ""),
    ("AAclust", "AAclustPlot", "Wrapper"),
    ("AAlogo", "AAlogoPlot", ""),
    ("dPULearn", "dPULearnPlot", "Wrapper"),
    ("TreeModel", "—", "Wrapper"),
    ("ShapModel  [pro]", "—", "Wrapper"),
    ("AAMut", "AAMutPlot", "planned"),
    ("SeqMut", "SeqMutPlot", "planned"),
    ("AAWindowSampler", "—", ""),
    ("SequenceFeature", "—", ""),
    ("NumericalFeature", "—", ""),
    ("SequencePreprocessor", "—", ""),
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
    ("PU labels", "dPULearn input: 1 = positive, 2 = unlabelled. Output: "
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
    "options": OPTIONS,
    "class_plot": CLASS_PLOT,
    "design_principles": DESIGN_PRINCIPLES,
    "citations": CITATIONS,
    "glossary": GLOSSARY,
    "footer_note": FOOTER_NOTE,
}
