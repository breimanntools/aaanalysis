.. _ecosystem:

The AAanalysis Ecosystem
========================
AAanalysis is the interpretable middle layer between bioinformatics I/O and the downstream machine
learning, explainable AI, and protein-design stack. It *consumes* upstream representations (sequences,
embeddings, structures) and even competitor descriptor sets, runs them through its interpretable core
(*Part × Split × Scale* · AAontology · CPP · ShapModel), and *exposes* the resulting features,
explanations, and objectives to the standard ML / XAI / optimization tools.

.. figure:: /_artwork/diagrams/aaanalysis_ecosystem.svg
   :alt: The AAanalysis ecosystem
   :width: 100%
   :align: center

   Where AAanalysis fits in the protein-ML stack — upstream complements feed the interpretable
   core, which exposes features, explanations, and objectives downstream. Package marks use each
   project's brand colors; the center carries the real AAanalysis logo.

Positioning brief
-----------------
The honest question for any new tool is whether it reinvents the wheel. AAanalysis does **not**
reinvent Biopython (bioinformatics I/O), scikit-learn (general ML), SHAP (model explanation), or
PyTorch / ESM (deep models and embeddings) — it sits between them, consuming their outputs and feeding
the next layer. The one place it genuinely overlaps is classical *protein-descriptor* libraries such
as iFeature, iFeatureOmega, propy3, and PyBioMed.

But it is a different axle on the same wheel. Descriptor libraries map a sequence to a large catalogue
of generic descriptors and stop. AAanalysis instead frames a biological task (residue, domain, or
protein level), constructs a test-versus-reference comparison, and discovers *Part × Split × Scale*
features through Comparative Physicochemical Profiling (CPP): each feature states **where** on the
sequence a signal sits, **how** that region is read, and **which** physicochemical property it
captures — then ranks them contrastively and explains them at single-residue resolution.

So the claim is deliberately narrow and defensible. AAanalysis is not broader than iFeature or
PyBioMed; it is more interpretable, task-aware, position-resolved, and small-data-aware (PU learning,
scarce negatives), with built-in biological plots and SHAP-based attribution. It is *small but sharp* —
it cannot win on breadth against the ecosystem giants, only on the interpretability axis that matters
for protein prediction.

Two relationships keep the map honest. A **complement** sits upstream or downstream and exchanges data;
a **comparison** occupies the same functional role and is a benchmark candidate. Most of the ecosystem
is complementary — only descriptor libraries are genuine comparisons. Downstream, AAanalysis is
positioned to feed ML and optimization and to make XAI-evaluation (Quantus / OpenXAI), causal
(DoWhy / EconML), uncertainty (MAPIE) and design (RFdiffusion / ProteinMPNN / PyRosetta) layers
biologically readable; these are shown as candidate / future integrations, not current core.

Categories and packages
-----------------------
The map groups the Python ecosystem by the role each project plays relative to the AAanalysis core.
The numbered labels match the nodes in the diagram above.

Upstream complements
~~~~~~~~~~~~~~~~~~~~~~
**1 · Biological data & I/O** *(upstream)*
    *Biopython · Biotite · bioservices · gget* — FASTA / PDB I/O and UniProt / NCBI access. Records
    are consumed as ``df_seq``; AAanalysis builds thin adapters, not a parser.

**2 · Protein representations** *(upstream)*
    *fair-esm · transformers (ProtT5) · bio-embeddings · Bio.PDB* — embeddings and structure become
    position-aware pseudo-scales via ``run_num``. Named physicochemical scales stay the most directly
    interpretable layer.

The comparison axis
~~~~~~~~~~~~~~~~~~~~~
**3 · Protein feature descriptors** *(comparison)*
    *iFeature · propy3 · PyBioMed* — the only genuine competitors. They enumerate descriptor vectors;
    AAanalysis discovers task-aware, position-aware, explanation-ready features. This is the honest
    axis to benchmark on.

The interpretable core
~~~~~~~~~~~~~~~~~~~~~~~~
**4 · AAanalysis** *(core)*
    *CPP · SequenceFeature · AAclust · dPULearn · TreeModel · ShapModel · CPPPlot* — task framing,
    test-vs-reference construction, *Part × Split × Scale* feature discovery, PU learning for missing
    negatives, biological explanation, and ΔCPP design steering.

Downstream consumers
~~~~~~~~~~~~~~~~~~~~~~
**5 · ML / DL models** *(downstream)*
    *scikit-learn · XGBoost · LightGBM · PyTorch* — train on the AAanalysis feature matrix. Target:
    sklearn-compatible transformers so CPP features drop into standard pipelines.

**6 · Explainability (XAI)** *(downstream)*
    *SHAP · Captum · LIME · DiCE* — generic attribution becomes biologically readable on CPP features:
    "JMD-C charge pattern raised substrate prediction" rather than "feature_17 +0.21".

**8 · XAI evaluation** *(downstream)*
    *Quantus · OpenXAI* — scores explanation quality (faithfulness, robustness, localization,
    complexity, randomization checks). Lets AAanalysis show its CPP and ShapModel attributions are
    faithful and stable, not just plausible.

**9 · Causal inference** *(downstream)*
    *DoWhy · EconML (PyWhy)* — CPP finds what distinguishes two groups; causal tools test what drives
    the outcome. Turns correlated discriminative features into refutable causal hypotheses.

**7 · Optimization & design** *(downstream)*
    *Optuna · pymoo · DEAP* — tune CPP / model settings and run multi-objective design. AAanalysis
    exposes objective functions and explains why a candidate improved (ΔCPP), rather than replacing
    the optimizers.

Side branch and cross-cutting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Proteomics / MS** *(side branch, optional)*
    *pyteomics · pyopenms* — MS data access and peptide mass / charge / pI. Relevant to the flyability
    use case: predict and explain sequence determinants downstream of MS workflows.

**Model validation** *(cross-cutting)*
    *sklearn.metrics · MLflow · MAPIE* — AAanalysis owns the protein-specific protocols: homology-aware
    splits, same-protein leakage checks, shuffled-label controls, feature stability, per-protein AP.

Comparison matrix
-----------------
.. list-table::
   :header-rows: 1
   :widths: 4 20 14 30 32

   * - \#
     - Category
     - Relationship
     - Python packages
     - What AAanalysis adds / why it complements
   * - 1
     - Biological data & I/O
     - upstream
     - Biopython, Biotite, bioservices, gget
     - File / database / structure plumbing; records consumed as ``df_seq``.
   * - 2
     - Protein representations
     - upstream
     - fair-esm, transformers (ProtT5), bio-embeddings, Bio.PDB
     - Makes embeddings / structure position-aware via ``run_num``; flags lower interpretability of raw dims.
   * - 3
     - Protein feature descriptors
     - comparison
     - iFeature, propy3, PyBioMed
     - The benchmark axis. Task-aware, position-aware, explanation-ready CPP features vs broad enumeration.
   * - 4
     - AAanalysis
     - core
     - CPP, SequenceFeature, AAclust, dPULearn, TreeModel, ShapModel, CPPPlot
     - The interpretable middle layer: framing, reference construction, feature discovery, PU learning, explanation, design.
   * - 5
     - ML / DL models
     - downstream
     - scikit-learn, XGBoost, LightGBM, PyTorch
     - Train on the feature matrix; target is sklearn-pipeline compatibility.
   * - 6
     - Explainability (XAI)
     - downstream
     - SHAP, Captum, LIME, DiCE
     - Makes attribution biologically meaningful; separates group-level importance from per-sample impact.
   * - 8
     - XAI evaluation
     - downstream
     - Quantus, OpenXAI
     - Scores faithfulness / robustness of CPP & ShapModel explanations — trustworthy, not just plausible.
   * - 9
     - Causal inference
     - downstream
     - DoWhy, EconML (PyWhy)
     - Turns correlated discriminative features into refutable causal hypotheses about drivers.
   * - 7
     - Optimization & design
     - downstream
     - Optuna, pymoo, DEAP
     - Exposes objective functions; design as interpretable scoring, filtering, and ΔCPP steering.
   * - –
     - Proteomics / MS
     - side branch
     - pyteomics, pyopenms
     - Optional peptide / flyability workflows downstream of MS data access.
   * - –
     - Model validation
     - cross-cutting
     - sklearn.metrics, MLflow, MAPIE
     - AAanalysis owns protein-specific protocols; uses generic tooling for metrics and tracking.

Only category 3 is a direct comparison; every other row is a complement that AAanalysis either consumes
from or feeds into.

Project scale
-------------
Across the ecosystem, approximate GitHub stars and order-of-magnitude source size group cleanly by role:
the upstream and downstream giants (Biopython, scikit-learn, PyTorch, SHAP) are orders of magnitude
larger than AAanalysis, while the genuine comparators — the descriptor libraries — are closest in scale.
The takeaway is the framing of this whole map: AAanalysis is *small but sharp*, with descriptor libraries
as its only direct comparators, competing on the interpretability axis rather than on breadth.

Strategic summary
-----------------
AAanalysis is not innovative because it computes protein features — iFeature, propy3, PyBioMed, AAindex
tools, and PLM-embedding workflows already do that. It is innovative as a **task-aware, contrastive,
biologically interpretable protein-feature discovery layer**. Its biggest gain is not "more descriptors"
but turning small, messy protein datasets into residue- and region-aware, test-versus-reference
explanations that biologists can read and ML pipelines can use.

**Where it is strongest:** small-data protein prediction (~20–500 positives); missing-negative settings
(dPULearn turns unlabeled backgrounds into usable references without pretending they are true negatives);
the test-vs-reference contrast that mirrors how biologists think (substrates vs non-substrates, detected
vs undetected peptides, PTM sites vs non-sites); biological readability of XAI (not "feature_137 is
important" but "JMD-C × pattern × charge raises substrate prediction"); and emerging ΔCPP design steering.

**Where it should not compete:** descriptor breadth (iFeature / PyBioMed), generic ML (scikit-learn /
XGBoost / LightGBM), deep learning (PyTorch / ESM / ProtT5), general XAI theory (SHAP / Captum / Quantus),
and MS processing (pyOpenMS / pyteomics). It integrates with these rather than replacing them — small but
sharp.

**One line.** AAanalysis is most innovative as a bridge: the interpretable, contrastive, task-aware
protein-feature layer that makes protein-prediction explanations actionable for biology — which region,
which residue pattern, and which physicochemical / pseudo-scale property distinguishes a functional group
or drives a prediction.

----

A designed, standalone version of this map (with brand icons and the project-scale figure) is also kept
as a printable `single-page layout <../_static/aaanalysis_ecosystem.html>`_.
