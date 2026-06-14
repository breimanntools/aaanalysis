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
   core, which exposes features, explanations, and objectives downstream.

Small but sharp
---------------
AAanalysis does **not** reinvent Biopython (bioinformatics I/O), scikit-learn (general ML), SHAP
(model explanation), or PyTorch / ESM (deep models and embeddings) — it sits between them, consuming
their outputs and feeding the next layer. The one place it genuinely overlaps is classical
*protein-descriptor* libraries such as iFeature, iFeatureOmega, propy3, and PyBioMed — but it is a
different axle on the same wheel. Descriptor libraries map a sequence to a large catalogue of generic
descriptors and stop. AAanalysis instead frames a biological task (residue, domain, or protein level),
constructs a test-versus-reference comparison, and discovers *Part × Split × Scale* features through
Comparative Physicochemical Profiling (CPP): each feature states **where** on the sequence a signal
sits, **how** that region is read, and **which** physicochemical property it captures — then ranks
them contrastively and explains them at single-residue resolution.

So the claim is deliberately narrow and defensible: AAanalysis is not broader than iFeature or
PyBioMed, it is more interpretable, task-aware, position-resolved, and small-data-aware. Two
relationships keep the map honest — a *complement* sits upstream or downstream and exchanges data;
a *comparison* occupies the same functional role and is a benchmark candidate. Most of the ecosystem
is complementary; only descriptor libraries are genuine comparisons.

Read the full ecosystem article
-------------------------------
The interactive `ecosystem article <../_static/aaanalysis_ecosystem.html>`_ continues from here with
the per-method maturity and adoption status behind the diagram: the full positioning brief, a
category-by-category package breakdown (upstream I/O, protein representations, descriptors, ML/DL,
XAI, optimization and design), a complement-vs-comparison matrix, the project-scale figure behind the
"small but sharp" framing, and a strategic summary of where AAanalysis is headed.
