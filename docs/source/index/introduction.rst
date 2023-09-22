Introduction
============

**AAanalysis** (Amino Acid analysis) is a Python framework for interpretable sequence-based protein prediction.
It was developed to compare two sets of protein sequences using amino acids scales, but can be generally used
for any (biological) sequence which can be represented by numerical values.

Workflow
--------
A typical workflow consists of the following steps:

- **AAclust**: k-optimized clustering wrapper framework to select redundancy-reduced sets of numerical scales (e.g., amino acid scales)
- **CPP**: Comparative Physicochemical Profiling, a feature engineering algorithm comparing two sets of protein sequences to identify the set of most distinctive features.
- **dPULearn**: deterministic Positive-Unlabeled (PU) Learning algorithm to enable training on unbalanced and small datasets.

Overview of documentation
-------------------------
See examples and practical usage in our :ref:`tutorials <tutorials>`.