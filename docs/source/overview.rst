.. _overview:

**AAanalysis** (Amino Acid analysis) is a Python framework for interpretable sequence-based protein prediction,
providing the following algorithms:

- **AAclust**: k-optimized clustering wrapper framework to select redundancy-reduced sets of numerical scales (e.g., amino acid scales)
- **CPP**: Comparative Physicochemical Profiling, a feature engineering algorithm comparing two sets of protein sequences to identify the set of most distinctive features.
- **dPULearn**: deterministic Positive-Unlabeled (PU) Learning algorithm to enable training on unbalanced and small datasets.

Moreover, AAanalysis provides functions for loading protein benchmark datasets (**load_data**),
amino acid scale sets (**load_scales**), and their in-depth two-level classification (**AAontology**).
