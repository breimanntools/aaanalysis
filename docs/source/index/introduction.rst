Introduction
============
**AAanalysis** is a Python framework designed for scientists and researchers focusing on interpretable sequence-based
protein prediction. Ideal for comparing protein sequences using amino acid scales, this toolkit is versatile enough
for any sequence analysis representable by numerical values.

Purpose and Audience
--------------------
Designed for biologists and computational biologists, AAanalysis streamlines the process of analyzing and comparing
protein sequences. By leveraging this framework, physicochemical signatures can be uncovered that are key to understanding
biological interactions and functions.

Overview of Documentation
-------------------------
Begin with AAanalysis by visiting our :ref:`contributing page <contributing>` for installation instructions
and information on contribution. Delve into the guiding principles and design philosophy of our key algorithms in the
:ref:`usage principles section <usage_principles>`. To get hands-on experience, explore our :ref:`tutorials <tutorials>`.
In addition to our detailed :ref:`API documentation <api>`, we compiled various :ref:`overview tables <tables>`
providing in-depth resources, including **AAontology**—our unique two-level classification of amino acid scales—and
various benchmark protein datasets.

AAanalysis Workflow
-------------------
Our typical workflow comprises the following key algorithms:

- **AAclust**: A k-optimized clustering wrapper that selects redundancy-reduced sets of numerical scales, such as amino acid scales.
- **CPP**: Comparative Physicochemical Profiling, an interpretable feature engineering algorithm comparing two sets of
  protein sequences to identify the set of most distinctive features.
- **dPULearn**: A deterministic Positive-Unlabeled (PU) Learning algorithm tailored for training on unbalanced and
  small datasets, enhancing predictive accuracy.

Data Flow and Entry Points
--------------------------
**Data Flow**:
AAanalysis provides a handful of DataFrames for seamless data management. Starting with amino acid scale information
(**df_scales**, **df_cat**) and protein sequences (**df_seq**), it enables segmentation into parts (**df_parts**)
and accommodates user-defined splitting (**split_kws**). Our CPP algorithm then utilizes these to generate
physicochemical features (**df_feat**) by comparing protein sequence sets.

See the primary data flow within the AAanalysis toolkit in this diagram:

.. image:: /_artwork/diagrams/components.png

**Entry Points**:
Our toolkit bridges seamlessly with external libraries, enhancing its versatility and integration capabilities
in diverse research environments.

[Link to entry point diagram]
