..
   Developer Notes:

    LEGACY-TODO if diagram is ready
    **Entry Points**:
    Our toolkit bridges seamlessly with external libraries, enhancing its versatility and integration capabilities
    in diverse research environments.

    [Link to entry point diagram]
..

Introduction
============

.. image:: /_artwork/logos/model_AAanalysis.png
   :alt: AAanalysis Model Overview
   :align: center
   :class: aa-model
   :width: 100%

**AAanalysis** is a Python framework designed for scientists and researchers focusing on interpretable sequence-based
protein prediction. Ideal for comparing protein sequences using amino acid scales, this toolkit is versatile enough
for any sequence analysis representable by numerical values.

Key Algorithms
--------------
- **CPP**: Comparative Physicochemical Profiling, an interpretable feature engineering algorithm comparing two sets of
  protein sequences to identify the set of most distinctive features.
- **dPULearn**: A deterministic Positive-Unlabeled (PU) Learning algorithm tailored for training on unbalanced and
  small datasets, enhancing predictive accuracy.
- **AAclust**: A k-optimized clustering wrapper that selects redundancy-reduced sets of numerical scales,
  such as amino acid scales.

Purpose and Audience
--------------------
For computational biologists, bioinformaticians, and protein engineers, AAanalysis facilitates the analysis and
comparison of proteins to discover interpretable physicochemical signatures, the features that distinguish groups of
proteins and underlie their biological interactions and functions. These signatures span the whole workflow, from
simple sequence analysis to interpretable protein prediction and protein engineering, integrating state-of-the-art
explainable AI (XAI) methods.

Overview of Documentation
-------------------------
The documentation is organized into four sections: *Overview*, *Guides*, *Reference*, and *Project*.

**Overview**: New to AAanalysis? Begin with :ref:`Getting Started <getting_started>` to install
the package and run your first analysis. Delve into the core concepts and design philosophy behind
the algorithms in the :ref:`Usage Principles <usage_principles>` section, equipping you with the
mental models necessary for effective application — including the evaluation strategies for a
transparent, objective analysis of the algorithms´ outcomes.

**Guides**: For hands-on experience, the :ref:`Tutorials <tutorials>` teach each tool with its
parameters and outputs, the :ref:`Protocols <protocols>` walk through complete, end-to-end
analyses for biological questions, and the :ref:`Use Cases <use_cases>` showcase published
studies end to end from bundled data.

**Reference**: Look up the exact signatures in the :ref:`API <api>` documentation, or reach for the
one-call golden pipelines in :ref:`API (Pipelines) <api_pipe>`. Browse the overview
:ref:`Data Tables <tables>` (including the **AAontology** scale classification and benchmark protein
datasets) and the :ref:`DataFrame schemas <df_schemas>` that define every ``df_*`` contract, check the
:ref:`Glossary <glossary>` of key terms, and discover the scientific foundation of AAanalysis in the
:ref:`Scientific References <references>` section.

**Project**: Development conventions and how to contribute live in the
:ref:`Contributing <contributing>` guide, the :ref:`Docstring Guide <docstring_guide>` documents the
docstring style, and the :ref:`Release Notes <release_notes>` track changes across versions.

Finally, four at-a-glance reference documents summarise the whole framework — keep them open while
you work:

- `Cheat Sheet <../_static/cheat_sheet.html>`_ — the canonical workflow, the main classes by
  capability, and the *Part × Split × Scale* feature ontology on three pages.
- `Decision Map <../_static/decision_map.html>`_ — a flowchart from your goal (*explore*, *predict*,
  or *optimize*) to the exact AAanalysis class or function to call.
- `Ecosystem Map <../_static/ecosystem_map.html>`_ — where AAanalysis fits among related
  bioinformatics, machine-learning, and explainability tools (or read the full
  `positioning article <../_static/aaanalysis_ecosystem.html>`_ with the map plus background).
- `Data Flow Map <../_static/dataflow_map.html>`_ — how data flows from sequences and scales through
  CPP to features, models, and explanations.