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
Designed for (computational) biologists, AAanalysis facilitates the analysis and comparison of protein sequences.
It enables the discovery of physicochemical signatures that underlie biological interactions and functions.

Overview of Documentation
-------------------------
The documentation is organized into four sections — *Overview*, *Guides*, *Reference*, and *Project*.

**Overview** — New to AAanalysis? Begin with :ref:`Getting Started <getting_started>` to install
the package and run your first analysis. Delve into the core concepts and design philosophy behind
the algorithms in the :ref:`Usage Principles <usage_principles>` section, equipping you with the
mental models necessary for effective application, and see the :ref:`Evaluation <evaluation>`
strategies for a transparent, objective analysis of the algorithms´ outcomes.

**Guides** — For hands-on experience, the :ref:`Tutorials <tutorials>` teach each tool with its
parameters and outputs, while the :ref:`Protocols <protocols>` walk through complete, end-to-end
analyses for biological questions.

**Reference** — Look up the exact signatures in the :ref:`API <api>` documentation, browse the
overview :ref:`Tables <tables>` (including the **AAontology** scale classification and benchmark
protein datasets), check the :ref:`Glossary <glossary>` of key terms, and discover the scientific
foundation of AAanalysis in the :ref:`References <references>` section.

**Project** — Development conventions and how to contribute live in the
:ref:`Contributing <contributing>` guide.

To see how AAanalysis fits among related bioinformatics, machine-learning, and explainability
tools, explore the `AAanalysis Ecosystem <../_static/aaanalysis_ecosystem.html>`_.