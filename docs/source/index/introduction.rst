Introduction
============
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
Start your journey with AAanalysis by visiting our :ref:`Contributing <contributing>` page for installation instructions
and information on contribution. Delve into the core concepts behind our algorithms in the
:ref:`Usage Principles <usage_principles>` section to understand our design philosophy, equipping
you with the mental models necessary for effective application. Our :ref:`Evaluation <evaluation>`
strategies are detailed to facilitate an transparent and objective analysis of our algorithms´s outcome.

To get hands-on experience, explore our :ref:`Tutorials <tutorials>`. In addition to our detailed :ref:`API <api>`
documentation, we compiled various overview :ref:`Tables <tables>` that provide in-depth resources, including our
**AAontology** scale classification and a variety of benchmark protein datasets. Discover the scientific foundation
of AAanalysis in the :ref:`References <references>` section.

.. comment::

    TODO if diagram is ready
    **Entry Points**:
    Our toolkit bridges seamlessly with external libraries, enhancing its versatility and integration capabilities
    in diverse research environments.

    [Link to entry point diagram]
