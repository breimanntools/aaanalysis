..
   Developer Notes:
   This is the index file that outlines the usage principles for the AAanalysis package.
   Files for individual usage principles are stored in the /usage_principles directory.

   This document provides an overview of:
   - Component diagram (illustrating internal dependencies)

   Instead of including comprehensive tables here, refer to tables in tables.rst with concise explanations.
   Always include brief code examples that mirror the corresponding usage examples.
..

.. _usage_principles:

Usage Principles
================
To get started with AAanalysis, import it as follows:

.. code-block:: python

    import aaanalysis as aa

AAanalysis streamlines a Python-based machine learning workflow for protein prediction, starting with protein
sequences typically retrieved from `UniProt <https://www.uniprot.org/>`_ and assessed for similarity by
`Biopython´s <https://biopython.org/>`_ functionalities. It processes redundancy-reduced sets of
these sequences to delineate their most discriminative features for machine learning prediction using
`scikit-learn <https://scikit-learn.org/stable/>`_.
For enhanced interpretability, AAanalysis integrates with the SHapley Additive exPlanations
(`SHAP <https://shap.readthedocs.io/en/latest/index.html>`_) framework to provide detailed explanations of prediction
results for individual sequences at single-residue resolution.

For a bird's-eye view of how AAanalysis fits between upstream bioinformatics I/O
(UniProt, Biopython, protein-language-model embeddings, structures) and the downstream
machine-learning, explainable-AI, and protein-design stack, see
:ref:`The AAanalysis Ecosystem <ecosystem>`. Either amino acid scales or protein
embeddings can serve as the numerical representation of amino acids; embeddings can be
created via
`Google Colab <https://colab.research.google.com/drive/1N3Sf5EDwqHEN2lyPNcW5w6Mct5FZ2-W2?usp=sharing>`_
and are integrated with CPP through ``CPP.run_num``.

AAanalysis provides a handful of DataFrames for seamless data management. Starting with amino acid scale information
(**df_scales**, **df_cat**) and protein sequences (**df_seq**), it enables segmentation into parts (**df_parts**)
and accommodates user-defined splitting (**split_kws**). Our CPP algorithm then utilizes these to generate
physicochemical features (**df_feat**) by comparing protein sequence sets.

See the primary analysis pipeline of the AAanalysis framework in the **Data Flow Map**
below: it runs from the external data sources through the two CPP entry points (``CPP.run`` for
amino acid scales, ``CPP.run_num`` for numeric values) to the feature matrix ``X`` and the
model, explanation, and design wrappers
(`open the full map <../_static/dataflow_map.html>`_):

.. figure:: /_artwork/diagrams/dataflow_map.png
   :align: center
   :alt: AAanalysis Data Flow Map
   :width: 100%

   The AAanalysis Data Flow Map. External data sources (gray; protein sequences,
   embeddings, structures, and annotations) feed the interpretable CPP core (blue),
   which turns them into the feature signature ``df_feat`` and the feature matrix
   ``X``. The wrapper classes (amber) then predict, explain, and design from ``X``.
   The map itself spells out every intermediate step.

New here? Start with **Prediction tasks**, the concept-overview page that maps a
biological question to the right AAanalysis workflow — by *unit of comparison* and
*reference construction*, not biological scale alone. Details on the foundational
concepts of AAanalysis are provided by the following sections:

.. toctree::
   :maxdepth: 1

   usage_principles/prediction_tasks
   usage_principles/aaontology
   usage_principles/aaclust
   usage_principles/feature_identification
   usage_principles/pu_learning
   usage_principles/xai
