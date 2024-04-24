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

.. figure:: /_artwork/diagrams/connections.png
   :align: center
   :alt: AAanalysis workflow

   General pipeline for sequence-based protein prediction in Python with AAanalysis.
   Either amino acid scales or protein embeddings can be used as numerical representation of amino acids
   (indicated by dashed lines). Protein embeddings can be created
   via `Google Colab <https://colab.research.google.com/drive/1N3Sf5EDwqHEN2lyPNcW5w6Mct5FZ2-W2?usp=sharing>`_
   and are currently integrated with CPP.

AAanalysis provides a handful of DataFrames for seamless data management. Starting with amino acid scale information
(**df_scales**, **df_cat**) and protein sequences (**df_seq**), it enables segmentation into parts (**df_parts**)
and accommodates user-defined splitting (**split_kws**). Our CPP algorithm then utilizes these to generate
physicochemical features (**df_feat**) by comparing protein sequence sets.

See the primary analysis pipeline of the AAanalysis framework in this diagram:

.. figure:: /_artwork/diagrams/components.png
   :align: center
   :alt: AAanalysis dataflow

   AAanalysis pipeline illustrating the typical data flow, represented as data frames, with key
   methods (Python classes) highlighted by black squares.

Details on the foundational concepts of AAnalysis are provided by the following sections:

.. toctree::
   :maxdepth: 1

   usage_principles/aaontology
   usage_principles/aaclust
   usage_principles/feature_identification
   usage_principles/pu_learning
   usage_principles/xai
