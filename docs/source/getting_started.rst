..
   Developer Notes:
   The paths to notebooks are relative to ensure compatibility with the Sphinx referencing system
   used throughout the documentation. Getting Started is the onboarding pillar: only the first
   success belongs here (install + first result + publication-ready plots), no deep theory — that
   is what the Tutorials, Protocols, and Use Cases pillars are for.
..


.. _getting_started:

Getting Started
===============
New to AAanalysis? Start here. Get a first result with the **Quick start**
below, then learn how to choose between the two interfaces. The
**A minimal CPP analysis** notebook is the shortest complete loop — load a
dataset, run CPP, read out the signature — and pairs with the
:ref:`Prediction tasks <prediction_tasks>` concept page. For a fuller
introduction, our **Quick start** and **Slow start** tutorials share the same
examples, the latter adding the conceptual background, and the
**Plotting Prelude** tutorial helps you create publication-ready plots.

Not sure which tool fits your question? The **Decision Map** turns
*"what do I want to do?"* into a concrete path through the AAanalysis API —
**explore** sequences and groups, **build** features to predict and explain, or
**optimize** for protein design. It is the fastest way to find the right starting
point; click the map to open it full-size.

.. raw:: html

   <a href="_static/decision_map.html" target="_blank" rel="noopener" title="Open the Decision Map">
     <img src="_static/decision_map.png" alt="AAanalysis Decision Map"
          style="width:100%; display:block; margin:0 auto; border:1px solid #e3e7ec; border-radius:4px;">
   </a>

These short notebooks walk through that first loop step by step — start with
**A minimal CPP analysis**, compare the two interfaces in **Quick start** and
**Slow start**, then make publication-ready figures with **Plotting Prelude**:

.. toctree::
   :maxdepth: 1

   generated/tutorial0_minimal
   generated/tutorial1_quick_start
   generated/tutorial1_slow_start
   generated/plotting_prelude

Quick start
-----------
The shortest complete loop: load a benchmark dataset, run CPP, and read out the
feature signature.

.. code-block:: python

    import aaanalysis as aa

    df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
    labels = df_seq["label"].to_list()

    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq)
    cpp = aa.CPP(df_parts=df_parts)
    df_feat = cpp.run(labels=labels)            # the CPP signature

``df_feat`` is the CPP signature, the interpretable feature table that the rest
of the workflow (modelling, explanation, plotting) builds on. The notebooks
above walk through this loop step by step.

The two APIs
------------
AAanalysis offers the same analysis two ways, both documented in the Reference:

- **Building blocks** (``import aaanalysis as aa``) — the individual objects and
  functions you compose, for full control over each step. See :ref:`API <api>`.
- **Golden pipelines** (``import aaanalysis.pipe as aap``) — stateless one-call
  wrappers that chain those building blocks into a sensible default workflow.
  See :ref:`API (Pipelines) <api_pipe>`.

.. code-block:: python

    import aaanalysis as aa            # explicit building blocks
    import aaanalysis.pipe as aap      # golden pipelines
