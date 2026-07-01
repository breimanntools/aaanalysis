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

.. admonition:: Not sure where to start? Follow the Decision Map
   :class: tip

   Skim the :ref:`Decision Map <gs_decision_map>` at the bottom of this page: it maps
   your goal (*explore*, *predict*, or *optimize*) to the exact AAanalysis class or
   function. Then run the notebooks below for your first result.

The fastest way in is the short notebooks below:

- :doc:`A minimal CPP analysis </generated/tutorial0_minimal>`
- :doc:`Quick start with AAanalysis </generated/tutorial1_quick_start>`
- :doc:`Slow start with AAanalysis </generated/tutorial1_slow_start>`
- :doc:`Plotting Prelude </generated/plotting_prelude>`

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

- **Building blocks** (``import aaanalysis as aa``): the individual objects and
  functions you compose, for full control over each step. See :ref:`API <api>`.
- **Golden pipelines** (``import aaanalysis.pipe as aap``): stateless one-call
  wrappers that chain those building blocks into a sensible default workflow.
  See :ref:`API (Pipelines) <api_pipe>`.

.. code-block:: python

    import aaanalysis as aa            # explicit building blocks
    import aaanalysis.pipe as aap      # golden pipelines

.. _gs_decision_map:

Decision Map
------------
The Decision Map lays out the whole framework as a single flowchart: from your goal
(*explore*, *predict*, or *optimize*) down to the exact AAanalysis class or function to
call, including the CPP feature-engineering panel. Use it to find the right tool for
your question; click it to open the full-size version.

.. raw:: html

   <a href="_static/decision_map.html" target="_blank" rel="noopener" title="Open the Decision Map">
     <img src="_static/decision_map.png" alt="AAanalysis Decision Map (click to open the full map)"
          style="width:70%; display:block; margin:0 auto; border:1px solid #e3e7ec; border-radius:4px;">
   </a>
