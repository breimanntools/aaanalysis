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

``df_feat`` is the CPP signature — the interpretable feature table that the rest
of the workflow (modelling, explanation, plotting) builds on. The **Quick
start** notebook below walks through this loop step by step.

Two interfaces: ``aa`` and ``aap``
----------------------------------
AAanalysis offers the same analysis two ways — the explicit building blocks and
the golden pipelines:

.. code-block:: python

    import aaanalysis as aa            # explicit building blocks — full control
    import aaanalysis.pipe as aap      # golden pipelines — one-call workflows

See the :ref:`API reference <api>` for the difference between the two and when
to reach for each.

.. toctree::
   :maxdepth: 1

   generated/tutorial0_minimal
   generated/tutorial1_quick_start
   generated/tutorial1_slow_start
   generated/plotting_prelude
