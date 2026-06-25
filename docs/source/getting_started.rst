..
   Developer Notes:
   The paths to notebooks are relative to ensure compatibility with the Sphinx referencing system
   used throughout the documentation. Getting Started is the onboarding pillar: only the first
   success belongs here (install + first result + publication-ready plots), no deep theory — that
   is what the Tutorials and Protocols pillars are for.
..


.. _getting_started:

Getting Started
===============
New to AAanalysis? Start here. The **A minimal CPP analysis** notebook is the
shortest complete loop — load a dataset, run CPP, read out the signature — and
pairs with the :ref:`Prediction tasks <prediction_tasks>` concept page. For a
fuller introduction, explore our **Quick start** and **Slow start** tutorials,
both offering the same examples, with the latter explaining the conceptual
background. The **Plotting Prelude** tutorial can help you create
publication-ready plots.

Two interfaces: ``aa`` and ``aap``
----------------------------------
AAanalysis offers the same analysis two ways. The **explicit** interface
(``import aaanalysis as aa``) exposes every object and step for full control.
The **golden-pipeline** interface (``import aaanalysis.pipe as aap``) wraps the
standard ``load → CPP → model → explain → plot`` flow into a single call.

.. code-block:: python

    # Explicit — full control over every step
    import aaanalysis as aa

    df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
    labels = df_seq["label"].to_list()

    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq)
    cpp = aa.CPP(df_parts=df_parts)
    df_feat = cpp.run(labels=labels)            # the CPP signature

.. code-block:: python

    # Golden pipeline — a result in one call
    import aaanalysis as aa
    import aaanalysis.pipe as aap

    df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
    labels = df_seq["label"].to_list()

    df_feat, ax, df_eval = aap.find_features(df_seq=df_seq, labels=labels)

Both produce a ``df_feat`` CPP signature. The explicit path exposes every step
to customise; ``aap.find_features`` additionally runs a cross-validated feature
search and returns an evaluation table — all in one call. Reach for ``aap`` to
move fast, and drop to ``aa`` for full control. See the
:ref:`API reference <api>` for both tiers.

.. toctree::
   :maxdepth: 1

   generated/tutorial0_minimal
   generated/tutorial1_quick_start
   generated/tutorial1_slow_start
   generated/plotting_prelude
