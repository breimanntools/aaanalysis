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

Once you have a first result, learn individual tools in the
:ref:`Tutorials <tutorials>`, design a valid analysis with the
:ref:`Protocols <protocols>`, and look up exact parameters in the
:doc:`API <api>`.

.. toctree::
   :maxdepth: 1

   generated/tutorial0_minimal
   generated/tutorial1_quick_start
   generated/tutorial1_slow_start
   generated/plotting_prelude
