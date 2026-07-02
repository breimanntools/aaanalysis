.. _usage_principles_xai:

Explainable AI with Single-Residue Resolution
=============================================

In the life sciences, AI and machine learning have significantly advanced protein
analysis, yet their full potential is held back by the black-box nature of traditional
models, which obscures how a prediction is made. Explainable AI (eXAI) closes this gap:
it not only forecasts an outcome but also surfaces the reasons for it, giving detailed
insight into protein function at the amino acid level. In AAanalysis this matters twice
over, because CPP features are already interpretable by construction, so an explanation
resolves to a concrete *Part × Split × Scale* signal rather than an anonymous input.

.. admonition:: Provided by
   :class: note

   In AAanalysis this is :class:`~aaanalysis.TreeModel` (predictions and *global* feature
   importance), :class:`~aaanalysis.ShapModel` (SHAP-based *local*, per-sequence feature
   impact), and :class:`~aaanalysis.CPPPlot` (the visualisations). See the
   :ref:`API reference <api>` for signatures and the :ref:`tutorials <tutorials>` for
   hands-on use.

What is explainable AI?
-----------------------
Explainable AI (eXAI) transforms opaque AI models into transparent systems, letting human
experts grasp the rationale behind specific predictions. This is particularly pivotal in
the life sciences, where understanding each amino acid's role can inform drug discovery
and disease treatment. Explanations come at two levels: *global* importance ranks which
features matter across a whole dataset (:class:`~aaanalysis.TreeModel`), while *local* attribution explains
one specific sequence, residue by residue (:class:`~aaanalysis.ShapModel`).

Combining CPP with SHAP
-----------------------
To explain machine-learning predictions for individual proteins at single-residue
resolution, AAanalysis combines CPP with the explainable-AI framework
`SHAP <https://shap.readthedocs.io/en/latest/index.html>`_. Because every CPP feature maps
back to a specific sequence part and physicochemical scale, a SHAP value can be projected
onto the exact residues it describes. AAanalysis exposes this through
:class:`~aaanalysis.CPPPlot`, which offers four complementary views: a group-level feature
map, a ranked feature profile, and per-sequence CPP-SHAP plots that colour the sequence by
each residue's contribution.

.. figure:: /_artwork/schemes/scheme_CPP_eAI.png

   Overview of explainable AI with single-residue resolution by combining CPP with SHAP,
   from [Breimann25]_.
