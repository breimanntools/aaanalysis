Explainable AI with Single-Residue Resolution
=============================================

In life sciences, AI and machine learning have significantly advanced protein analysis, yet their full potential
is hindered by the black-box nature of traditional models, which obscures the understanding of how predictions are made.
This gap is bridged by explainable AI (eXAI), which not only forecasts outcomes but also demystifies the underlying
reasons, offering detailed insights into protein functions at the amino acid level.


What is explainable AI?
-----------------------
Explainable AI (eXAI) transforms opaque AI models into transparent systems, enabling human experts to grasp the rationale
behind specific predictions. This is particularly pivotal in life sciences, where comprehending each amino acid's role
can revolutionize drug discovery and disease treatment.

Combining CPP with SHAP
-----------------------
To explain machine learning-based prediction results for individual proteins with single-residue resolution,
we combined CPP with the explainable AI framework `SHAP <https://shap.readthedocs.io/en/latest/index.html>`_.
AAanalysis offers four different visualizations:

.. image :: /_artwork/schemes/scheme_CPP_eAI.png