..
   Developer Notes:
   The paths to tutorials are relative to ensure compatibility with the Sphinx referencing system
   used throughout the documentation.
..


.. _tutorials:

Tutorials
=========
Tutorials teach the AAanalysis **tools** — what each one does, its parameters, and
the outputs it returns. They cover the *mechanics*; for how to combine tools into a
valid end-to-end analysis, see the :ref:`Protocols <protocols>`, which link back
here for the mechanics instead of repeating them — so the two stay distinct with no
overlap. New to AAanalysis? Begin with :ref:`Getting Started <getting_started>` for
your first result, then return here to go deeper on each tool.

The tutorials at a glance — each tile is the headline figure of one tutorial;
click it to open that tutorial. The full, section-ordered list follows below.

.. raw:: html

   <style>
   .aa-tutorial-gallery{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin:1.6em 0;}
   .aa-tutorial-gallery a{display:flex;flex-direction:column;text-decoration:none;color:inherit;
     border:1px solid #dcdcdc;border-radius:6px;padding:7px;background:#fff;
     transition:border-color .15s ease,box-shadow .15s ease,transform .15s ease;}
   .aa-tutorial-gallery a:hover{border-color:#8a8a8a;box-shadow:0 3px 10px rgba(0,0,0,.13);transform:translateY(-2px);}
   .aa-tutorial-gallery img{width:100%;height:auto;border-radius:3px;}
   .aa-tutorial-gallery .cap{margin-top:6px;text-align:center;font-size:.82em;font-weight:600;line-height:1.2;}
   @media(max-width:900px){.aa-tutorial-gallery{grid-template-columns:repeat(3,1fr);}}
   @media(max-width:560px){.aa-tutorial-gallery{grid-template-columns:repeat(2,1fr);}}
   </style>
   <div class="aa-tutorial-gallery">
     <a href="generated/tutorial2a_data_loader.html"><img src="_static/img/thumbs/tut2a.png" alt="Load datasets"><div class="cap">Load datasets</div></a>
     <a href="generated/tutorial2b_scales_loader.html"><img src="_static/img/thumbs/tut2b.png" alt="Load amino acid scales"><div class="cap">Load scales</div></a>
     <a href="generated/tutorial3a_aaclust.html"><img src="_static/img/thumbs/tut3a.png" alt="AAclust scale clustering"><div class="cap">AAclust</div></a>
     <a href="generated/tutorial3b_sequence_feature.html"><img src="_static/img/thumbs/tut3b.png" alt="SequenceFeature"><div class="cap">SequenceFeature</div></a>
     <a href="generated/tutorial3c_cpp.html"><img src="_static/img/thumbs/tut3c.png" alt="CPP feature engineering"><div class="cap">CPP</div></a>
     <a href="generated/tutorial3d_data_representations.html"><img src="_static/img/thumbs/tut3d.png" alt="Data representations"><div class="cap">Data representations</div></a>
     <a href="generated/tutorial4a_dpulearn.html"><img src="_static/img/thumbs/tut4a.png" alt="dPULearn PU learning"><div class="cap">dPULearn</div></a>
     <a href="generated/tutorial5a_shap_model.html"><img src="_static/img/thumbs/tut5a.png" alt="ShapModel explanation"><div class="cap">ShapModel</div></a>
     <a href="generated/tutorial6_comparison_harness.html"><img src="_static/img/thumbs/tut6.png" alt="Evaluation and comparison"><div class="cap">Evaluation</div></a>
     <a href="generated/tutorial7_protein_engineering.html"><img src="_static/img/thumbs/tut7.png" alt="SeqOpt protein engineering"><div class="cap">Protein engineering</div></a>
   </div>

Data Handling
-------------
Learn how to load protein benchmarking datasets and amino acid scale sets in the **Data Loader** and **Scale Loader**  tutorials.

.. toctree::
   :maxdepth: 1

   generated/tutorial2a_data_loader
   generated/tutorial2b_scales_loader

Feature Engineering
-------------------
Explore interpretable feature engineering, the core of AAanalysis, with the :class:`~aaanalysis.AAclust`, :class:`~aaanalysis.SequenceFeature`,
and :class:`~aaanalysis.CPP` tutorials, then see how CPP turns different data representations (scales, embeddings, structure)
into features. Because :meth:`~aaanalysis.SequenceFeature.feature_matrix` returns a plain numeric matrix, these features
drop directly into a stock ``scikit-learn`` ``Pipeline`` — the prediction protocol demonstrates this end to end.

.. toctree::
   :maxdepth: 1

   generated/tutorial3a_aaclust
   generated/tutorial3b_sequence_feature
   generated/tutorial3c_cpp
   generated/tutorial3d_data_representations

PU Learning
-----------
Start positive-Unlabeled (PU) learning to tackle unbalanced and small data through our :class:`~aaanalysis.dPULearn` tutorial.

.. toctree::
   :maxdepth: 1

   generated/tutorial4a_dpulearn

Explainable AI
--------------
Explaining sample level predictions at single-residue resolution is introduced in our :class:`~aaanalysis.ShapModel` tutorial.

.. toctree::
   :maxdepth: 1

   generated/tutorial5a_shap_model

Evaluation & Comparison
-----------------------
Learn the evaluation tools — :class:`~aaanalysis.CPPGrid` configuration sweeps, per-protein
site-localization metrics, and fair ranking under cross-validation — in the
:class:`~aaanalysis.CPPGrid` tutorial. These are the mechanics that the *P10: Validation*
protocol puts to work end to end.

.. toctree::
   :maxdepth: 1

   generated/tutorial6_comparison_harness

Protein Engineering
-------------------
Optimize an existing sequence with :class:`~aaanalysis.SeqOpt` — machine-learning-guided directed
evolution — and read the results with :class:`~aaanalysis.SeqOptPlot`. This is **protein engineering**
(mutating a known protein), distinct from **de novo protein design** (generating new
proteins, e.g. RFdiffusion → ProteinMPNN → AlphaFold). The :class:`~aaanalysis.SeqOpt`
tutorial walks a complete case study: training a substrate classifier, engineering a
"super substrate" for gamma-secretase, and visualizing the Pareto front, convergence,
mutation map and lineage.

.. toctree::
   :maxdepth: 1

   generated/tutorial7_protein_engineering