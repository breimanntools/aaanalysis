..
   Developer Notes:
   Protocol notebooks live in the top-level ``protocols/`` folder (flat) and are
   auto-converted to ``generated/`` by ``create_notebooks_docs.py`` (called from
   ``conf.py``). The gallery image is regenerated from the protocols' own output
   figures. Protocols are workflows and must stay distinct from the per-function
   Tutorials (no content overlap).
..


.. _protocols:

Protocols
=========

**Tutorials and protocols are different things.** A :ref:`tutorial <tutorials>`
teaches you *one function* (what :class:`~aaanalysis.CPP` or :class:`~aaanalysis.ShapModel` does and how to call it).
A **protocol** teaches you a *workflow*: it answers a single biological question
from start to finish and, above all, builds the mental model for **when and why**
to reach for each tool. Where a protocol uses a function, it links to that
function's tutorial for the mechanics instead of repeating them, so the two stay
distinct with no overlap.

**New here?** The :ref:`Prediction tasks <prediction_tasks>` concept-overview page
is the front door: it maps a biological question to the right task — by *unit of
comparison* and *reference construction* — and points into the protocols below.

The protocols catalog at a glance — each tile is the headline figure of one
protocol; click it to open that protocol.

.. raw:: html

   <style>
   .aa-protocol-gallery{display:grid;grid-template-columns:repeat(5,1fr);gap:16px;margin:1.6em 0;}
   .aa-protocol-gallery a{display:flex;flex-direction:column;text-decoration:none;color:inherit;
     border:1px solid #dcdcdc;border-radius:6px;padding:8px;background:#fff;
     transition:border-color .15s ease,box-shadow .15s ease,transform .15s ease;}
   .aa-protocol-gallery a:hover{border-color:#8a8a8a;box-shadow:0 3px 10px rgba(0,0,0,.13);transform:translateY(-2px);}
   .aa-protocol-gallery img{width:100%;height:auto;border-radius:3px;}
   .aa-protocol-gallery .cap{margin-top:7px;text-align:center;font-size:.86em;font-weight:600;line-height:1.2;}
   @media(max-width:900px){.aa-protocol-gallery{grid-template-columns:repeat(3,1fr);}}
   @media(max-width:560px){.aa-protocol-gallery{grid-template-columns:repeat(2,1fr);}}
   </style>
   <div class="aa-protocol-gallery">
     <a href="generated/protocol1_cpp_signature.html"><img src="_static/img/thumbs/protocol1.png" alt="P1: CPP signature"><div class="cap">P1: CPP signature</div></a>
     <a href="generated/protocol2_sequence_analysis.html"><img src="_static/img/thumbs/protocol2.png" alt="P2: Exploratory sequence analysis"><div class="cap">P2: Exploratory sequence analysis</div></a>
     <a href="generated/protocol3_sampling.html"><img src="_static/img/thumbs/protocol3.png" alt="P3: Sampling"><div class="cap">P3: Sampling</div></a>
     <a href="generated/protocol4_prediction_tasks.html"><img src="_static/img/thumbs/protocol4.png" alt="P4: Prediction levels"><div class="cap">P4: Prediction levels</div></a>
     <a href="generated/protocol5_engineer_features.html"><img src="_static/img/thumbs/protocol5.png" alt="P5: Engineer features"><div class="cap">P5: Engineer features</div></a>
     <a href="generated/protocol6_compositional_positional.html"><img src="_static/img/thumbs/protocol6.png" alt="P6: Compositional vs positional"><div class="cap">P6: Compositional vs positional</div></a>
     <a href="generated/protocol7_feature_selection.html"><img src="_static/img/thumbs/protocol7.png" alt="P7: Select &amp; reduce features"><div class="cap">P7: Select &amp; reduce features</div></a>
     <a href="generated/protocol8_prediction.html"><img src="_static/img/thumbs/protocol8.png" alt="P8: Prediction"><div class="cap">P8: Prediction</div></a>
     <a href="generated/protocol9_interpretability.html"><img src="_static/img/thumbs/protocol9.png" alt="P9: Interpretability"><div class="cap">P9: Interpretability</div></a>
     <a href="generated/protocol10_validation.html"><img src="_static/img/thumbs/protocol10.png" alt="P10: Validation"><div class="cap">P10: Validation</div></a>
   </div>

AAanalysis turns a biological *question* into an
*interpretable comparison*. You bring two (or more) groups of sequences; CPP reads
out the **signature** (the physicochemical features, resolved by position, that
distinguish them), and the rest of the pipeline helps you sample fairly, engineer
features, select what matters, predict, explain, and check that the signal is
real. The catalog follows that data flow, opening with the CPP signature, then an
exploratory no-label first look, and on through sampling, feature engineering,
selection, modelling, explanation, and validation.

.. toctree::
   :maxdepth: 1
   :hidden:

   generated/protocol1_cpp_signature
   generated/protocol2_sequence_analysis
   generated/protocol3_sampling
   generated/protocol4_prediction_tasks
   generated/protocol5_engineer_features
   generated/protocol6_compositional_positional
   generated/protocol7_feature_selection
   generated/protocol8_prediction
   generated/protocol9_interpretability
   generated/protocol10_validation
