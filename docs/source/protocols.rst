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
teaches you *one function* (what ``CPP`` or ``ShapModel`` does and how to call it).
A **protocol** teaches you a *workflow*: it answers a single biological question
from start to finish and, above all, builds the mental model for **when and why**
to reach for each tool. Where a protocol uses a function, it links to that
function's tutorial for the mechanics instead of repeating them, so the two stay
distinct with no overlap.

.. figure:: /_static/img/protocols_gallery.png
   :align: center
   :width: 100%
   :alt: Gallery of AAanalysis protocol outputs

   The protocols catalog at a glance: each tile is the headline figure of one
   protocol, from the CPP signature through validating a result.

**The mental model.** AAanalysis turns a biological *question* into an
*interpretable comparison*. You bring two (or more) groups of sequences; CPP reads
out the **signature** (the physicochemical features, resolved by position, that
distinguish them), and the rest of the pipeline helps you sample fairly, engineer
features, select what matters, predict, explain, and check that the signal is
real. The catalog follows that data flow, opening with the CPP signature; the
exploratory, no-label first look (P10) is there for when you do not yet have
groups to compare.

.. toctree::
   :maxdepth: 1

   generated/protocol1_cpp_signature
   generated/protocol2_prediction_tasks
   generated/protocol3_sampling
   generated/protocol4_engineer_features
   generated/protocol5_compositional_positional
   generated/protocol6_feature_selection
   generated/protocol7_classifier
   generated/protocol8_interpretability
   generated/protocol9_validation
   generated/protocol10_sequence_analysis
