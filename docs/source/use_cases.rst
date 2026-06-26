..
   Developer Notes:
   Use-case notebooks live in the top-level ``use_cases/`` folder (flat) and are
   auto-converted to ``generated/`` by ``create_notebooks_docs.py`` (called from
   ``conf.py``). A use case reproduces a published study end to end from bundled
   data; it must stay distinct from the per-function Tutorials and the
   single-question Protocols (no content overlap).
..


.. _use_cases:

Use Cases
=========

**Tutorials, protocols, and use cases are different things.** A
:ref:`tutorial <tutorials>` teaches you *one function*. A :ref:`protocol <protocols>`
teaches you a *workflow* that answers one biological question. A **use case** goes
one level up: it **reproduces a published study** end to end with AAanalysis, so you
can see that a real result drops out of the standard pipeline — and use it as the
template to adapt to your own paper-style analysis. Where a use case calls a tool,
it links to that tool's tutorial for the mechanics instead of repeating them, so the
three stay distinct with no overlap.

Each use case runs from **bundled data only** (no downloads) and is a simplified,
fast reproduction: it reproduces the *key results* and the *biology*, and points to
the Protocols for scaling up to the full study.

.. toctree::
   :maxdepth: 1

   generated/use_case1_gamma_secretase
