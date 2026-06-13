.. _ecosystem:

The AAanalysis Ecosystem
========================
AAanalysis is the interpretable middle layer between bioinformatics I/O and the downstream machine
learning, explainable AI, and protein-design stack. It *consumes* upstream representations (sequences,
embeddings, structures) and even competitor descriptor sets, runs them through its interpretable core
(*Part × Split × Scale* · AAontology · CPP · ShapModel), and *exposes* the resulting features,
explanations, and objectives to the standard ML / XAI / optimization tools.

.. figure:: /_artwork/diagrams/aaanalysis_ecosystem.svg
   :alt: The AAanalysis ecosystem
   :width: 100%
   :align: center

   Where AAanalysis fits in the protein-ML stack — upstream complements feed the interpretable
   core, which exposes features, explanations, and objectives downstream.

An interactive version of this map (per-method maturity and adoption status) is available as a
`standalone page <../_static/aaanalysis_ecosystem.html>`_.
