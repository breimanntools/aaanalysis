.. _usage_principles_aaclust:

AAclust: Selecting Redundancy-Reduced Scale Sets
================================================

**AAclust (Amino Acid clustering)** is a clustering wrapper framework for selecting redundancy-reduced sets
of amino acid scales, introduced in [Breimann24a]_. Using Pearson correlation, AAclust optimizes the number of clusters
(*k*) and selects one representative scale per cluster, as illustrated in the figure below:


.. figure:: /_artwork/schemes/scheme_AAclust1.png
   :align: center
   :scale: 100%
   :alt: AAclust algorithm.

   Scheme of AAclust algorithm with clustering of amino acid scales and scale selection from [Breimann24a]_.

AAclust introduces two modes for defining the number of clusters (*k*):

    - *k*-optimized: AAclust automatically optimizes *k*, streamlining the scale selection process.
    - *k*-based: The user specifies *k*, allowing for custom configurations.

TThe distinctions between these modes and their respective AAclust options are depicted below:


.. figure:: /_artwork/schemes/scheme_AAclust2.png
   :align: center
   :scale: 75%
   :alt: AAclust options

   Operational modes of AAclust for determining the number of clusters, from [Breimann24a]_.

See our :doc:`AAclust Tutorial </generated/tutorial3a_aaclust>` for hands-on examples.
