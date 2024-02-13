.. _usage_principles_aaclust:

AAclust: Selecting Redundancy-Reduced Scale Sets
================================================

AAclust is a clustering wrapper framework. introduced in [Breimann24a]_.

Taking any a set of amino acid scales as input, it estimates and optimizes the number of clusters to obtain a
redundancy-reduced scale set:

.. figure:: /_artwork/schemes/scheme_AAclust1.png

   Scheme of AAclust algorithm with clustering of amino acid scales and scale selection from [Breimann24a]_.

AAclust provides two selection modi: k-optimized and k-based selection:

.. figure:: /_artwork/schemes/scheme_AAclust2.png

   Two AAclust modi for number of clusters (k): k-optimized (AAclust optimized) and k-based (user-defined ),
   from [Breimann24a]_.

